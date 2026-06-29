#!/usr/bin/env python3
"""
TransfQMix Inference Node — loads a trained Transformer agent checkpoint and
runs greedy action selection at 10 Hz for 3-agent cooperative TDOA tracking.

Architecture is BYTE-IDENTICAL to the training codebase:
  tdoa_tracking/src/modules/layer/transformer.py
  tdoa_tracking/src/modules/agents/n_transf_agent.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32MultiArray


# ═══════════════════════════════════════════════════════════════
# Training architecture — EXACT copies (byte-identical to source)
# ═══════════════════════════════════════════════════════════════

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, q, k, mask):
        h = self.heads
        b_q, t_q, e_q = q.size()
        b, t_k, e = k.size()
        assert b == b_q and e == e_q

        keys = self.tokeys(k).view(b, t_k, h, e)
        values = self.tovalues(k).view(b, t_k, h, e)
        queries = self.toqueries(q).view(b, t_q, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t_k, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t_k, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_q, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t_q, t_k)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t_q, e)
        out = out.transpose(1, 2).contiguous().view(b, t_q, h * e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(emb, heads=heads, mask=mask)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, q_k_mask):
        q, k, mask = q_k_mask
        attended = self.attention(q, k, mask)
        x = self.norm1(attended + q)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x, k, mask


class Transformer(nn.Module):
    def __init__(self, emb, heads, depth, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False,
                                 ff_hidden_mult=ff_hidden_mult, dropout=dropout))
        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, q, k, mask=None):
        x, k, mask = self.tblocks((q, k, mask))
        return x


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = getattr(args, "n_entities_obs", args.n_entities)
        self.feat_dim = args.obs_entity_feats
        self.emb_dim = args.emb

        self.feat_embedding = nn.Linear(self.feat_dim, self.emb_dim)
        self.transformer = Transformer(
            args.emb, args.heads, args.depth,
            args.ff_hidden_mult, args.dropout
        )
        self.q_basic = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        return torch.zeros(1, self.emb_dim)

    def forward(self, inputs, hidden_state):
        b, a, _ = inputs.size()
        n_entities = inputs.shape[-1] // self.feat_dim
        inputs = inputs.view(-1, n_entities, self.feat_dim)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)

        embs = self.feat_embedding(inputs)
        x = torch.cat((hidden_state, embs), 1)
        embs = self.transformer.forward(x, x)
        h = embs[:, 0:1, :]
        q = self.q_basic(h)
        return q.view(b, a, -1), h.view(b, a, -1)


# ═══════════════════════════════════════════════════════════════
# Agent arguments (matching transf_qmix.yaml)
# ═══════════════════════════════════════════════════════════════

class AgentArgs:
    n_agents = 3
    n_entities = 4          # 3 agents + 1 target
    obs_entity_feats = 6    # 5 + n_landmarks
    emb = 32
    heads = 4
    depth = 2
    n_actions = 5
    ff_hidden_mult = 4
    dropout = 0.0
    device = 'cpu'


# ═══════════════════════════════════════════════════════════════
# ROS 2 Inference Node
# ═══════════════════════════════════════════════════════════════

class TransfQMixInferenceNode(Node):
    """Loads Transformer agent checkpoint and runs inference at 10 Hz."""

    def __init__(self):
        super().__init__('transf_qmix_inference_node')

        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('hidden_reset_interval', 100)

        checkpoint_path = self.get_parameter('checkpoint_path').value
        self.rate_hz = self.get_parameter('rate_hz').value
        self.hidden_reset_interval = int(self.get_parameter('hidden_reset_interval').value)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self.get_logger().error(f'Checkpoint not found: {checkpoint_path}')
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        # Build agent
        self.args = AgentArgs()
        self.agent = TransformerAgent(None, self.args)
        self.agent.eval()

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.agent.load_state_dict(state_dict)
        self.get_logger().info(f'Loaded agent checkpoint from {checkpoint_path}')

        # Hidden states (one per agent, re-initialised each step via init_hidden)
        self.hidden_states = [self.agent.init_hidden() for _ in range(self.args.n_agents)]

        # Subscriber: /tdoa/entity_obs (72 floats = 3 agents * 4 entities * 6 features)
        self.obs_sub = self.create_subscription(
            Float64MultiArray, '/tdoa/entity_obs',
            self._obs_callback, 10
        )

        # Publisher: /tdoa/actions (3 integers)
        self.action_pub = self.create_publisher(
            Int32MultiArray, '/tdoa/actions', 10
        )

        self.get_logger().info(
            f'TransfQMix Inference Node ready: {self.args.n_agents} agents, '
            f'{self.args.n_actions} actions, {self.rate_hz}Hz'
        )

    def _obs_callback(self, msg: Float64MultiArray):
        """Receive entity observations, run inference, publish actions."""
        obs_data = np.array(msg.data, dtype=np.float32)
        n_agents = self.args.n_agents
        n_entities = self.args.n_entities
        feat_dim = self.args.obs_entity_feats

        # Reshape flat 72 to (n_agents, n_entities, feat_dim) = (3, 4, 6)
        # Then add batch dim: (1, n_agents, n_entities*feat_dim) = (1, 3, 24)
        obs_batch = obs_data.reshape(n_agents, n_entities * feat_dim)
        obs_tensor = torch.from_numpy(obs_batch).unsqueeze(0)  # (1, 3, 24)

        # Stack hidden states: (1, 3, emb_dim)
        h_batch = torch.stack([h.unsqueeze(0) for h in self.hidden_states], dim=1)

        with torch.no_grad():
            q_values, new_h = self.agent(obs_tensor, h_batch)

        # Greedy action selection
        q_squeezed = q_values.squeeze(0)  # (3, 5)
        actions = q_squeezed.argmax(dim=-1).tolist()  # [a1, a2, a3]

        # Debug: log Q-values every 50 calls (~every 5 seconds)
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1

        # Reset hidden state at episode boundaries (mirrors training reset)
        if self.hidden_reset_interval > 0 and self._call_count % self.hidden_reset_interval == 0:
            self.hidden_states = [self.agent.init_hidden() for _ in range(self.args.n_agents)]
            if self._call_count <= self.hidden_reset_interval:
                pass  # first reset is a no-op (already zeros)
            else:
                self.get_logger().info(f'Hidden state reset at step {self._call_count}')

        if self._call_count == 1:
            obs_reshaped = obs_data.reshape(n_agents, n_entities, feat_dim)
            self.get_logger().info(
                f'[DBG] first obs received:\n'
                f'  Agent0 landmark row: rel_x={obs_reshaped[0,3,0]:.5f} rel_y={obs_reshaped[0,3,1]:.5f} range={obs_reshaped[0,3,4]:.5f}\n'
                f'  Agent1 landmark row: rel_x={obs_reshaped[1,3,0]:.5f} rel_y={obs_reshaped[1,3,1]:.5f} range={obs_reshaped[1,3,4]:.5f}\n'
                f'  Agent2 landmark row: rel_x={obs_reshaped[2,3,0]:.5f} rel_y={obs_reshaped[2,3,1]:.5f} range={obs_reshaped[2,3,4]:.5f}\n'
                f'  Agent0 agent-row ranges: [{obs_reshaped[0,0,4]:.5f} {obs_reshaped[0,1,4]:.5f} {obs_reshaped[0,2,4]:.5f}]\n'
                f'  Hidden state norm: {h_batch.norm().item():.5f}'
            )
        if self._call_count % 50 == 1:
            q_np = q_squeezed.numpy()
            self.get_logger().info(
                f'Q-values:\n'
                f'  Agent1: {q_np[0]} -> action {actions[0]}\n'
                f'  Agent2: {q_np[1]} -> action {actions[1]}\n'
                f'  Agent3: {q_np[2]} -> action {actions[2]}'
            )

        # Update hidden states
        new_h = new_h.squeeze(0)  # (3, emb_dim)
        for i in range(n_agents):
            self.hidden_states[i] = new_h[i:i+1]  # (1, emb_dim)

        # Publish
        out = Int32MultiArray()
        out.data = actions
        self.action_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TransfQMixInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
