import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.n_transf_mixer import TransformerMixer
from envs.matrix_game import print_matrix_status_transf
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num

class NQTransfLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device(getattr(args, "device", "cuda" if args.use_cuda else "cpu"))
        self.params = list(mac.parameters())

        if args.mixer == "transf_mixer":
            self.mixer = TransformerMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # LR warmup: linearly ramp LR from 0 to args.lr over warmup_steps gradient updates
        self.warmup_steps = getattr(self.args, 'lr_warmup_steps', 0)
        self.train_t = 0
        if self.warmup_steps > 0:
            self._set_lr(0.0)
            self.logger.console_logger.info(
                f"LR warmup enabled: {self.warmup_steps} steps (0 → {self.args.lr})"
            )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

        # mixed precision (AMP) — speeds up forward/backward, same gradients
        self.use_amp = getattr(self.args, 'use_amp', False)
        self.scaler = th.cuda.amp.GradScaler(enabled=self.use_amp)

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def _set_lr(self, lr):
        """Set the learning rate for all parameter groups."""
        for pg in self.optimiser.param_groups:
            pg['lr'] = lr
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        # initialize the q-values of the agents
        mac_out = th.zeros(
            batch.batch_size,
            batch.max_seq_length,
            self.args.n_agents,
            self.args.n_actions
        ).to(self.args.device)
        # initialize the hidden_sates of the agents
        mac_hs = th.zeros(
            batch.batch_size,
            batch.max_seq_length,
            self.args.n_agents,
            self.args.emb # embedding dimension        
        ).to(self.args.device)
        self.mac.init_hidden(batch.batch_size)
        with th.cuda.amp.autocast(enabled=self.use_amp):
            for t in range(batch.max_seq_length):
                agent_outs, hidden_states = self.mac.forward(batch, t=t, return_hs=True) # (batch_size, n_agents, n_actions)
                mac_out[:, t, :, :] = agent_outs
                mac_hs[:, t, :, :] = hidden_states

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_ = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            # initialize the q-values of the agents
            target_mac_out = th.zeros(
                batch.batch_size,
                batch.max_seq_length,
                self.args.n_agents,
                self.args.n_actions
            ).to(self.args.device)
            # initialize the hidden_sates of the agents
            target_mac_hs = th.zeros(
                batch.batch_size,
                batch.max_seq_length,
                self.args.n_agents,
                self.args.emb # embedding dimension        
            ).to(self.args.device)
            target_hidden_states = self.target_mac.init_hidden(batch.batch_size)
            with th.cuda.amp.autocast(enabled=self.use_amp):
                for t in range(batch.max_seq_length):
                    target_agent_outs, target_hidden_states = self.target_mac.forward(batch, t=t, return_hs=True)
                    target_mac_out[:, t] = target_agent_outs
                    target_mac_hs[:, t] = target_hidden_states

            # Max over target Q-Values/ Double q learning -> consider only the qvals of the chosen actions
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals_ = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3) # (batch_size, max_seq_length, n_agents)
            
            # Calculate n-step Q-Learning targets
            hyper_weights = self.target_mixer.init_hidden().expand(batch.batch_size, self.args.n_agents, -1)
            target_max_qvals = th.zeros(batch.batch_size, batch.max_seq_length, 1).to(self.args.device)
            with th.cuda.amp.autocast(enabled=self.use_amp):
                for t in range(batch.max_seq_length):
                    target_mixer_out, hyper_weights = self.target_mixer(
                        target_max_qvals_[:, t].view(-1, 1, self.args.n_agents), # (batch, 1, n_agents)
                        target_mac_hs[:, t],
                        hyper_weights,
                        batch["state"][:, t],
                        batch["obs"][:, t]
                    )
                    target_max_qvals[:, t] = target_mixer_out.squeeze(-1)
            
            #target_max_qvals = self.target_mixer(target_max_qvals, batch["state"], batch["obs"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"], batch["obs"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        hyper_weights = self.mixer.init_hidden().expand(batch.batch_size, self.args.n_agents, -1)
        chosen_action_qvals = th.zeros(batch.batch_size, batch.max_seq_length-1, 1).to(self.args.device)
        with th.cuda.amp.autocast(enabled=self.use_amp):
            for t in range(batch.max_seq_length - 1):
                mixer_out, hyper_weights = self.mixer(
                    chosen_action_qvals_[:, t].view(-1, 1, self.args.n_agents),
                    mac_hs[:, t,].detach(),
                    hyper_weights,
                    batch["state"][:, t],
                    batch["obs"][:, t])
                chosen_action_qvals[:, t] = mixer_out.squeeze(-1)
        #chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], batch["obs"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise (with AMP gradient scaling when enabled)
        self.optimiser.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimiser)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.scaler.step(self.optimiser)
        self.scaler.update()

        # LR warmup: linearly ramp from 0 to args.lr over warmup_steps gradient updates
        self.train_t += 1
        if self.warmup_steps > 0 and self.train_t <= self.warmup_steps:
            scale = self.train_t / self.warmup_steps
            self._set_lr(self.args.lr * scale)
        elif self.warmup_steps > 0 and self.train_t == self.warmup_steps + 1:
            self._set_lr(self.args.lr)
            self.logger.console_logger.info(
                f"LR warmup complete. LR set to {self.args.lr} (step {self.train_t})"
            )

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status_transf(batch, self.mixer, mac_out, mac_hs)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def to(self, device):
        self.device = th.device(device)
        self.mac.to(device)
        self.target_mac.to(device)
        if self.mixer is not None:
            self.mixer.to(device)
            self.target_mixer.to(device)
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        expanded = False
        expanded |= self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            mixer_state = th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            try:
                self.mixer.load_state_dict(mixer_state)
            except RuntimeError as e:
                if 'size mismatch for feat_embedding.weight' in str(e):
                    old_shape = mixer_state['feat_embedding.weight'].shape
                    new_shape = self.mixer.feat_embedding.weight.shape
                    if old_shape[1] < new_shape[1]:
                        print(f"Expanding mixer feat_embedding from {old_shape} to {new_shape}")
                        new_linear = th.nn.Linear(new_shape[1], new_shape[0])
                        new_linear.weight.data[:, :old_shape[1]] = mixer_state['feat_embedding.weight']
                        new_linear.weight.data[:, old_shape[1]:] = 0.0
                    else:
                        print(f"Shrinking mixer feat_embedding from {old_shape} to {new_shape}")
                        new_linear = th.nn.Linear(new_shape[1], new_shape[0])
                        new_linear.weight.data[:, :] = mixer_state['feat_embedding.weight'][:, :new_shape[1]]
                    expanded = True
                    if 'feat_embedding.bias' in mixer_state:
                        new_linear.bias.data = mixer_state['feat_embedding.bias']
                    mixer_state['feat_embedding.weight'] = new_linear.weight.data
                    if 'feat_embedding.bias' in mixer_state:
                        mixer_state['feat_embedding.bias'] = new_linear.bias.data
                    self.mixer.load_state_dict(mixer_state)
                else:
                    raise
        if expanded:
            print("Embedding was expanded — skipping optimizer state (Adam will re-init moments)")
        else:
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
