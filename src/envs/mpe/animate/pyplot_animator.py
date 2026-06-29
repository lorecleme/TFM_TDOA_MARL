import numpy as np
import io
import logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import imageio


class MPEAnimator:

    def __init__(self,
                 agent_positions,
                 landmark_positions,
                 episode_rewards,
                 mask_agents=False,
                 episode_surround=None,
                 episode_min_dist=None):

        # general parameters
        self.frames = (agent_positions.shape[1])
        self.n_agents = len(agent_positions)
        self.n_landmarks = len(landmark_positions)
        self.lags = self.frames

        self.agent_positions = agent_positions
        self.landmark_positions = landmark_positions
        self.episode_rewards = episode_rewards
        self.episode_surround = episode_surround
        self.episode_min_dist = episode_min_dist

        # create the subplots
        self.fig = plt.figure(figsize=(20, 14), dpi=120)
        self.ax_episode = self.fig.add_subplot(2, 2, 1)
        self.ax_reward = self.fig.add_subplot(2, 2, 2)
        self.ax_surround = self.fig.add_subplot(2, 2, 3)
        self.ax_min_dist = self.fig.add_subplot(2, 2, 4)

        self.ax_episode.set_title('Episode')
        self.ax_reward.set_title('Reward')
        self.ax_surround.set_title('Surround quality')
        self.ax_min_dist.set_title('Min distance to target')

        # colors
        self.agent_colors = cm.Dark2.colors
        self.landmark_colors = [cm.summer(l*10) for l in range(self.n_landmarks)]

        # init the lines (pre-draw empty)
        self.lines_episode = self._init_episode_animation(self.ax_episode)
        self.lines_reward  = self._init_reward_animation(self.ax_reward)
        self.lines_surround = self._init_line_animation(self.ax_surround, 'Surround quality', 0, 1)
        self.lines_min_dist = self._init_line_animation(self.ax_min_dist, 'Min dist (m)', 0, None)

    def save_animation(self, savepath='episode'):
        if self.frames == 0:
            return

        gif_frames = []
        for frame in range(self.frames):
            self._draw_frame(frame)
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            img = plt.imread(buf)
            buf.close()
            gif_frames.append((img[:,:,:3] * 255).astype(np.uint8))

        try:
            imageio.mimsave(savepath + '.gif', gif_frames, fps=10)
            self.fig.savefig(savepath + '.png')
        except Exception as e:
            print(f"[anim] save error (non-fatal): {e}")

        plt.close(self.fig)

    def _episode_update(self, data, line, frame, lags, name=None):
        line.set_data(data[max(0,frame-lags):frame, 0], data[max(0,frame-lags):frame, 1])
        if name is not None:
            line.set_label(name)

    def _frameline_update(self, data, line, frame, name=None):
        line.set_data(np.arange(1,frame+1), data[:frame])
        if name is not None:
            line.set_label(name)

    def _draw_frame(self, frame):

        # Update the episode subplot
        line_episode = 0
        # update agents heads
        for n in range(self.n_agents):
            self._episode_update(self.agent_positions[n], self.lines_episode[line_episode], frame, 1, f'Agent_{n+1}')
            line_episode += 1

        # update agents trajectories
        for n in range(self.n_agents):
            self._episode_update(self.agent_positions[n], self.lines_episode[line_episode], max(0,frame-1), self.lags)
            line_episode += 1

        # landmark head (current position only)
        for n in range(self.n_landmarks):
            self._episode_update(self.landmark_positions[n], self.lines_episode[line_episode], frame, 1, f'Target_{n+1}')
            line_episode += 1

        # landmark trajectory (faded past positions)
        for n in range(self.n_landmarks):
            self._episode_update(self.landmark_positions[n], self.lines_episode[line_episode], max(0,frame-1), self.lags)
            line_episode += 1

        self.ax_episode.legend()

        # Update the reward subplot
        self._frameline_update(self.episode_rewards, self.lines_reward[0], frame)

        # Update surround quality
        if self.episode_surround is not None:
            self._frameline_update(self.episode_surround, self.lines_surround[0], frame)

        # Update min distance
        if self.episode_min_dist is not None:
            self._frameline_update(self.episode_min_dist, self.lines_min_dist[0], frame)

    def _init_episode_animation(self, ax):
        x_max = max(self.agent_positions[:,:,0].max(),
                    self.landmark_positions[:,:,0].max())
        x_min = min(self.agent_positions[:,:,0].min(),
                    self.landmark_positions[:,:,0].min())
        y_max = max(self.agent_positions[:,:,1].max(),
                    self.landmark_positions[:,:,1].max())
        y_min = min(self.agent_positions[:,:,1].min(),
                    self.landmark_positions[:,:,1].min())
        abs_min = min(x_min, y_min)
        abs_max = max(x_max, y_max)

        ax.set_xlim(abs_min-1, abs_max+1)
        ax.set_ylim(abs_min-1,abs_max+1)
        ax.set_ylabel('Y Position')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        lines = [ax.plot([],[],'o',color=self.agent_colors[a], alpha=0.8,markersize=8)[0] for a in range(self.n_agents)] + \
                [ax.plot([],[],'o',color=self.agent_colors[a], alpha=0.2,markersize=4)[0] for a in range(self.n_agents)] + \
                [ax.plot([],[],'s',color=self.landmark_colors[l], alpha=0.8,markersize=8)[0] for l in range(self.n_landmarks)] + \
                [ax.plot([],[],'s',color=self.landmark_colors[l], alpha=0.2,markersize=4)[0] for l in range(self.n_landmarks)]

        return lines

    def _init_reward_animation(self, ax):
        ax.set_xlim(0, self.frames)
        ax.set_ylim(self.episode_rewards.min(), self.episode_rewards.max()+1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        return [ax.plot([],[], color='green')[0]]

    def _init_line_animation(self, ax, ylabel, ymin, ymax):
        ax.set_xlim(0, self.frames)
        if ymax is None:
            ax.set_ylim(ymin - 0.1, 1.1)
        else:
            ax.set_ylim(ymin - 0.1, ymax + 0.1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        return [ax.plot([], [], color='blue')[0]]
