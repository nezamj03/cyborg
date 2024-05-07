import copy
from .buffer import EpisodeBatch
from .Mixer.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, controller, logger, scheme, config):
        self.config = config
        self.scheme = scheme
        self.controller = controller
        self.logger = logger
        self.mixer = QMixer(scheme, config.mixer)

        self.params = list(controller.parameters()) + list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(
            params=self.params,
            lr=config.lr,
            alpha=config.alpha,
            eps=config.eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any controller
        self.target_controller = copy.deepcopy(controller)
        
        self.sync_freq = config.sync_freq
        self.log_freq = config.log_freq

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["action"][:, :-1].long()
        terminated = batch["done"][:, :-1].float()
        avail_actions = batch["action_mask"]

        # Calculate estimated Q-Values
        controller_out = []
        self.controller.setup(batch_size=batch.batch_size) # make initial recurrent part 0
        for t in range(batch.episode_length()):
            agent_outs = self.controller.forward(batch, t=t)
            controller_out.append(agent_outs)
        controller_out = th.stack(controller_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(controller_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_controller_out = []
        self.target_controller.setup(batch_size= batch.batch_size)
        for t in range(batch.seq_length):
            target_agent_outs = self.target_controller.forward(batch, t=t)
            target_controller_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_controller_out = th.stack(target_controller_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_controller_out[avail_actions[:, 1:] == 0] = -float('inf')

        # Max over target Q-Values
        target_max_qvals = target_controller_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.config.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).sum() / th.ones_like(rewards).sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.config.grad_norm_clip)
        self.optimiser.step()

        if episode_num > 0 and episode_num % self.sync_freq == 0:
            self._update_targets()

        if t_env > 0 and t_env % self.log_freq == 0:
            self.logger.stat("loss", loss.item(), t_env)
            self.logger.stat("grad_norm", grad_norm, t_env)
            self.logger.stat("td_error_abs", (td_error.abs().sum().item()), t_env)
            self.logger.stat("q_taken_mean", chosen_action_qvals.sum().item()/self.scheme.n_agents, t_env)
            self.logger.stat("target_mean", targets.sum().item()/self.scheme.n_agents, t_env)

    def _update_targets(self):
        self.target_controller.load_state(self.controller)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.info("Updated target network")

    def save_models(self, path):
        self.controller.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.controller.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_controller.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))