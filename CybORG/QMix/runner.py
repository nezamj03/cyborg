from functools import partial
from .buffer import Episode
import numpy as np

class EpisodeRunner:

    def __init__(self, env_fn, scheme):
        self.scheme = scheme
        self.env_fn = env_fn
        self.env = self.env_fn()
        self.t = 0

    def setup(self, controller):
        self.new_episode = partial(Episode, self.scheme)
        self.controller = controller

    def close(self):
        self.env.close()

    def reset(self):
        self.episode = self.new_episode()
        obs, info = self.env.reset()
        self.t = 0
        return obs, info

    def run(self, t_env, test=False):

        obs, info = self.reset()

        done = False
        self.controller.setup(batch_size=1)

        while not done:

            pre_transition_data = {
                "state": info['state'],
                "action_mask": info['action_mask'],
                "obs": obs
            }

            self.episode.update(pre_transition_data)

            actions = self.controller.select_actions(self.episode.to_batch(), t=self.t, t_env=t_env, test=test)
            obs, reward, terminated, truncated, info = self.env.step(actions.squeeze(0), messages={})
            done = (terminated | truncated).any()

            post_transition_data = {
                "action": actions,
                "reward": reward,
                "done": terminated | truncated,
            }

            self.episode.update(post_transition_data)

            self.t += 1
            t_env += 1

        return self.episode, t_env
        # return self.episode, t_env