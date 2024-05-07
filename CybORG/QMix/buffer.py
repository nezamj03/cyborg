import numpy as np
import torch as th
import torch.nn.functional as F

class Episode:

    def __init__(self, scheme):
    
        self.batch = None
        self.n_actions = scheme.n_actions
        self.batch_size = 1
        self.episode = {
            "state": [],
            "action_mask": [],
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "action_onehot": []
        }

    def update(self, experience):

        for k, v in experience.items():
            if k in self.episode:
                self.episode[k].append(v)
            if k == "action":
                self.episode["action_onehot"].append(F.one_hot(v, num_classes=self.n_actions))

    def __getitem__(self, item):
        return th.tensor(np.array(self.episode[item]))
    
    def to_batch(self):
        if self.batch is None:
            self.batch = EpisodeBatch([self])
        return self.batch
                
class ReplayBuffer:

    def __init__(self, config):
        self.buffer_size = config.buffer_size
        self.buffer_index = 0
        self.buffer_length = 0

        self.replay = np.full(shape=self.buffer_size, fill_value=None, dtype=object)
    
    def insert_episode(self, episode : Episode):

        if self.buffer_index == self.buffer_size - 1:
            np.roll(self.replay, shift=1)
            self.replay[0] = episode
        else:
            self.replay[-self.buffer_index - 1] = episode
            self.buffer_index += 1

    def can_sample(self, batch_size):
        return self.buffer_index >= batch_size
    
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if len(self.replay) == batch_size:
            return EpisodeBatch(self.replay)
        else:
            episode_ids = -np.random.choice(self.buffer_index, batch_size, replace=False) - 1
            return EpisodeBatch(self.replay[episode_ids])

class EpisodeBatch:

    def __init__(self, episodes):
        self.episodes = episodes
        self.batch_size = len(episodes)
        self.seq_length = len(np.random.choice(episodes)['state'])
    
    def __getitem__(self, item):
        return th.from_numpy(np.stack([episode[item] for episode in self.episodes], axis=0)).float()
    
    def episode_length(self):
        return len(np.random.choice(self.episodes)['state'])
