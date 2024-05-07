import os
import yaml
from ...QMix.utils.config import AttributeDict
from functools import partial
import torch as th 

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers.QMixWrapper import QMixWrapper


from ...QMix.runner import EpisodeRunner
from ...QMix.buffer import ReplayBuffer
from ...QMix.ActionSelector.rnn_controller import RecurrentMAC
from ...QMix.ActionSelector.mlp_controller import MLPMAC
from ...QMix.q_learner import QLearner
from ...QMix.utils.logger import SimpleLogger

def load_env(**kwargs):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    # cyborg = BlueFlatWrapper(cyborg, **kwargs)
    cyborg = QMixWrapper(cyborg, **kwargs)
    return cyborg

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
class QMixTrainer:

    def __init__(self, 
                 env_fn,
                 logger,
                 scheme,
                 config
                ):

        self.env_fn = env_fn
        self.config = config
        self.timesteps = config.timesteps
        self.batch_size = config.batch_size
        self.logger = logger
        self.controller = MLPMAC(scheme, config)
        self.runner = EpisodeRunner(env_fn, scheme)
        self.runner.setup(self.controller)
        self.buffer = ReplayBuffer(config)
        self.learner = QLearner(self.controller, self.logger, scheme, config)
        self.learn_freq = config.learn_freq

    def train(self):

        self.t = episode = 0
        res = []

        while self.t < self.timesteps:
            print(episode)
            
            rollout, self.t = self.runner.run(t_env=self.t, test= False)
            rewards = rollout['reward'].squeeze(0)
            discount = self.config.gamma ** th.arange(rewards.size(0)).unsqueeze(1) 
            returns = (rewards * discount).sum().item()
            self.buffer.insert_episode(rollout)

            if episode % self.learn_freq == 0 and self.buffer.can_sample(self.batch_size):
                batch = self.buffer.sample(self.batch_size)
                self.learner.train(batch, self.t, episode)
            
            episode += 1
            res.append(returns)

        if self.config.save_model and \
            self.t >= self.timesteps:
            # self.t % (self.timesteps // self.config.save_count) == 0:
            path = os.path.join("res", "models", self.config.token, str(self.t))
            os.makedirs(path, exist_ok=True)
            self.logger.info("Saving models to {}".format(path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            self.learner.save_models(path)
        
        return res