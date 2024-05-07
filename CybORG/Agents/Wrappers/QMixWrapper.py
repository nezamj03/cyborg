from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Simulator.Actions import Action
from typing import Any
import numpy as np

class QMixWrapper(BlueFlatWrapper):

    def _get_info(self, observations, info):
        res = {}
        res['state'] = np.concatenate(tuple(observations.values()))
        res['action_mask'] = np.array([v['action_mask'] for v in info.values()])
        return res

    def _remove_keys(self, dict):
        return np.array(list(dict.values()))

    def reset(self, *args, **kwargs):
        observations, info = super().reset(*args, **kwargs)
        return self._remove_keys(observations), self._get_info(observations, info)

    def step(
        self,
        actions,
        messages,
        **kwargs,
    ):
        
        action_dict = {agent : actions[i] for i, agent in enumerate(self.agents)}
        observations, rewards, terminated, truncated, info = super().step(
            actions=action_dict, messages=messages, **kwargs
        )
        return (self._remove_keys(observations), 
                self._remove_keys(rewards), 
                self._remove_keys(terminated), 
                self._remove_keys(truncated), 
                self._get_info(observations, info))

    