import torch
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule


class EpsilonGreedyActionSelector:

    """
    Action selector for agents according to an ε-policy-greedy decision rule
    """
    def __init__(self, args):
        self.schedule = DecayThenFlatSchedule(
            args.start,
            args.finish,
            args.anneal_time,
            decay='linear'
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t, test=False):

        self.epsilon = self.schedule.eval(t)
        if test: self.epsilon = 0.0

        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions