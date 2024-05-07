from ..Agent.rnn import RNNAgent
from .action_selector import EpsilonGreedyActionSelector
import torch as th

class RecurrentMAC:

    def __init__(self,
                 scheme, 
                 config):
        
        self.n_agents = scheme.n_agents
        self.n_actions = scheme.n_actions

        self.scheme = scheme
        
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape, config.agent)

        self.action_selector = EpsilonGreedyActionSelector(config.epsilon_schedule)
        self.hidden_states = None

    def select_actions(self, batch, t, t_env, test=False):
        avail_actions = batch["action_mask"]
        avail_actions = avail_actions[:, t]
        agent_outputs = self.forward(batch, t)
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, t_env, test=test)
        return chosen_actions.squeeze(0)

    def forward(self, batch, t):
        agent_inputs = self._build_inputs(batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs.view(batch.batch_size, self.n_agents, -1)

    def setup(self, **kwargs):
        batch_size = kwargs['batch_size']
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
    
    def _build_agents(self, input_shape, args):
        self.agent = RNNAgent(input_shape, self.n_actions, args=args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if t == 0: inputs.append(th.zeros(bs, self.n_agents, self.n_actions))
        else: inputs.append(batch["action_onehot"][:, t-1])
        inputs.append(th.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    
    def _get_input_shape(self, scheme):
        input_shape = scheme.obs_shape
        return input_shape + self.n_actions + self.n_agents