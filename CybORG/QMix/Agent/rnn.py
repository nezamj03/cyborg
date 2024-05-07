import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):

    def __init__(self, input_shape, n_actions, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.hidden_shape = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.hidden_shape)
        self.rnn = nn.GRUCell(self.hidden_shape, self.hidden_shape)
        self.fc2 = nn.Linear(self.hidden_shape, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_shape).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_shape)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
