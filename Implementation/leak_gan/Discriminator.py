""" Discriminator Module of the leak-gan """
import torch as th


class LSTMDiscriminator(th.nn.Module):
    """ LSTM discriminator (Also leaks features as mentioned in the paper) """

    def __init__(self, input_size, hidden_size, num_layers):
        """
        constructor for the LSTM discriminator
        :param input_size: num_features in the input
        :param hidden_size: hidden size of the LSTM
        :param num_layers: number of LSTM layers
        """
        from torch.nn import LSTM, Linear

        super(LSTMDiscriminator, self).__init__()

        # create the state
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create the RNN (LSTM layer)
        self.rnn = LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            batch_first=True,
        )

        # Output linear layer
        self.decision_maker = Linear(self.hidden_size, 1, bias=False)

    def forward(self, x, prev_states=None):
        """
        forward pass of the Discriminator
        :param x: input tensor of shape [batch_size x sequence_len x feature_dim]
        :param prev_states: (h_prev, c_prev) of shape [num_layers x batch_size x hidden_size]
        :return: preds, feats, f_states
                    => predictions, leaked features and final_states (for inference)
        """
        from torch.nn.functional import sigmoid

        features, final_states = self.rnn(x) if prev_states is None else self.rnn(x, prev_states)

        # project the output to get the predictions
        predictions = sigmoid(self.decision_maker(features))

        return predictions, features, final_states


class CNNDiscriminator(th.nn.Module):
    """ Convolutional Discriminator (leaks features) """

    def __init__(self, inp_size):

        from torch.nn import Sequential, Conv1d, \
             BatchNorm1d, LeakyReLU, Linear

        super(CNNDiscriminator, self).__init__()

        # state:
        self.inp_size = inp_size

        # define the computation module:
        self.model = Sequential(
            # block 1
            Conv1d(self.inp_size, 32, 3, padding=1),
            Conv1d(32, 64, 3, padding=1),
            BatchNorm1d(64),
            LeakyReLU(0.2),

            # block 2
            Conv1d(64, 128, 3, padding=1),
            Conv1d(128, 256, 3, padding=1),
            BatchNorm1d(256),
            LeakyReLU(0.2),

            # block 3
            Conv1d(256, 512, 3, padding=1),
            Conv1d(512, 512, 3, padding=1),
            BatchNorm1d(512),
            LeakyReLU(0.2),

            # block 4
            Conv1d(512, 512, 3, padding=1),
            Conv1d(512, 512, 3, padding=1),
            BatchNorm1d(512),
            LeakyReLU(0.2)
        )

        # final decision maker
        self.decision_maker = Linear(512, 1, bias=False)

    def forward(self, x):
        from torch.nn.functional import sigmoid

        # transpose x for proper output:
        x = x.permute(0, 2, 1)

        # apply the 1d convolutions to obtain the features
        features = self.model(x)

        # transpose the features before returning them
        features = features.permute(0, 2, 1)

        # apply the final softmax layer
        predictions = sigmoid(self.decision_maker(features))

        # return the predictions and the leaked features
        return predictions, features
