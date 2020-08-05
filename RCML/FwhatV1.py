import torch

# In V1: the Fwhat module is a simple sequence encoder, that encodes directly into the parameters to be used by
# the Fhow module. There is no inner training loop that occurs, as the Fhow module only uses the params given to it
# by Fwhat. This is probably not the best approach, but it is the simplest and I'm curious to see if it works. Also,
# this is NOT few-shot!

class Fwhat(torch.nn.Module):

    def __buildModel(self):
        self.rnn_layer = torch.nn.RNN(self.X_DIM, self.H_DIM, self.NUM_LAYERS, batch_first=True)
        self.output_layer = torch.nn.Linear(self.H_DIM, self.Y_DIM)

        self.model = torch.nn.Sequential(
            self.rnn_layer,
            self.output_layer
        )

        self.rnn_layer.double()
        self.output_layer.double()
        self.model.double()

    def __init__(self, X_DIM, H_DIM, Y_DIM, NUM_LAYERS, SEQ_LENGTH, USE_GPU=True, LEARNING_RATE=0.01):
        super(Fwhat, self).__init__()
        self.X_DIM = X_DIM
        self.H_DIM = H_DIM
        self.Y_DIM = Y_DIM
        self.NUM_LAYERS = NUM_LAYERS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.USE_GPU = USE_GPU
        self.__buildModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def predict(self, X):
        return self.forward(X)

    def forward(self, x_sequence_batch):
        inputs = x_sequence_batch.double()
        batch_size = inputs.size(0)

        # Initializing hidden state for first input
        if self.USE_GPU:
            hidden = torch.zeros(self.NUM_LAYERS, batch_size, self.H_DIM, device="cuda:0")
        else:
            hidden = torch.zeros(self.NUM_LAYERS, batch_size, self.H_DIM)

        hidden = hidden.double()

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn_layer(inputs, hidden)

        # only keep the last in the out sequence
        out = out[:, -1, :]
        out = self.output_layer(out)

        return torch.reshape(out, [-1, self.Y_DIM])