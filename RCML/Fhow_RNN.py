import torch
import numpy as np

class Fhow_RNN(torch.nn.Module):

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

    def getCoeffsList(self):
        # convert from the "per-module" structure to a flat numpy array of parameters
        param_list = list(self.model.parameters())

        output = []
        for params in param_list:
            parameters = np.reshape(params.data.numpy(), [-1])
            print("getCoeffsList: num params = ", len(parameters))
            for p in parameters:
                output.append(p)

        return np.array(output)

    def applyParamCoeffs(self, coeffs):
        # TODO
        pass
        # # convert from flat numpy array to the "per-module" internal structure of our model
        # end_input_idx = (self.X_DIM + self.Y_DIM) * self.H_DIM
        # input_weights = np.reshape(coeffs[:end_input_idx], [self.H_DIM, self.X_DIM + self.Y_DIM])
        # input_biases = np.reshape(coeffs[end_input_idx:end_input_idx + self.H_DIM], [self.H_DIM])
        # end_input_idx += self.H_DIM
        # self.input_layer.weight = torch.nn.Parameter(torch.from_numpy(input_weights))
        # self.input_layer.bias = torch.nn.Parameter(torch.from_numpy(input_biases))
        #
        # end_hidden_idx = end_input_idx + (self.H_DIM * self.H_DIM)
        # hidden_weights = np.reshape(coeffs[end_input_idx:end_hidden_idx], [self.H_DIM, self.H_DIM])
        # hidden_biases = np.reshape(coeffs[end_hidden_idx:end_hidden_idx + self.H_DIM], [self.H_DIM])
        # end_hidden_idx += self.H_DIM
        # self.hidden_layer.weight = torch.nn.Parameter(torch.from_numpy(hidden_weights))
        # self.hidden_layer.bias = torch.nn.Parameter(torch.from_numpy(hidden_biases))
        #
        # output_weights = np.reshape(coeffs[end_hidden_idx:-self.Y_DIM], [self.Y_DIM, self.H_DIM])
        # output_biases = np.reshape(coeffs[-self.Y_DIM:], [self.Y_DIM])
        # self.output_layer.weight = torch.nn.Parameter(torch.from_numpy(output_weights))
        # self.output_layer.bias = torch.nn.Parameter(torch.from_numpy(output_biases))

    def __init__(self, X_DIM, Y_DIM, H_DIM, NUM_HIDDEN_LAYERS, LEARNING_RATE=0.01):
        super(Fhow_RNN, self).__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.H_DIM = H_DIM
        self.NUM_LAYERS = NUM_HIDDEN_LAYERS
        self.__buildModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def predict(self, X):
        return self.forward(X)

    def forward(self, x_sequence_batch):
        inputs = torch.from_numpy(x_sequence_batch)

        batch_size = inputs.size(0)

        # Initializing hidden state for first input
        hidden = torch.zeros(self.NUM_LAYERS, batch_size, self.H_DIM)

        inputs = inputs.double()
        hidden = hidden.double()
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn_layer(inputs, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.H_DIM)
        out = self.output_layer(out)
        out = torch.reshape(out[-1], [x_sequence_batch.shape[0], self.Y_DIM])

        return out