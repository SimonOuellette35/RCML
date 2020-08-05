import torch
import numpy as np

class Fhow_RNN(torch.nn.Module):

    def displayLayers(self):
        print("==> rnn_layer weights: ", self.rnn_layer.all_weights)

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
            parameters = np.reshape(params.cpu().data.numpy(), [-1])
            print("getCoeffsList: num params = ", len(parameters))
            for p in parameters:
                output.append(p)

        return np.array(output)

    def applyParamCoeffsTensors(self, coeffs):
        # convert from flat tensor to the "per-module" internal structure of our model
        # the rnn module
        #print("Setting to coeffs: ", coeffs)
        end_input_idx = self.X_DIM * self.H_DIM
        input_weights = torch.reshape(coeffs[:end_input_idx], [self.H_DIM, self.X_DIM])
        self.model.state_dict()['0.weight_ih_l0'][:] = input_weights

        end_hidden_idx = end_input_idx + (self.H_DIM * self.H_DIM)
        hidden_weights = torch.reshape(coeffs[end_input_idx:end_hidden_idx], [self.H_DIM, self.H_DIM])
        self.model.state_dict()['0.weight_hh_l0'][:] = hidden_weights

        end_input_bias = end_hidden_idx + self.H_DIM
        input_bias = torch.reshape(coeffs[end_hidden_idx:end_input_bias], [self.H_DIM])
        self.model.state_dict()['0.bias_ih_l0'][:] = input_bias

        end_hidden_bias = end_input_bias + self.H_DIM
        hidden_bias = torch.reshape(coeffs[end_input_bias:end_hidden_bias], [self.H_DIM])
        self.model.state_dict()['0.bias_hh_l0'][:] = hidden_bias

        # the linear module
        end_linear_input = end_hidden_bias + (self.H_DIM * self.Y_DIM)
        linear_weights = torch.reshape(coeffs[end_hidden_bias:end_linear_input], [self.Y_DIM, self.H_DIM])
        self.model.state_dict()['1.weight'][:] = linear_weights

        linear_bias = torch.reshape(coeffs[end_linear_input:], [self.Y_DIM])
        self.model.state_dict()['1.bias'][:] = linear_bias

        #print("==> new state_dict = ", self.model.state_dict())

    def applyParamCoeffs(self, coeffs):
        state_dict = self.model.state_dict()
        print("==> initial state_dict = ", state_dict)
        # convert from flat numpy array to the "per-module" internal structure of our model
        # the rnn module
        end_input_idx = self.X_DIM * self.H_DIM
        input_weights = np.reshape(coeffs[:end_input_idx], [self.H_DIM, self.X_DIM])
        state_dict['0.weight_ih_l0'] = torch.from_numpy(input_weights)

        end_hidden_idx = end_input_idx + (self.H_DIM * self.H_DIM)
        hidden_weights = np.reshape(coeffs[end_input_idx:end_hidden_idx], [self.H_DIM, self.H_DIM])
        state_dict['0.weight_hh_l0'] = torch.from_numpy(hidden_weights)

        end_input_bias = end_hidden_idx + self.H_DIM
        input_bias = np.reshape(coeffs[end_hidden_idx:end_input_bias], [self.H_DIM])
        state_dict['0.bias_ih_l0'] = torch.from_numpy(input_bias)

        end_hidden_bias = end_input_bias + self.H_DIM
        hidden_bias = np.reshape(coeffs[end_input_bias:end_hidden_bias], [self.H_DIM])
        state_dict['0.bias_hh_l0'] = torch.from_numpy(hidden_bias)

        # the linear module
        end_linear_input = end_hidden_bias + (self.H_DIM * self.Y_DIM)
        linear_weights = np.reshape(coeffs[end_hidden_bias:end_linear_input], [self.Y_DIM, self.H_DIM])
        state_dict['1.weight'] = torch.from_numpy(linear_weights)

        linear_bias = np.reshape(coeffs[end_linear_input:], [self.Y_DIM])
        state_dict['1.bias'] = torch.from_numpy(linear_bias)

        print("==> new state_dict = ", state_dict)

        self.model.load_state_dict(state_dict)
        self.displayLayers()

    def __init__(self, X_DIM, Y_DIM, H_DIM, NUM_HIDDEN_LAYERS, USE_GPU=True, LEARNING_RATE=0.01):
        super(Fhow_RNN, self).__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.H_DIM = H_DIM
        self.NUM_LAYERS = NUM_HIDDEN_LAYERS
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