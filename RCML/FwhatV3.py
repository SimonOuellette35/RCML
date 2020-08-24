import torch

# In V3: the Fwhat module is essentially a neural attention mechanism (e.g. Neural Episodic Control). The input sequence
# S maps, in the DND, to the optimal task parameters T. There are differentiable lookup and write operations. The inner
# loop determines the optimal Fhow parameters to write as the T value in that DND.

class FwhatV3(torch.nn.module):

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

    def __init__(self, SEQ_LENGTH, USE_GPU=True, LEARNING_RATE=0.01):
        super(FwhatV3, self).__init__()
        self.SEQ_LENGTH = SEQ_LENGTH
        self.USE_GPU = USE_GPU
        self.__buildModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

