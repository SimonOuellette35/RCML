import torch

# In V2: the Fwhat module is essentially MAML. It learns a set of initialization parameters, gives them to Fhow, and
# Fhow runs a single or multi-step gradient update loop before running on the test data.
# 2 approaches are possible to constantly update the meta-weights:
# 1 - the FTML approach of a rolling update.
# 2 - the Continual-MAML approach of detecting new out-of-sample tasks and incorporating the new information
# TODO: Is MetaCoG also a variant of this?

class FwhatV2(torch.nn.module):

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
        super(FwhatV2, self).__init__()
        self.SEQ_LENGTH = SEQ_LENGTH
        self.USE_GPU = USE_GPU
        self.__buildModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

