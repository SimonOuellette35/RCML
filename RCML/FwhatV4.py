import torch

# In V4: the Fwhat module is essentially V1 + a cosine similarity approach. There are N Fhow modules. Fwhat
# outputs a parameter set, and the normalized dot products (cosine similarity) are produced on all of the Fhow modules.
# Their respective predictions on the data are combined linearly by the similarity-weighted ensemble. This is fully
# differentiable and it trains the existing Fhow weights as well as the Fwhat weights. However, if the max similarity
# seen during that training iteration was below a certain threshold, we create a new Fhow module initalized from the
# parameters provided by the Fwhat module. This specific operation is not differentiable, but it doesn't matter because
# there is no optimization to be done on it. This is inspired by the Visual Few-Shot CL learning paper.

class FwhatV4(torch.nn.module):

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
        super(FwhatV4, self).__init__()
        self.SEQ_LENGTH = SEQ_LENGTH
        self.USE_GPU = USE_GPU
        self.__buildModel()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

