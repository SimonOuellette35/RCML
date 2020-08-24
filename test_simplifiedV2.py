import RCML.Fhow_RNN as Fhow
import RCML.FwhatV1 as Fwhat
import numpy as np
import torch
import matplotlib.pyplot as plt

# TODO: 2. add another dimension to the time series that is a sine wave. Can it learn?
# 3. next step towards being the same as test_RCMLV1.py

N = 1000
NUM_EPOCHS = 30
SEQ_LENGTH = 5
OUTPUT_DIM = 2
NUM_TASK_PARAMS = SEQ_LENGTH+OUTPUT_DIM

# meta-model
class Meta(torch.nn.Module):

    def __buildModel(self):
        self.meta_layer = torch.nn.Linear(SEQ_LENGTH, NUM_TASK_PARAMS)
        self.output_layer = torch.nn.Linear(SEQ_LENGTH, OUTPUT_DIM)

        self.model = torch.nn.Sequential(
            self.meta_layer,
            self.output_layer
        )

        self.meta_layer.double()
        self.output_layer.double()
        self.model.double()

    def __init__(self):
        super(Meta, self).__init__()
        self.__buildModel()
        self.taskCriterion = torch.nn.MSELoss()
        self.taskOptimizer = torch.optim.Adam(self.output_layer.parameters(), lr=0.01)

        self.metaCriterion = torch.nn.MSELoss()
        self.metaOptimizer = torch.optim.Adam(self.meta_layer.parameters(), lr=0.0001)  # NOTE: This learning rate differential
            # between meta and task-specific training is crucial. Meta must be slower than task-specific.

    def predictMeta(self, X):
        return self.meta_layer(X)

    def predictTask(self, x):
        return self.output_layer(x)

    def getCoeffsList(self):
        # convert from the "per-module" structure to a flat numpy array of parameters
        param_list = list(self.output_layer.parameters())

        output = []
        for params in param_list:
            parameters = np.reshape(params.cpu().data.numpy(), [-1])
            #print("getCoeffsList: num params = ", len(parameters))
            for p in parameters:
                output.append(p)

        return np.array(output)

    def applyCoeffs(self, coeffs):
        #print("==> state_dict = ", self.output_layer.state_dict())

        task_weights = torch.reshape(coeffs[:SEQ_LENGTH], [OUTPUT_DIM, SEQ_LENGTH])
        self.output_layer.state_dict()['weight'][:] = task_weights

        task_bias = torch.reshape(coeffs[SEQ_LENGTH:SEQ_LENGTH+OUTPUT_DIM], [OUTPUT_DIM, OUTPUT_DIM])
        self.output_layer.state_dict()['bias'][:] = task_bias

metaModel = Meta()

def generateTimeSeries(N):
    ts = []

    for t in range(0, N):
        datapoint = np.zeros(2)
        datapoint[0] = np.sin(t * 0.1)
        datapoint[1] = np.cos(t * 0.25)

        ts.append(datapoint)

    return np.array(ts)

np_ts = generateTimeSeries(N)
ts = torch.from_numpy(np_ts)

preds = []
for epoch in range(NUM_EPOCHS):
    avg_loss = 0.
    for t in range(SEQ_LENGTH, len(ts)):
        x_seq = ts[t-SEQ_LENGTH:t]
        y = ts[t]

        # Task gradient update
        taskEstimate = metaModel.predictMeta(x_seq)
        metaModel.applyCoeffs(taskEstimate)
        y_pred = metaModel.predictTask(x_seq)

        if epoch == NUM_EPOCHS - 1:
            preds.append(y_pred[0])

        metaModel.taskOptimizer.zero_grad()

        taskLoss = metaModel.taskCriterion(y_pred, y)
        taskLoss.backward()
        metaModel.taskOptimizer.step()

        coeffs = metaModel.getCoeffsList()

        avg_loss += taskLoss.cpu().data.numpy()

        # Meta gradient update
        metaModel.metaOptimizer.zero_grad()

        metaLoss = metaModel.metaCriterion(taskEstimate, torch.from_numpy(coeffs))
        metaLoss.backward()
        metaModel.metaOptimizer.step()

    avg_loss /= float(N)
    print("Epoch %s loss = %s" % (epoch+1, avg_loss))

plt.plot(preds[:100])
plt.plot(np_ts[:100])
plt.show()
