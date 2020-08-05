import RCML.Fhow_RNN as Fhow
import RCML.FwhatV1 as Fwhat
import numpy as np
import torch
import matplotlib.pyplot as plt

# meta-model
class Meta(torch.nn.Module):

    def __buildModel(self):
        self.meta_layer = torch.nn.Linear(1, 2)
        self.output_layer = torch.nn.Linear(1, 1)

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
        self.metaOptimizer = torch.optim.Adam(self.meta_layer.parameters(), lr=0.01)

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

        #print("==> coeffs = ", coeffs)
        input_weights = torch.reshape(coeffs[:1], [1, 1])
        self.output_layer.state_dict()['weight'][:] = input_weights

        hidden_weights = torch.reshape(coeffs[1:2], [1, 1])
        self.output_layer.state_dict()['bias'][:] = hidden_weights

metaModel = Meta()

def generateData(N):
    X = []
    Y = []

    for _ in range(N):
        x = np.random.normal()
        y = x * 0.25 + 2.5

        X.append([x])
        Y.append([y])

    return np.array(X), np.array(Y)

N = 1000
NUM_EPOCHS = 30
X, Y = generateData(N)

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

preds = []
for epoch in range(NUM_EPOCHS):
    avg_loss = 0.
    for i in range(N):
        coeffs = metaModel.getCoeffsList()

        # Task gradient update
        taskEstimate = metaModel.predictMeta(X[i])
        metaModel.applyCoeffs(taskEstimate)
        y_pred = metaModel.predictTask(X[i])

        if epoch == NUM_EPOCHS - 1:
            preds.append(y_pred[0])

        metaModel.taskOptimizer.zero_grad()

        taskLoss = metaModel.taskCriterion(y_pred, Y[i])
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
plt.plot(Y[:100])
plt.show()
