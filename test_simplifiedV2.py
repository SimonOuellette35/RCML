import RCML.Fhow_RNN as Fhow
import RCML.FwhatV1 as Fwhat
import numpy as np
import torch
import matplotlib.pyplot as plt

# meta-model
class Meta(torch.nn.Module):

    def __buildModel(self):
        npMetaVar = np.ones([1, 2])
        #npTaskVar = np.ones([1, 1])
        self.metaVar = torch.nn.Parameter(data=torch.from_numpy(npMetaVar))
        #self.taskVar = torch.nn.Parameter(data=torch.from_numpy(npMetaVar))

    def __init__(self):
        super(Meta, self).__init__()
        self.__buildModel()
        self.taskCriterion = torch.nn.MSELoss()
        params = torch.nn.ParameterList([self.metaVar])
        self.taskOptimizer = torch.optim.Adam(params, lr=0.01)

    # x: N x D
    def forward(self, x):
        params = self.metaVar * x

        # params: N x 2
        y = torch.transpose(x, 0, 1) * params[:, 0] + params[:, 1]

        # y: N x 1
        return y

    # def predictMeta(self, X):
    #     return self.meta_layer(X)
    #
    # def predictTask(self, x):
    #     return self.output_layer(x)

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
X, Y = generateData(N)

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

for epoch in range(500):
#    avg_loss = 0.
    # Task gradient update
    y_pred = metaModel.forward(X)

    metaModel.taskOptimizer.zero_grad()

    taskLoss = metaModel.taskCriterion(y_pred, Y)
    taskLoss.backward()
    metaModel.taskOptimizer.step()

    avg_loss = taskLoss.data.numpy()
 #   avg_loss += taskLoss.cpu().data.numpy()

  #  avg_loss /= float(N)
    print("Epoch %s loss = %s" % (epoch+1, avg_loss))