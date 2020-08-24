import RCML.Fhow_RNN as Fhow
import RCML.FwhatV1 as Fwhat
import numpy as np
import torch
import matplotlib.pyplot as plt

# TODO: IDEA: train individual models on each task, then train the meta-module to output weights similar to the
#  individual model's weights for each task.

USE_GPU = False
VISUALIZE_FIT = False

X_DIM = Y_DIM = SERIES_DIM = 3
PARAM_DIM = 68
META_H_DIM = 1
META_NUM_LAYERS = 1
META_LEARNING_RATE = 0.0001

H_DIM = 5
NUM_HIDDEN_LAYERS = 1
SEQ_LENGTH = 5
NUM_EPOCHS = 100
TASK_LEARNING_RATE = 0.01
N = 5000
BATCH_SIZE = 100

modelFwhat = Fwhat.Fwhat(X_DIM, META_H_DIM, PARAM_DIM, META_NUM_LAYERS, SEQ_LENGTH, USE_GPU, META_LEARNING_RATE)
modelFhow = Fhow.Fhow_RNN(X_DIM, Y_DIM, H_DIM, NUM_HIDDEN_LAYERS, USE_GPU, TASK_LEARNING_RATE)
coeffList = modelFhow.getCoeffsList()
print("coeffList = ", coeffList.shape)
if USE_GPU:
    modelFwhat.to(torch.device("cuda:0"))
    modelFhow.to(torch.device("cuda:0"))

def generateTimeSeries(N):
    time_series = [np.random.normal(size=[SERIES_DIM])]

    def step(x, t):
        new_data = np.zeros_like(x)
        new_data[0] = np.sin(t*0.25) + x[0]
        new_data[1] = -0.01 * x[1] + np.random.normal() * 1.
        new_data[2] = (new_data[1] ** 2.0) / 2

        return new_data

    for t in range(N-1):
        new_data = step(time_series[t-1], t)
        time_series.append(new_data)

    return np.array(time_series)

def variance(Y):
    if USE_GPU:
        y = Y.cpu().data.numpy()
    else:
        y = Y.data.numpy()

    means = np.mean(y, axis=0)

    MSE = np.zeros(Y_DIM)
    for t in range(len(y)):
        sqerror = (y[t] - means) ** 2.0
        MSE += sqerror

    MSE /= float(len(y))

    return MSE

def baselineMSE(Y):

    if USE_GPU:
        y = Y.cpu().data.numpy()
    else:
        y = Y.data.numpy()

    MSE = np.zeros(Y_DIM)
    for t in range(1, len(y)):
        #print("y[t] = %s, y[t-1] = %s" % (y[t], y[t-1]))
        sqerror = (y[t] - y[t-1]) ** 2.0
        MSE += sqerror

    MSE /= float(len(y) - 1)

    return MSE

time_series = generateTimeSeries(N)
# plt.plot(time_series[:100])
# plt.show()

# TODO: BUG: not learning. Why?
for epoch in range(NUM_EPOCHS):
    avg_loss = 0.
    for t in range(SEQ_LENGTH, len(time_series)):
        s_t = torch.from_numpy(np.array([time_series[t-SEQ_LENGTH:t]]))
        if USE_GPU:
            s_t = s_t.cuda()

        taskEstimate = modelFwhat.predict(s_t)
        modelFhow.applyParamCoeffsTensors(taskEstimate[0])

        # Task gradient update
        y_pred = modelFhow.predict(s_t)
        modelFhow.optimizer.zero_grad()
        y = np.array([time_series[t]])
        taskLoss = modelFhow.criterion(y_pred, torch.from_numpy(y))
        taskLoss.backward()
        modelFhow.optimizer.step()

        actualCoeffs = np.reshape(modelFhow.getCoeffsList(), [1, -1])
        avg_loss += taskLoss.cpu().data.numpy()

        # Meta gradient update
        modelFwhat.optimizer.zero_grad()
        metaLoss = modelFwhat.criterion(taskEstimate, torch.from_numpy(actualCoeffs))
        metaLoss.backward()
        modelFwhat.optimizer.step()

    avg_loss /= float(len(time_series) - SEQ_LENGTH)
    print("Epoch #%s loss = %s" % (epoch+1, avg_loss))

# for epoch in range(NUM_EPOCHS):
#     avg_loss = 0.
#     for t in range(SEQ_LENGTH, len(time_series)):
#         s_t = torch.from_numpy(np.array([time_series[t-SEQ_LENGTH:t]]))
#         if USE_GPU:
#             s_t = s_t.cuda()
#
#         pred = modelFwhat.predict(s_t)
#
#         # apply gradient step on Fwhat params
#         modelFwhat.optimizer.zero_grad()
#
#         y = np.array([time_series[t]])
#         loss = modelFwhat.criterion(pred, torch.from_numpy(y))
#         # baseline_loss = baselineMSE(Y)
#         # variance_loss = variance(Y)
#
#         loss.backward()
#         modelFwhat.optimizer.step()
#         avg_loss += loss.cpu().data.numpy()
#
#     avg_loss /= float(len(time_series) - SEQ_LENGTH)
#     print("Epoch #%s loss = %s" % (epoch+1, avg_loss))
