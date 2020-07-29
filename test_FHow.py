import RCML.Fhow_RNN as Fhow
import numpy as np
import torch

X_DIM = Y_DIM = SERIES_DIM = 3
H_DIM = 4
NUM_HIDDEN_LAYERS = 2
SEQ_LENGTH = 5
NUM_EPOCHS = 500

# TODO: can I save and load the parameter coeffs as a flat list?
model = Fhow.Fhow_RNN(X_DIM, Y_DIM, H_DIM, NUM_HIDDEN_LAYERS)
coeffs = model.getCoeffsList()
randomCoeffs = np.random.normal(size=[len(coeffs)])
model.applyParamCoeffs(randomCoeffs)

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

def trainFhow(X, Y):
    pred = model.predict(X)

    # apply gradient step on theta params
    model.optimizer.zero_grad()

    loss = model.criterion(pred, torch.from_numpy(Y))

    loss.backward()
    model.optimizer.step()

    return pred, loss

time_series = generateTimeSeries(10000)
# plt.plot(time_series[:100])
# plt.show()

for epoch in range(NUM_EPOCHS):
    losses = []
    for i in range(SEQ_LENGTH, len(time_series)-1):
        X = np.array([time_series[i-SEQ_LENGTH:i]])
        Y = np.array([time_series[i]])

        _, loss = trainFhow(X, Y)
        losses.append(loss.data.numpy())

    print("Epoch %s avg. loss: %s" % (epoch+1, np.mean(losses)))
