import RCML.Fhow_RNN as Fhow
import numpy as np
import torch
import matplotlib.pyplot as plt

USE_GPU = False
VISUALIZE_FIT = False

X_DIM = Y_DIM = SERIES_DIM = 3
H_DIM = 5
NUM_HIDDEN_LAYERS = 1
SEQ_LENGTH = 5
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
N = 5000
BATCH_SIZE = 100

# TODO: can I save and load the parameter coeffs as a flat list?
model = Fhow.Fhow_RNN(X_DIM, Y_DIM, H_DIM, NUM_HIDDEN_LAYERS, USE_GPU, LEARNING_RATE)
if USE_GPU:
    model.to(torch.device("cuda:0"))

coeffs = model.getCoeffsList()
randomCoeffs = np.random.normal(size=[len(coeffs)])
model.applyParamCoeffs(randomCoeffs)

model.displayLayers()

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

def trainFhow(X, Y):
    pred = model.predict(X)

    # apply gradient step on theta params
    model.optimizer.zero_grad()

    loss = model.criterion(pred, Y)
    baseline_loss = baselineMSE(Y)
    variance_loss = variance(Y)

    loss.backward()
    model.optimizer.step()

    return pred, loss, baseline_loss, variance_loss

time_series = generateTimeSeries(N)
# plt.plot(time_series[:100])
# plt.show()

def produceXY(ts):
    X = []
    Y = []
    for i in range(SEQ_LENGTH, len(ts)-1):
        x = np.array(ts[i-SEQ_LENGTH:i])
        y = np.array([ts[i]])

        X.append(x)
        Y.append(y)

    return np.array(X), np.reshape(Y, [-1, Y_DIM])

X_dataset, Y_dataset = produceXY(time_series)

training_batches = []
current_batchX = []
current_batchY = []
for pair in zip(X_dataset, Y_dataset):
    if len(current_batchX) < BATCH_SIZE:
        current_batchX.append(pair[0])
        current_batchY.append(pair[1])
    else:
        X = torch.from_numpy(np.array(current_batchX))
        Y = torch.from_numpy(np.array(current_batchY))

        if USE_GPU:
            X = X.cuda()
            Y = Y.cuda()

        training_batches.append((X, Y))

        current_batchX = []
        current_batchY = []

fit_visualization = []
for epoch in range(NUM_EPOCHS):
    losses = None
    avg_baseline_loss = np.zeros(Y_DIM)
    avg_variance_loss = np.zeros(Y_DIM)

    for b in training_batches:
        pred, loss, baseline_loss, variance_loss = trainFhow(b[0], b[1])

        if VISUALIZE_FIT and epoch == NUM_EPOCHS - 1:
            if USE_GPU:
                fit_visualization.append([pred.cpu().data.numpy(), b[1].cpu().data.numpy()])
            else:
                fit_visualization.append([pred.data.numpy(), b[1].data.numpy()])
        loss = torch.reshape(loss, [1])
        avg_baseline_loss += baseline_loss
        avg_variance_loss += variance_loss

        if losses is None:
            losses = loss
        else:
            losses = torch.cat((losses, loss))

    mean_loss = torch.mean(losses)
    mean_baseline_loss = avg_baseline_loss / float(len(training_batches))
    mean_variance_loss = avg_variance_loss / float(len(training_batches))
    print("Epoch %s avg. loss: %s (baseline loss: %s, variance: %s)" % (
        epoch+1,
        mean_loss.cpu().data.numpy(),
        np.mean(mean_baseline_loss),
        np.mean(mean_variance_loss)))

# display final fit
if VISUALIZE_FIT:
    final_y = []
    preds = []

    VIZ_WINDOW = 100
    for b in fit_visualization:
        for pred_y in b[0]:
            preds.append(pred_y)

        for actual_y in b[1]:
            final_y.append(actual_y)

    plt.plot(final_y[:VIZ_WINDOW])
    plt.plot(preds[:VIZ_WINDOW], linestyle=':')
    plt.show()

# get the trained model's coefficients
optimal_coeffs = model.getCoeffsList()

# instantiate an entirely new module
new_model = Fhow.Fhow_RNN(X_DIM, Y_DIM, H_DIM, NUM_HIDDEN_LAYERS, USE_GPU, LEARNING_RATE)

# show that it isn't trained (poor predictive accuracy)
def testModel(m, x, y):
    x_data = torch.from_numpy(np.array(x))
    if USE_GPU:
        x_data = x_data.cuda()

    preds = m.predict(x_data)

    if USE_GPU:
        preds = preds.cpu().data.numpy()
    else:
        preds = preds.data.numpy()

    sqerror = (preds - y) ** 2.0
    return np.mean(sqerror)

modelMSE = testModel(model, X_dataset,Y_dataset)
newMSE = testModel(new_model, X_dataset,Y_dataset)

print("Optimal model MSE: ", modelMSE)
print("New model MSE: ", newMSE)

# apply previously learned weights
new_model.applyParamCoeffs(optimal_coeffs)
print("===> Old Model:")
model.displayLayers()
print("===> New Model:")
new_model.displayLayers()

newcoeffs = new_model.getCoeffsList()

# show equal accuracy as previous model
newMSE2 = testModel(new_model, X_dataset, Y_dataset)
print("New model MSE after applying optimal weights: ", newMSE2)