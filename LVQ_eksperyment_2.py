import numpy as np
import hickle as hkl
import math
import nnet as net
import matplotlib.pyplot as plt

def lvqTraining(xt, y, lr, max_epoch, err_goal):
    SSE_vec = []
    label, train_idx = np.unique(y, return_index=True)
    W = xt[train_idx].astype(np.float64)
    filtrX = np.array([e for i, e in enumerate(xt) if i not in train_idx])
    filtrY = np.array([e for i, e in enumerate(y) if i not in train_idx])
    for epoch in range(1, max_epoch + 1):
        SSE = 0.0
        predicted_labels = []
        for i, x in enumerate(filtrX):
            d = [math.sqrt(sum((w - x) ** 2)) for w in W]
            Dmin = np.argmin(d)
            predicted_labels.append(label[Dmin])
            if filtrY[i] == label[Dmin]:
                W[Dmin] = W[Dmin] + lr * (x - W[Dmin])
            else:
                W[Dmin] = W[Dmin] - lr * (x - W[Dmin])
            SSE = net.sumsqr((W[Dmin] - x).reshape(1, -1)) / len(x)
        if epoch % 10:
            SSE_vec.append(SSE)
            # print("Epoch: %5d | SSE: %5.5e" % (epoch, SSE))
        if SSE < err_goal or np.isnan(SSE) or np.isinf(SSE):
            # print("Epoch: %5d | SSE: %5.5e" % (epoch, SSE))
            break
    return W, label

def lvqTest(x, W):
    weights, label = W
    d = [math.sqrt(sum((w - x) ** 2)) for w in weights]
    return label[np.argmin(d)]

def calculate_PK(data, target, lr_values, max_epoch, err_goal):
    PK_values = []
    for lr in lr_values:
        W = lvqTraining(data, target, lr, max_epoch, err_goal)
        predicted = []
        for sample in data:
            predicted.append(lvqTest(sample, W))
        result = np.array(predicted)
        PK = np.sum(result == target) / len(target)
        PK_values.append(PK)
    return PK_values

x_nie_norm, y_t, x = hkl.load('sonar-zadanie.hkl')
data = x.T
target = np.squeeze(y_t)
lr_values = [0.0001, 0.0005, 0.001, 0.01]  # Different learning rates to try
max_epoch = 500
err_goal = 1e-10

PK_values = calculate_PK(data, target, lr_values, max_epoch, err_goal)

plt.plot(lr_values, PK_values, marker='o', linestyle='None')
plt.xlabel('Learning Rate')
plt.ylabel('PK')
plt.title('PK vs Learning Rate')
plt.xscale('log')
plt.grid(True)
plt.show()
