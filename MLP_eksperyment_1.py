import hickle as hkl
import numpy as np
import nnet_jit4 as net
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer

class mlp___3w:
    def __init__(self, x, y_t, K1, K2, lr, err_goal, disp_freq, max_epoch):
        self.x          = x
        self.L          = self.x.shape[0]
        self.y_t        = y_t
        self.K1         = K1
        self.K2         = K2
        self.lr         = lr
        self.err_goal   = err_goal
        self.disp_freq  = disp_freq
        self.max_epoch  = max_epoch
        self.K3         = y_t.shape[0] if len(y_t.shape)>1 else 1
        self.SSE_vec = []
        self.PK_vec = []

        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        self.SSE = 0
        self.lr_vec = list()

    def predict(self,x):
        n = np.dot(self.w1, x)
        self.y1 = net.tansig(n,  self.b1*np.ones(n.shape))
        n = np.dot(self.w2, self.y1)
        self.y2 = net.tansig(n,  self.b2*np.ones(n.shape))
        n = np.dot(self.w3, self.y2)
        self.y3 = net.purelin(n, self.b3*np.ones(n.shape))
        return self.y3

    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch+1):
            self.y3 = self.predict(x_train)
            self.e = y_train - self.y3

            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (1 - sum((abs(self.e)>=0.5).astype(int)[0])/self.e.shape[1] ) * 100
            self.PK_vec.append(self.PK)
            if self.SSE < self.err_goal or self.PK == 100:
                break

            if np.isnan(self.SSE):
                break

            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(self.x,  self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            self.w1 += self.dw1
            self.b1 += self.db1
            self.w2 += self.dw2
            self.b2 += self.db2
            self.w3 += self.dw3
            self.b3 += self.db3

            self.SSE_vec.append(self.SSE)


x_nie_norm, y_t, x = hkl.load('sonar-zadanie.hkl')

max_epoch = 1000
err_goal = 1e-10
disp_freq = 10
lr = 0.0001
K1_values = [75, 80, 90]
K2_values = [40, 50, 60]

data = x.T
target = y_t

PK_mesh = np.zeros((len(K1_values), len(K2_values)))
SSE_mesh = np.zeros((len(K1_values), len(K2_values), max_epoch))

for i, K1 in enumerate(K1_values):
    for j, K2 in enumerate(K2_values):
        CVN = 10
        skfold = StratifiedKFold(n_splits=CVN)
        PK_vec = np.zeros(CVN)
        SSE_epoch = np.zeros((CVN, max_epoch))

        start = timer()
        for k, (train, test) in enumerate(skfold.split(data, np.squeeze(target)), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[test]

            mlpnet = mlp___3w(x_train.T, y_train, K1, K2, lr, err_goal, disp_freq, max_epoch)
            mlpnet.train(x_train.T, y_train.T)
            result = mlpnet.predict(x_test.T)

            n_test_samples = test.size
            PK_vec[k] = (1 - sum((abs(result - y_test)>=0.5).astype(int)[0])/n_test_samples ) * 100
            SSE_epoch[k, :len(mlpnet.SSE_vec)] = mlpnet.SSE_vec

            print("Test #{:<2}: PK_vec {} test_size {}".format(k, PK_vec[k], n_test_samples))

        print("Execution time for K1={}, K2={}: {:.2f} seconds".format(K1, K2, timer() - start))
        PK_mesh[i, j] = np.mean(PK_vec)
        print("PK for K1={}, K2={}: {:.2f}".format(K1, K2, PK_mesh[i, j]))
        SSE_mesh[i, j] = np.mean(SSE_epoch, axis=0)

best_K1_idx, best_K2_idx = np.unravel_index(np.argmax(PK_mesh), PK_mesh.shape)
best_K1 = K1_values[best_K1_idx]
best_K2 = K2_values[best_K2_idx]
best_PK = PK_mesh[best_K1_idx, best_K2_idx]

print("Best K1: {}, Best K2: {}, Best PK: {:.2f}".format(best_K1, best_K2, best_PK))

print("PK Mesh Values:")
print(PK_mesh)

K1_range, K2_range = np.meshgrid(K1_values, K2_values)

fig = go.Figure(data=[go.Surface(z=PK_mesh.T, x=K1_range, y=K2_range)])
fig.update_layout(scene=dict(
                    xaxis_title='K1',
                    yaxis_title='K2',
                    zaxis_title='Average PK'),
                    title='3D Surface Plot of Average PK',
                    autosize=False,
                    width=800,
                    height=800,
                    margin=dict(l=65, r=50, b=65, t=90))
fig.show()

for i, K1 in enumerate(K1_values):
    for j, K2 in enumerate(K2_values):
        plt.plot(range(max_epoch), SSE_mesh[i, j], label=f'K1={K1}, K2={K2}')
plt.xlabel('Epoch')
plt.ylabel('SSE')
plt.title('SSE per Epoch for Different K1 and K2')
plt.legend()
plt.show()
