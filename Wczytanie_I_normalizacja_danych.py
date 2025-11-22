import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
filename = 'sonar.all-data'
data = np.loadtxt(filename, delimiter=',', dtype=str)
x = data[:, 0:-1].astype(float).T
y_t = data[:, -1].astype(float)
y_t = y_t.reshape(1, y_t.shape[0])
print(np.transpose([np.array(range(1, x.shape[0] + 1)), x.min(axis=1), x.max(axis=1)]))
x_min = x.min(axis=1).reshape(-1, 1)
x_max = x.max(axis=1).reshape(-1, 1)
x_norm_max = 1
x_norm_min = -1
x_norm = (x_norm_max - x_norm_min) * (x - x_min) / (x_max - x_min) + x_norm_min
print(np.transpose([np.array(range(1, x.shape[0] + 1)), x_norm.min(axis=1), x_norm.max(axis=1)]))
for i in range(x.shape[0]):
    x_norm[i, :] = (x_norm_max - x_norm_min) / (x_max[i] - x_min[i]) * \
                   (x[i, :] - x_min[i]) + x_norm_min
print(np.transpose([np.array(range(1, x.shape[0] + 1)), x_norm.min(axis=1), x_norm.max(axis=1)]))
plt.plot(y_t[0])
hkl.dump([x, y_t, x_norm], 'sonar-zadanie.hkl')
print(x_norm)
