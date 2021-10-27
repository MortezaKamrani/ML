

import numpy as np
import matplotlib.pyplot as plt

# x = np.random.randint(30, 100, (1,70))
# y = np.zeros((1,70)) + 1
# m = x.shape[0]
# mean = np.mean(x)
# print(mean)
# z = x - mean

# sigma = np.sum(z**2) / m
# # print(sigma)

# norm = z / sigma
# # print(norm)

# # v = np.vstack([x, norm])
# # y = np.vstack([y, y+1])
# # plt.plot(X, y, 'ro')


# fig, axs = plt.subplots(2, 2, figsize=(2, 1))
# axs[0, 0].plot(x, y, 'ro')
# axs[1, 0].plot(norm, y, 'ro')
# plt.show()
x = np.random.randn(1, 10)
y = np.random.randn(10, 20)*0.01
y = np.dot(x, y)
print(y)
y = np.dot(y, np.random.randn(20, 20)*0.01)
# print(y)
y = np.dot(y, np.random.randn(20, 20)*0.01)
# print(y)
y = np.dot(y, np.random.randn(20, 20)*0.01)
print(y)

# print(z)