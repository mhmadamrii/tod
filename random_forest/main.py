import numpy as np

np.random.seed(42)
num_samples = 100

x = np.random.rand(num_samples, 3)
y = np.random.rand(0, 2, num_samples)
print(y)
