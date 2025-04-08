import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        # Xavier 初始化
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / (in_dim + out_dim))
        self.b = np.zeros(out_dim)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad / self.x.shape[0]
        self.db = np.mean(grad, axis=0)
        return grad @ self.W.T