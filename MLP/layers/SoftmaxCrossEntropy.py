import numpy as np

class SoftmaxCrossEntropy():
    def __init__(self):
        self.loss = None
        self.y = None  # softmax 输出
        self.t = None  # 监督数据（one-hot）

    def softmax(self, x):
        # c = np.max(x)
        # exp_x = np.exp(x - c)
        # sum_exp_x = np.sum(exp_x)

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        return  exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            # print("t:", t)
            y = y.reshape(1, y.size)
            # print("y:", y)
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)  / batch_size
        # dx = (self.y - self.t)
        return dx
