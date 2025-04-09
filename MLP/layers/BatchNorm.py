import numpy as np
from MLP.layers.Linear import Linear

class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum

        # 参数 gamma 和 beta，用于缩放和平移
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

        # 存储训练过程中的均值和方差
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

        # 缓存变量
        self.x_norm = None
        self.mean = None
        self.var = None
        self.x_centered = None

    def forward(self, x, training=True):
        if training:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)

            # 标准化
            self.x_centered = x - self.mean
            self.x_norm = self.x_centered / np.sqrt(self.var + self.eps)

            # 更新运行中的均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            # 测试阶段使用运行中的均值和方差
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # 缩放和平移
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, grad):
        batch_size = grad.shape[0]
        # 参数梯度
        dgamma = np.sum(grad * self.x_norm, axis=0)
        dbeta = np.sum(grad, axis=0)

        # 输入梯度
        dx_norm = grad * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.var + self.eps)**(-1.5), axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)

        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * self.x_centered / batch_size + dmean / batch_size

        # 更新参数
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
    
