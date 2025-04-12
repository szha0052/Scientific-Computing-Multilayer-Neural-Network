import numpy as np
from MLP.layers.Linear import Linear
from MLP.layers.BatchNorm import BatchNorm

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化 Adam 优化器
        :param lr: 学习率
        :param beta1: 一阶矩估计的指数衰减率
        :param beta2: 二阶矩估计的指数衰减率
        :param epsilon: 防止除零的小值
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩
        self.t = 0   # 时间步

    def increment_time_step(self):
        """
        在每个 epoch 开始时调用，更新时间步
        """
        self.t += 1


    def update(self, layers, lr, weight_decay):
        """
        更新给定层的参数
        :param layer: 当前层
        :param v: 动量字典，包含 'W' 和 'b'
        :param lr: 学习率
        """
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                # 初始化一阶矩和二阶矩
                if i not in  self.m:
                    self.m[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                # 计算梯度
                grad_W = layer.dW + weight_decay * layer.W
                grad_b = layer.db

                # 更新一阶矩和二阶矩
                self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * grad_W
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * grad_b
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * (grad_W ** 2)
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * (grad_b ** 2)

                # 计算偏差修正
                m_hat_W = self.m[i]['W'] / (1 - self.beta1 ** self.t)
                m_hat_b = self.m[i]['b'] / (1 - self.beta1 ** self.t)
                v_hat_W = self.v[i]['W'] / (1 - self.beta2 ** self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2 ** self.t)

                # 更新参数
                layer.W -= lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
                layer.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            elif isinstance(layer, BatchNorm):
                # 更新 BatchNorm 的 gamma 和 beta 参数
                layer.gamma -= lr * layer.dgamma
                layer.beta -= lr * layer.dbeta