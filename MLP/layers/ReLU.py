
class ReLU:
    def __init__(self):
        self.activated = None    # 记录 ReLU 激活后的输出
        self.mask = None    # mask TRUE,说明x<=0

    def forward(self, x):
        self.mask = (x <= 0)  # 记录小于等于0的位置
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, grad):
        # dout[self.mask] = 0    # mask TRUE,说明x<=0，将这些位置的梯度置为0
        # dx = dout #只有正向传播中大于0的输入才会传递梯度
        dx = grad.copy()
        dx[self.mask] = 0
        return dx