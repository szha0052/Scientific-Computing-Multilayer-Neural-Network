
class ReLU:
    def __init__(self):
        self.activated = None    # 记录 ReLU 激活后的输出

    def forward(self, x):
        out = x.copy()    #复制x，不影响原始数据
        out[x < 0] = 0    # mask TRUE,说明x<=0，在out中将这些位置的值置为0
        self.activated = out    # 记录 ReLU 激活后的输出
        return out

    def backward(self, grad):
        # dout[self.mask] = 0    # mask TRUE,说明x<=0，将这些位置的梯度置为0
        # dx = dout #只有正向传播中大于0的输入才会传递梯度

        return grad * (self.activated > 0)