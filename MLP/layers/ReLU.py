
class ReLU:
    def __init__(self):
        self.mask = None    # 布尔数组，用于记录哪些输入是小于等于0的（即 ReLU 输出为0 的部分），用于反向传播时屏蔽梯度。

    def forward(self, x):
        self.mask = (x <= 0)    # 找出哪些位置的 x 是 <= 0，记录在 mask 中，标记为true（True 表示被“抹掉”）
        out = x.copy()    #复制x，不影响原始数据
        out[self.mask] = 0    # mask TRUE,说明x<=0，在out中将这些位置的值置为0

        return out

    def backward(self, dout):
        dout[self.mask] = 0    # mask TRUE,说明x<=0，将这些位置的梯度置为0
        dx = dout #只有正向传播中大于0的输入才会传递梯度

        return dx