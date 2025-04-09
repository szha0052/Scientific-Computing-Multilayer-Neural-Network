import numpy as np
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio#dropout比例
        self.mask = None #mask是和输入数据x形状相同的数组，元素为True/False，表示对应位置的元素是否被抹掉

    def forward(self, x, train_flg=True):
        if train_flg:#TRUE表示处于训练模式，随机屏蔽一部分神经元，防止过拟合
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio#生成和x形状相同的数组，元素为0-1之间的随机数，如果大于dropout_ratio则为True
            return x * self.mask #保留为TRUE
        else:
            # return x * (1.0 - self.dropout_ratio)#测试时缩放输出神经元，不再随意丢弃神经元
            return x if train_flg else x * (1.0 / (1.0 - self.dropout_ratio))

    def backward(self, dout):
        a= dout * self.mask
        return dout * self.mask#只保留mask为True的神经元的梯度，其余梯度为0
        #只有未被丢弃的神经元参与梯度更新，避免某些神经元影响过大