import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.logits = None
        self.labels = None
        self.probs = None

    def forward(self, logits, labels):
        """
        logits: 预测值 (未经过softmax的输出)，shape=(batch_size, num_classes)
        labels: 真实标签（one-hot 编码），shape=(batch_size, num_classes)
        """
        self.logits = logits
        self.labels = labels

        # 防止数值溢出
        logits_max = np.max(logits, axis=1, keepdims=True)
        stable_logits = logits - logits_max
        exp_logits = np.exp(stable_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 交叉熵损失
        loss = -np.sum(labels * np.log(self.probs + 1e-12)) / logits.shape[0]
        return loss

    def backward(self):
        """
        计算损失函数相对logits的梯度
        """
        batch_size = self.logits.shape[0]
        grad = (self.probs - self.labels) / batch_size
        return grad
