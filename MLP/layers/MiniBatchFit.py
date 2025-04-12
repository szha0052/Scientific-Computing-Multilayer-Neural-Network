
import numpy as np

class MiniBatchFit:
    def __init__(self, model, optimizer, 
                 X_train, Y_train, 
                 output_dim, 
                 num_epochs=10, 
                 batch_size=32, 
                 learning_rate=0.01):
        """
        :param model: 你的模型对象，内部应包含forward、compute_loss、backward、update等方法
        :param optimizer: 优化器对象，示例里仅使用了optimizer.increment_time_step()
        :param X_train: 训练特征，形如 (N, feature_dim)
        :param Y_train: 训练标签，形如 (N, )
        :param output_dim: 输出维度，用于生成one-hot标签
        :param num_epochs: 训练轮数
        :param batch_size: 批大小
        :param learning_rate: 学习率
        :param velocity: 动量（若模型更新使用了动量相关机制）
        """
        self.model = model
        self.optimizer = optimizer
        self.X_train = X_train
        self.Y_train = Y_train
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self):
        for epoch in range(self.num_epochs):
            # 若你的optimizer需要按时间步数增加，这里调用
            if hasattr(self.optimizer, 'increment_time_step'):
                # 这里假设optimizer有一个increment_time_step方法
                # 用于更新时间步数或其他状态
                self.optimizer.increment_time_step()

            epoch_loss = 0
            # 按batch遍历数据
            for i in range(0, len(self.X_train), self.batch_size):
                X_batch = self.X_train[i:i+self.batch_size]
                Y_batch = self.Y_train[i:i+self.batch_size]

                # 生成one-hot编码
                Y_batch_one_hot = np.eye(self.output_dim)[Y_batch]

                # 前向传播
                logits = self.model.forward(X_batch, training=True)

                # 计算损失
                loss = self.model.compute_loss(logits, Y_batch_one_hot)
                epoch_loss += loss

                # 反向传播
                self.model.backward()

                # 更新参数
                self.model.update(self.learning_rate)

            # 计算并打印每个epoch的平均损失
            avg_loss = epoch_loss / (len(self.X_train) // self.batch_size)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}')
