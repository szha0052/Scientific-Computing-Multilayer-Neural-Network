import numpy as np
from MLP.layers.Linear import Linear
from MLP.layers.ReLU import ReLU
from MLP.layers.SoftmaxCrossEntropy import SoftmaxCrossEntropy
from MLP.layers.Dropout import Dropout
from MLP.layers.BatchNorm import BatchNorm
# from layers import Linear, ReLU, SoftmaxCrossEntropy, Dropout, BatchNorm # waiting for rename/adjustments

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5, weight_decay=0.0001):
        # initialize
        self.layers = []
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        # combine dims
        dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(dims) - 1
        # introduce Linear layer (Fully Connected Layer)
        for i in range(self.num_layers):
            self.layers.append(Linear(dims[i], dims[i+1]))
            if i < self.num_layers - 1:
                self.layers.append(BatchNorm(dims[i+1]))
                self.layers.append(ReLU())
                self.layers.append(Dropout(dropout_rate))
        # output layer
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, training=True):
        # training=True randomly turns some neuron outputs to 0
        # training=False dont discard any neurons, use all activation values directly
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training)
            elif isinstance(layer, BatchNorm):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        output = x
        return output

    def compute_loss(self, logits, labels):
        ce_loss = self.loss_fn.forward(logits, labels)# CrossEntropy Loss
        reg_loss = self.weight_decay_loss()# L2 Regularization
        total_loss = ce_loss + reg_loss
        return total_loss

    def backward(self):
        grad = self.loss_fn.backward()
        # backward from last layer
        for layer in reversed(self.layers):
            # compute gradient backward
            grad = layer.backward(grad)
        return grad

    def weight_decay_loss(self):
        # initialize decay value
        decay = 0.0
        for layer in self.layers:
            # only penalize Linear layer (Fully Connected Layer)
            if isinstance(layer, Linear):
                decay += np.sum(layer.W ** 2)
        reg = self.weight_decay * decay / 2
        # return regularization loss
        return reg

    def update(self, lr, momentum, velocity):
        # find weighted Linear layer
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                if i not in velocity:
                    # initialize momentum
                    velocity[i] = {
                        'W': np.zeros_like(layer.W), #v_Weight
                        'b': np.zeros_like(layer.b) # v_bias
                    }
                v = velocity[i]
                # final gradient
                # grad_W = layer.dW + self.weight_decay * layer.W
                grad_W = np.mean(layer.dW, axis=0) + self.weight_decay * layer.W
                # grad_b = layer.db 
                grad_b = np.mean(layer.db, axis=0)
                # update momentum
                v['W'] = momentum * v['W'] - lr * grad_W
                v['b'] = momentum * v['b'] - lr * grad_b

                layer.W += v['W']
                layer.b += v['b']

            elif isinstance(layer, BatchNorm):
                    # 更新 BatchNorm 的 gamma 和 beta 参数
                    layer.gamma -= lr * layer.dgamma
                    layer.beta -= lr * layer.dbeta

    def predict_and_evaluate(self, x_test, y_test):
        # Disable dropout
        logits = self.forward(x_test, training=False)
        print("logits:", logits)

        
        # y_pred label
        y_pred = np.argmax(logits, axis=1)

        # Compute acc
        acc = np.mean(y_pred == y_test)
        print(f" Test Accuracy: {acc * 100:.2f}%")

        return y_pred


