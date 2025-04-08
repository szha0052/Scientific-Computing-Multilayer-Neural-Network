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
                grad_W = layer.dW + self.weight_decay * layer.W
                grad_b = layer.db 
                # update momentum
                v['W'] = momentum * v['W'] - lr * grad_W
                v['b'] = momentum * v['b'] - lr * grad_b

                layer.W += v['W']
                layer.b += v['b']

            # elif isinstance(layer, BatchNorm):
            #     if i not in velocity:
            #         velocity[i] = {
            #             'gamma': np.zeros_like(layer.gamma),
            #             'beta': np.zeros_like(layer.beta)
            #         }
            #     v = velocity[i]
            #     v['gamma'] = momentum * v['gamma'] - lr * layer.dgamma
            #     v['beta'] = momentum * v['beta'] - lr * layer.dbeta
            #     layer.gamma += v['gamma']
            #     layer.beta += v['beta']