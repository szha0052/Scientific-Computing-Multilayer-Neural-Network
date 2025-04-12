import numpy as np
from MLP.layers.Linear import Linear
from MLP.layers.BatchNorm import BatchNorm

class SGDMomentum:
    """
    SGD with Momentum optimizer class.

    Attributes:
    -----------
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient (e.g., 0.9).
    weight_decay : float
        L2 regularization coefficient.
    velocity : dict
        A dictionary storing the momentum buffers for each layer's parameters.
    """
    def __init__(self, momentum=0.9):
        """
        Initialize the SGDMomentum optimizer.

        Parameters:
        -----------
        lr : float
            Learning rate.
        momentum : float, optional
            Momentum coefficient (default is 0.9).
        weight_decay : float, optional
            L2 regularization coefficient (default is 0.0).
        """
        self.momentum = momentum
        self.velocity = {}

    def update(self, layers, lr, weight_decay):
        """
        Perform one step of SGD with Momentum update on all given layers.

        Parameters:
        -----------
        layers : list
            List of network layers (e.g., Linear, BatchNorm, etc.).
        """
        for i, layer in enumerate(layers):
            # If this layer is a Linear layer, apply momentum-based updates to W and b
            if isinstance(layer, Linear):
                # Initialize momentum buffers if they do not exist
                if i not in self.velocity:
                    self.velocity[i] = {
                        'W': np.zeros_like(layer.W),
                        'b': np.zeros_like(layer.b)
                    }

                v = self.velocity[i]

                # Compute gradients (adding weight decay)
                grad_W = layer.dW + weight_decay * layer.W
                grad_b = layer.db

                # Update momentum buffers
                v['W'] = self.momentum * v['W'] - lr * grad_W
                v['b'] = self.momentum * v['b'] - lr * grad_b

                # Update parameters
                layer.W += v['W']
                layer.b += v['b']

            # If this layer is a BatchNorm layer, apply updates to gamma and beta (no momentum here)
            elif isinstance(layer, BatchNorm):
                layer.gamma -= lr * layer.dgamma
                layer.beta -= lr * layer.dbeta
