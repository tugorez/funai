import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, weights, bias, weights_gradient, bias_gradient):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, learning_rate=1e-2):
        super().__init__(learning_rate)

    def step(self, weights, bias, weights_gradient, bias_gradient):
        weights -= self.learning_rate * weights_gradient
        bias -= self.learning_rate * bias_gradient