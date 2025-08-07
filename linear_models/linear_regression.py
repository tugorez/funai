import numpy as np
from regression import Regression
from optimizers import SGD


class MeanSquaredError:
    def loss(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)

    def gradient(self, y_true, y_predicted):
        return y_predicted - y_true

def identity(x):
    return x

if __name__ == '__main__':
    x = np.arange(-1, 1, 0.1).reshape(-1, 1)
    y = 3 * x + 10

    loss_fn = MeanSquaredError()
    activation_fn = identity
    model = Regression(input_dim = 1, output_dim = 1, loss_fn = loss_fn, activation_fn=activation_fn, optimizer=SGD())
    model.train(x, y)

    print(f'Learned Parameters: slope (w) = {model.weights[0][0]:.4f}, intercept (b) = {model.bias[0][0]:.4f}')
    print(f'True Parameters:    slope (w) = 3.0000, intercept (b) = 10.0000')