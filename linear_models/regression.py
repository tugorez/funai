import numpy as np

class Regression:
    def __init__(self, input_dim, output_dim, loss_fn, activation_fn, name='Regression'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = loss_fn
        self.activation_fn = activation_fn

        self.weights = np.random.randn(self.input_dim, self.output_dim) * 1e-2
        self.bias = np.random.randn(1, self.output_dim)
    
    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'
    
    def __call__(self, x):
        return self._predict(x)

    def _predict(self, x):
        linear_output = x @ self.weights + self.bias
        return self.activation_fn(linear_output)
    
    def train(self, x, y, epochs = 2000, learning_rate = 1e-1):
        num_samples = x.shape[0]
        last_loss = np.inf
 
        for epoch in range(1, epochs + 1):
            p = self._predict(x)

            error = self.loss_fn.gradient(y, p)
            weights_gradient = (1 / num_samples) * (x.T @ error)
            bias_gradient = (1 / num_samples) * np.sum(error, axis=0, keepdims=True)

            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * bias_gradient

            loss = self.loss_fn.loss(y, p)

            if np.isclose(last_loss, loss, atol = 1e-7):
                print(f'({epoch}/{epochs}) Loss converged. Done.')
                break
            else:
                last_loss = loss
            
            if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
                print(f'({epoch}/{epochs}) Loss: {loss:.8f}')