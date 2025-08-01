import numpy as np

class LinearRegression:
    def __init__(self, input_dim, output_dim, name = 'LinearRegression'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.rand(input_dim, output_dim)
        self.bias = np.random.rand(1, output_dim)

    def __call__(self, x):
        return self._predict(x)

    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'
    
    def _predict(self, x):
        return x @ self.weights + self.bias
    
    def train(self, x, y, epoch = 1000, learning_rate = 1e-1):
        num_samples = x.shape[0]

        for i in range(epoch):
            # Step 1: Predict
            p = self._predict(x)

            # Step 2: Calculate errors
            e = p - y

            # Calculate gradients.
            c = 2 / num_samples
            grad_w = c * (x.T @ e)
            grad_b = c * np.sum(e, axis = 0, keepdims = True)

            # Update params.
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
            
            if i % 10 == 0:
                loss = np.sum(e ** 2) / num_samples
                print(f'Loss {loss:.4f}')

if __name__ == '__main__':
    x = np.arange(-1, 2, 0.0001).reshape(-1, 1)
    y = 3 * x + 10

    lr = LinearRegression(input_dim = 1, output_dim = 1)
    lr.train(x, y)

    print(lr.weights, lr.bias)