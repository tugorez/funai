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
        last_loss = np.inf

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
            
            loss = np.sum(e ** 2) / num_samples

            # If it does not improve, halt the training process.
            if (last_loss - loss) == 0:
                print(f'({i}/{epoch}) Function is optimized, loss was not improved. Done.')
                return
            else:
                last_loss = loss
        
            # Always print the first and the last iteration.
            if i % 10 == 0 or i == epoch - 1:
                print(f'({i}/{epoch}) Loss {loss:.8f}')

if __name__ == '__main__':
    x = np.arange(-1, 1, 0.1).reshape(-1, 1)
    y = 3 * x + 10

    lr = LinearRegression(input_dim = 1, output_dim = 1)
    lr.train(x, y)

    print(f'm = {lr.weights}, b = {lr.bias}')

    results = lr([[30], [40]])
    print(results)