import numpy as np

class LogisticRegression:
    def __init__(self, input_dim, num_classes, name = 'LogisticRegression'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = num_classes

        self.weights = np.random.rand(input_dim, self.output_dim)
        self.bias = np.random.rand(1, self.output_dim)

    def __call__(self, x, threshold = 0.5):
        p = self._predict(x)
        return (p >= threshold).astype(int)

    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'
    
    def _predict(self, x):
        linear = x @ self.weights + self.bias
        return 1 / (1 + np.exp(-linear))
    
    def train(self, x, y, epoch = 1000, learning_rate = 1e-1):
        num_samples = x.shape[0]
        last_loss = np.inf

        for i in range(epoch):
            # Step 1: Predict
            p = self._predict(x)

            # Step 2: Calculate errors
            e = p - y

            # Calculate gradients.
            c = 1 / num_samples
            grad_w = c * (x.T @ e)
            grad_b = c * np.sum(e, axis = 0, keepdims = True)

            # Update params.
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
            
            loss = -np.mean(y * np.log(p +  1e-9) + (1 - y) * np.log(1 - p +  1e-9))

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
    x = np.random.randint(0, 256, size=(200, 1))
    y =  (x >= 128).astype(int)
    x = x / 255.0

    lr = LogisticRegression(input_dim = 1, num_classes = 1)
    lr.train(x, y)

    print("\n--- Making Predictions ---")
    test_values = np.array([[30], [110], [128], [150], [245]]) / 255.0
    predictions = lr(test_values)
    label_map = {0: "Dark ⚫️", 1: "Light ⚪️"}

    for i, value in enumerate(test_values):
        prediction_label = label_map[predictions[i][0]]
        print(f"Grayscale value {value[0] * 255} is classified as: {prediction_label}")