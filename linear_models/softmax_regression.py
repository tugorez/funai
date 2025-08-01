import numpy as np

class SoftmaxRegression:
    def __init__(self, input_dim, num_classes, name = 'SoftmaxRegression'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = num_classes

        self.weights = np.random.rand(input_dim, self.output_dim)
        self.bias = np.random.rand(1, self.output_dim)

    def __call__(self, x):
        p = self._predict(x)
        # Each entry has a probability for each class so we pick the index of the one with the MAX.
        return np.argmax(p, axis = 1)

    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'
    
    def _predict(self, x):
        linear = x @ self.weights + self.bias
        exp_scores = np.exp(linear - np.max(linear, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def train(self, x, y, epoch = 1000, learning_rate = 1e-3):
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
            
            loss = -np.mean(np.sum(y * np.log(p +  1e-9), axis = 1))

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
    color_options = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
    color_vectors = np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [128, 0, 128],  # Purple
        [0, 255, 255],  # Cyan
    ])
    num_samples_per_color = 50
    x, y = [], []
    for i, base_color in enumerate(color_vectors):
        # Generate 50 vectors that will be added to each one of the base colors above.
        base_color_deltas = np.random.randint(-50, 50, size = (num_samples_per_color, 3))
        base_color_samples = np.clip(base_color + base_color_deltas, 0, 255)
        x.append(base_color)
        x.extend(base_color_samples)
        y.extend([i] * (num_samples_per_color + 1))
    x = np.array(x) / 255
    y = np.eye(len(color_vectors))[y]

    print('Training the model')
    model = SoftmaxRegression(input_dim = 3, num_classes = len(color_vectors))
    model.train(x, y, epoch = 2000, learning_rate = 1e1)

    # Test the model
    print("--- Making Predictions ---")
    tests = np.array([
        [240, 20, 10],  # A very reddish color
        [10, 250, 15],  # A very greenish color
        [240, 250, 20], # A very yellowish color
        [150, 30, 140], # A purplish color,
        [118, 65, 217], # Another purple like color
        [224, 190, 20], # A Yellowish color
    ]) / 255
    
    predictions = model(tests)
    for i, rgb in enumerate(predictions):
        predicted_color_name = color_options[predictions[i]]
        print(f"RGB{rgb} was classified as: {predicted_color_name}")
