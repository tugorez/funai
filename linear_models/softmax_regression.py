import numpy as np
from regression import Regression

class CategoricalCrossEntropy:
    def loss(self, y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        correct_confidences = np.sum(y_pred * y_true, axis=1)
        return -np.mean(np.log(correct_confidences))

    def gradient(self, y_true, y_pred):
        return y_pred - y_true

def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

if __name__ == '__main__':
    color_options = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
    color_vectors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [128, 0, 128], [0, 255, 255]
    ])
    
    # --- Generate Dataset ---
    num_classes = len(color_options)
    num_samples_per_class = 50
    x_list, y_list = [], []

    for i, base_color in enumerate(color_vectors):
        noise = np.random.randint(-50, 50, size=(num_samples_per_class, 3))
        samples = np.clip(base_color + noise, 0, 255)
        x_list.append(samples)
        y_list.extend([i] * num_samples_per_class)

    # Normalize features and one-hot encode labels
    x = np.vstack(x_list) / 255.0
    y = np.eye(num_classes)[y_list]

    # --- Train the Model ---
    print('--- Training the model ---')
    model = Regression(
        input_dim=3,
        output_dim=num_classes,
        loss_fn=CategoricalCrossEntropy(),
        activation_fn=softmax,
        name='SoftmaxRegression'
    )
    model.train(x, y, epochs=2000, learning_rate=1.0)

    # --- Test the Model ---
    print("\n--- Making Predictions ---")
    test_colors = np.array([
        [240, 20, 10],   # Reddish
        [10, 250, 15],   # Greenish
        [240, 250, 20],  # Yellowish
        [150, 30, 140],  # Purplish
    ])
    test_colors_normalized = test_colors / 255.0
    
    probabilities = model(test_colors_normalized)
    predictions = np.argmax(probabilities, axis=1)

    for i, predicted_index in enumerate(predictions):
        predicted_color_name = color_options[predicted_index]
        print(f"RGB {test_colors[i]} is classified as: {predicted_color_name}")
