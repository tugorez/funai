import numpy as np
from regression import Regression

class BinaryCrossEntropy:
    def loss(self, y_true, y_pred):
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        return y_pred - y_true

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    x = np.random.randint(0, 256, size=(200, 1))
    y =  (x >= 128).astype(int)
    x = x / 255.0

    model = Regression(input_dim=1, output_dim=1, loss_fn=BinaryCrossEntropy(), activation_fn=sigmoid)
    model.train(x, y)

    print("\n--- Making Predictions ---")
    test_values = np.array([[30], [110], [128], [150], [245]]) / 255.0
    predictions = model(test_values)
    label = lambda p: "Dark ⚫️" if p <=0.5 else "Light ⚪️"

    for i, value in enumerate(test_values):
        prediction_label = label(predictions[i][0])
        print(f"Grayscale value {value[0] * 255} is classified as: {prediction_label}")