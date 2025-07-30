import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, name = 'Layer'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.random((output_dim, input_dim))
        self.bias = np.random.random((output_dim, 1))

    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'
    
    def _activation(self, x):
        return np.maximum(0, x)
    

    def forward(self, batch):
        linear_output = np.matmul(self.weights, batch) + self.bias
        return self._activation(linear_output)

class FNN:
    def __init__(self, topology, name = 'FNN'):
        self.name = name

        self.layers = [
            Layer(topology[i - 1], topology[i], name = f'Layer{i}') for i in range(1, len(topology))
        ]
    
    def __str__(self):
        lines = ['  -- ' + str(layer) for layer in self.layers]
        lines = [self.name] + lines
        return '\n'.join(lines)
    
    def forward(self, batch):
        result = batch.T
        for layer in self.layers:
            result = layer.forward(result)
        return result

if __name__ == '__main__':
    fnn = FNN(topology = [3, 2, 2])
    r = fnn.forward(np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]))
    print(r)