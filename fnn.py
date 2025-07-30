import torch

class Layer:
    def __init__(self, input_dim, output_dim, name = 'Layer'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = torch.zeros(output_dim, input_dim)

    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'

if __name__ == '__main__':
    layer = Layer(3, 2)
    print(layer)
    print(layer.weights)