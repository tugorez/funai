import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim, name = 'LinearLayer'):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weights = np.random.rand(output_dim, input_dim)
        self.bias = np.random.rand(output_dim, 1)
    
    def __str__(self):
        return f'{self.name}({self.input_dim}, {self.output_dim})'

    def forward(self, inputs):
        return np.matmul(self.weights, inputs.T) + self.bias
    
    def backwards(self, grad):
        pass

if __name__ == '__main__':
    ll = LinearLayer(3, 2)
    print(ll)
    print(ll.weights)
    print(ll.bias)
    x = np.array([
        [1, 0, 0]
    ])
    print(ll.forward(x))
