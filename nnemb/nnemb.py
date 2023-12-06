import torch
import copy
import numpy as np
from torch import nn

class NeuralNetworkEmbeddingPotential:
    class NeuralNetwork(nn.Module):
        def __init__(self, input_dims, output_dims):
            super().__init__()
            self.fcs = nn.ModuleList([nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(input_dims, output_dims)])
        
        def forward(self, x):
            for fclayer in self.fcs[:-1]:
                x = fclayer(x)
                x = torch.sigmoid(x)
            x = self.fcs[-1](x)
                #x = nn.functional.elu(x)
            return x

    def __init__(self, descriptor_dim, fc_dims):
        self.descriptor_dim = descriptor_dim
        self.fc_dims = np.array(fc_dims)

    def set_nn(self):
        input_dims = copy.copy(self.fc_dims)
        output_dims = copy.copy(self.fc_dims)
        input_dims = np.insert(input_dims, 0, self.descriptor_dim)
        output_dims = np.append(output_dims, 1)
        self.nn = NeuralNetworkEmbeddingPotential.NeuralNetwork(input_dims, output_dims)
        print(self.nn)

    def init_zero_weight_bias(self):
        with torch.no_grad():
            for layer in self.nn.fcs:
                print(layer, 'filling zero weights and biases')
                layer.weight = nn.Parameter(torch.zeros_like(layer.weight, dtype=torch.double))
                layer.bias = nn.Parameter(torch.zeros_like(layer.bias, dtype=torch.double))

if __name__ == '__main__':
    fc_dims = [40, 40, 40]
    descriptor_dim = 10
    nnemb = NeuralNetworkEmbeddingPotential(descriptor_dim, fc_dims)
    nnemb.set_nn()
        
    for name, param in nnemb.nn.named_parameters():
        #print(name, param)
        pass
    nnemb.init_zero_weight_bias()
    for name, param in nnemb.nn.named_parameters():
        #print(name, param)    
        pass

    test_input = np.array([0.0,]*descriptor_dim)
    test_input = torch.from_numpy(test_input).double()
    with torch.no_grad():
        test_output = nnemb.nn(test_input)
        # for sigmoid
        assert np.isclose(test_output.numpy(), np.array([0]))
