import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)
    
class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1,0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        return quad + linear
    
class DenseICNN_U(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections.
    Unrestricted weights. No convexification'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', dropout=0.03
    ):
        super(DenseICNN_U, self).__init__()
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation
        self.rank = rank
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
                nn.Dropout(dropout)
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.Dropout(dropout)
            )
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output)
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).cuda().float()
        )[0]
        return output    
    
class Conv2dConvexQuadratic(nn.Module):
    '''Convolutional Input-Convex Quadratic Layer'''
    def __init__(
        self, in_channels, out_channels, kernel_size, rank,
        stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
    ):
        super(Conv2dConvexQuadratic, self).__init__()
        
        assert rank > 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
    
        self.quadratic_decomposed = nn.Conv2d(
            in_channels, out_channels * rank, kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False,
            padding_mode=self.padding_mode
        )
        
        self.linear = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode
        )
        
    def forward(self, input):
        output = (self.quadratic_decomposed(input) ** 2)
        n, c, h, w = output.size()
        output = output.reshape(n, c // self.rank, self.rank, h, w).sum(2)
        output += self.linear(input)
        return output
                
class ConvICNN64(nn.Module):
    def __init__(
        self, strong_convexity=0.01,
        dropout=0.02, rank=4, batch_size=64,
        weights_init_std=0.00001
    ):
        super(ConvICNN64, self).__init__()
        
        self.strong_convexity = strong_convexity
        self.dropout = dropout
        self.rank = rank
        self.dim = 64*64*3
        self.batch_size = batch_size
        
        self.convex_layers = nn.ModuleList([
            nn.Conv2d(128, 256, 4, padding=1, stride=2), # bs x 256 x 16 x 16
        ])
        
        self.quadratic_layers = nn.ModuleList([
            Conv2dConvexQuadratic(3, 128, 4, rank=self.rank, padding=1, stride=2, bias=False),  # bs x 128 x 32 x 32
            Conv2dConvexQuadratic(3, 256, 7, rank=self.rank, padding=2, stride=4, bias=False),
        ])
        
        self.pos_features = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv2d(256, 512, 4, padding=1, stride=2), # bs x 512 x 8 x 8
            nn.CELU(0.2),
            nn.Dropout(self.dropout),
            nn.Conv2d(512, 1024, 4, padding=1, stride=2), # bs x 1024 x 4 x 4
            nn.CELU(0.2),
            nn.Dropout(self.dropout),
            nn.Conv2d(1024, 1, 4, padding=0, stride=1), # bs x 1 x 1 x 1
            View(1),
        )
        
        self._init_weights(weights_init_std)
        
    def _init_weights(self, std):
        for p in self.parameters():
            p.data = (torch.randn(p.shape, dtype=torch.float32) * std).to(p)   
                
    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            output = torch.celu(output)
            if self.training:
                output = F.dropout2d(output, p=self.dropout)
        output = self.pos_features(output)
        
        return output + .5 * self.strong_convexity * (input ** 2).flatten(start_dim=1).sum(dim=1).reshape(-1, 1)
    
    def push(self, input, create_graph=True, retain_graph=True):
        assert len(input) <= self.batch_size
        assert input.requires_grad
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=create_graph, retain_graph=retain_graph,
            only_inputs=True,
            grad_outputs=torch.ones(input.shape[0], 1).to(input)
        )[0]
        return output
    
    def convexify(self):
        for layer in list(self.convex_layers):
            if (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)
        
        for layer in list(self.pos_features):
            if (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)
    
class CompositeICNN(nn.Module):
    def __init__(self, L, D, strong_convexity=0.01):
        super(CompositeICNN, self).__init__()
        self.L = L
        self.D = D
        self.strong_convexity = strong_convexity
        
    def forward(self, input):
        output = self.D(self.L(input))
        return output + self.strong_convexity * (input.flatten(start_dim=1) ** 2).sum(dim=1, keepdims=True) / 2.
    
    def push(self, input, create_graph=True, retain_graph=True):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=create_graph, retain_graph=retain_graph,
            only_inputs=True,
            grad_outputs=torch.ones(input.shape[0], 1).to(input)
        )[0]
        return output
        
class GlobalPermutation(nn.Module):
    def __init__(self, in_features):
        super(GlobalPermutation, self).__init__()
        self.in_features = in_features
        self.idx = np.random.permutation(self.in_features)
        
    def forward(self, input):
        return input.flatten(start_dim=1)[:, self.idx].reshape(input.shape)
    
class GlobalReflection(nn.Module):
    def __init__(self, in_features):
        super(GlobalReflection, self).__init__()
        signs = np.random.randint(0,2, size=in_features) * 2 - 1
        self.signs = nn.Parameter(torch.tensor(signs, dtype=torch.float32))
        
    def forward(self, input):
        return (input.flatten(start_dim=1) * self.signs.flatten()).reshape(input.shape)