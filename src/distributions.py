import torch
import numpy as np
import random
from scipy.linalg import sqrtm
import sklearn.datasets

def symmetrize(X):
    return np.real((X + X.T) / 2)

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class StandardNormalSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(StandardNormalSampler, self).__init__(device)
        self.dim, self.shape = dim, (dim,)
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var, self.cov = float(dim), np.eye(dim, dtype=np.float32)
        
    def sample(self, size=10):
        return torch.randn(
            size, self.dim,
            device=self.device
        )
    
class SwissRollSampler(Sampler):
    def __init__(
        self, estimate_size=100000, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        self.dim = 2
        
        batch = self.sample(1024).to('cpu').numpy().astype(np.float32)
        self.mean = np.mean(batch, axis=0)
        self.cov = np.cov(batch.T)
        
    def sample(self, batch_size=10):
        batch = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype(np.float32)[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    
    
class NormalSampler(Sampler):
    def __init__(
        self, mean, cov=None, weight=None, device='cuda'
    ):
        super(NormalSampler, self).__init__(device=device)
        self.mean = np.array(mean, dtype=np.float32)
        self.dim = self.mean.shape[0]
        
        if weight is not None:
            weight = np.array(weight, dtype=np.float32)
        
        if cov is not None:
            self.cov = np.array(cov, dtype=np.float32)
        elif weight is not None:
            self.cov = weight @ weight.T
        else:
            self.cov = np.eye(self.dim, dtype=np.float32)
            
        if weight is None:
            weight = symmetrize(sqrtm(self.cov))
            
        self.var = np.trace(self.cov)
        
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(self.mean, device=self.device, dtype=torch.float32)

    def sample(self, batch_size=4):
        batch = torch.randn(batch_size, self.dim, device=self.device)
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        return batch
    
class CubeUniformSampler(Sampler):
    def __init__(
        self, dim=1, centered=False, normalized=False, device='cuda'
    ):
        super(CubeUniformSampler, self).__init__(
            device=device,
        )
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = np.eye(self.dim, dtype=np.float32) if self.normalized else np.eye(self.dim, dtype=np.float32) / 12
        self.mean = np.zeros(self.dim, dtype=np.float32) if self.centered else .5 * np.ones(self.dim, dtype=np.float32)
        
        self.bias = torch.tensor(self.mean, device=self.device)
        
    def sample(self, batch_size=10):
        return np.sqrt(self.var) * (torch.rand(
            batch_size, self.dim, device=self.device
        ) - .5) / np.sqrt(self.dim / 12)  + self.bias

    
class Transformer(object):
    def __init__(
        self, device='cuda'
    ):
        self.device = device
        
class LinearTransformer(Transformer):
    def __init__(
        self, weight, bias=None, base_sampler=None,
        device='cuda'
    ):
        super(LinearTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        self.dim = weight.shape[0]
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32, requires_grad=False)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32, requires_grad=False)
        else:
            self.bias = torch.zeros(self.dim, device=device, dtype=torch.float32, requires_grad=False)
        
        
        if base_sampler is not None:
            self.fit(base_sampler)

        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        weight, bias = self.weight.cpu().numpy(), self.bias.cpu().numpy()
        
        self.mean = weight @ self.base_sampler.mean + bias
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)
        
        self.fitted = True
        return self
        
    def sample(self, batch_size=4):
        assert self.fitted == True
        
        batch = torch.tensor(
            self.base_sampler.sample(batch_size),
            device=self.device, 
        )
        with torch.no_grad():
            batch = batch @ self.weight.T
            if self.bias is not None:
                batch += self.bias
        batch = batch.detach()
        return batch