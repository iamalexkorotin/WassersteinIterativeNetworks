import pandas as pd
import numpy as np

import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing
import h5py

from PIL import Image
from .inception import InceptionV3
from tqdm import tqdm
from .fid_score import calculate_frechet_distance

from torch.utils.data import TensorDataset

import gc

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def read_images(paths, mode='RGB', verbose=True):
    images = []
    for path in paths:
        try:
            with Image.open(path, 'r') as im:
                images.append(im.convert(mode).copy())
        except:
            if verbose:
                print('Failed to read {}'.format(path))
    return images

class ImagesReader:
    def __init__(self, mode='RGB', verbose=True):
        self.mode = mode
        self.verbose = verbose
        
    def __call__(self, paths):
        return read_images(paths, mode=self.mode, verbose=self.verbose)

def read_image_folder(path, mode='RGB', verbose=True, n_jobs=1):
    paths = [os.path.join(path, name) for name in os.listdir(path)]
    
    chunk_size = (len(paths) // n_jobs) + 1
    chunks = [paths[x:x+chunk_size] for x in range(0, len(paths), chunk_size)]
    
    pool = multiprocessing.Pool(n_jobs)
    
    chunk_reader = ImagesReader(mode, verbose)
    
    images = list(itertools.chain.from_iterable(
        pool.map(chunk_reader, chunks)
    ))
    pool.close()
    return images

def get_generated_stats(G, Z_sampler, size, batch_size=8, inception=False, verbose=False):
    if inception:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).cuda()
        freeze(model)
    else:
        dims = np.prod(G(Z_sampler.sample(1))[0].shape)
    
    freeze(G); pred_arr = np.empty((size, dims))
    with torch.no_grad():
        for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
            start, end = i, min(i + batch_size, size)

            batch = ((G(Z_sampler.sample(end-start)) + 1) / 2).type(torch.FloatTensor).cuda()
            if inception:
                pred = model(batch)[0]
            else:
                pred = batch
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_inverse_stats(Ts_inv, Y_samplers, batch_size=8, inception=False, verbose=False):
    if inception:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).cuda()
        freeze(model)
    else:
        dims = np.prod(Y_samplers[0].sample(1)[0].shape)
    
    mus, sigmas = [], []
    for k in range(len(Ts_inv)):
        size = len(Y_samplers[k].dataset)
        freeze(Ts_inv[k]); pred_arr = np.empty((size, dims))
        with torch.no_grad():
            for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
                start, end = i, min(i + batch_size, size)

                batch = Y_samplers[k].dataset[start:end].type(torch.FloatTensor).cuda()
                batch = Ts_inv[k](batch)
                if inception:
                    pred = model(batch)[0]
                else:
                    pred = batch
                pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

        mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
        mus.append(mu); sigmas.append(sigma)
    
    gc.collect(); torch.cuda.empty_cache()
    return mus, sigmas

class SumSequential(nn.Module):
    def __init__(self, G, Ts, alphas):
        super(SumSequential, self).__init__()
        self.G = G
        self.Ts = nn.ModuleList(Ts)
        self.alphas = alphas
        
    def forward(self, input):
        G_input = self.G(input)
        out = torch.zeros_like(G_input)
        for alpha, T in zip(self.alphas, self.Ts):
            out += alpha * T(G_input)
        return out

def score_gen(benchmark, G, Z_sampler, score_size=100000):
    assert benchmark.gauss_bar_sampler != None
    
    Z = Z_sampler.sample(score_size)
    with torch.no_grad():
        G_Z = G(Z).cpu().detach().numpy()
    G_Z_cov = np.cov(G_Z.T)
    G_Z_mean = np.mean(G_Z, axis=0)   
    BW2_UVP_G = 100 * calculate_frechet_distance(
        G_Z_mean, G_Z_cov,
        benchmark.gauss_bar_sampler.mean, benchmark.gauss_bar_sampler.cov,
    ) / benchmark.gauss_bar_sampler.var
        
    return BW2_UVP_G

def h5py_to_dataset(path):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1

    return TensorDataset(dataset, torch.zeros(len(dataset)))