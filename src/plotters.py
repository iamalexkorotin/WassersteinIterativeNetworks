import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .tools import ewma, freeze

import torch
import gc
from torch import autograd

def plot_bar_images(Z, Ys, G, Ts, alphas):
    freeze(G);
    for T in Ts:
        freeze(T)
    with torch.no_grad():
        X = G(Z)
        T_X = torch.zeros_like(X)
        Ts_X = []
        for k in range(len(alphas)):
            Ts_X.append(Ts[k](X))
            T_X += alphas[k] * Ts_X[k]
    
    Ts_X_Ys = []
    for k in range(len(alphas)):
        Ts_X_Ys.append(Ts_X[k])
        Ts_X_Ys.append(Ys[k])
    with torch.no_grad():
        imgs = torch.cat([X, T_X, *Ts_X_Ys]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(2 + len(alphas) * 2, 10, figsize=(14.2, 8.5 * ((2 + 2 * len(alphas)) / 6)), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        if imgs[i].shape[2] == 3:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i][:, :, 0], cmap='gray')
            
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=16)
    axes[1, 0].set_ylabel('T(X)', fontsize=16)
    for k in range(len(alphas)):
        axes[2*k+2, 0].set_ylabel(f'T{k}(X)', fontsize=16)
        axes[2*k+3, 0].set_ylabel(f'Y{k}', fontsize=16)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_bar_random_images(Z_sampler, Y_samplers, G, Ts, alphas):
    Z = Z_sampler.sample(10)
    Ys = [Y_samplers[k].sample(10) for k in range(len(alphas))]
    return plot_bar_images(Z, Ys, G, Ts, alphas)

def plot_inv_images(Z, Ys, G, Ts_inv):
    freeze(G);
    for T_inv in Ts_inv:
        freeze(T_inv)
    with torch.no_grad():
        X = G(Z)
        Ts_inv_Y = []
        for T_inv, Y in zip(Ts_inv, Ys):
            Ts_inv_Y.append(T_inv(Y))
    
    imgs = [X]
    for k in range(len(Ts_inv)):
        imgs.append(Ys[k]); imgs.append(Ts_inv_Y[k])
        
    imgs = torch.cat(imgs).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(1 + len(Ts_inv) * 2, 10, figsize=(14.2, 8.5 * ((1 + 2 * len(Ts_inv)) / 6)), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        if imgs[i].shape[2] == 3:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i][:, :, 0], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
    
    for k in range(len(Ts_inv)):
        axes[0, 0].set_ylabel('X=G(Z)', fontsize=16)
        axes[2*k+1, 0].set_ylabel(f'Y{k}', fontsize=16)
        axes[2*k+2, 0].set_ylabel(r'$T^{-1}$' + f'{k}(Y)', fontsize=16)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_inv_random_images(Z_sampler, Y_samplers, G, Ts_inv):
    Z = Z_sampler.sample(10)
    Ys = [Y_samplers[k].sample(10) for k in range(len(Ts_inv))]
    return plot_inv_images(Z, Ys, G, Ts_inv)

def plot_training_phase(
    benchmark, pca, T_list, T_inv_list,
    G, Z_sampler,
    plot_batchsize=250, partsize=(3, 3), dpi=150
):
    fig, axes = plt.subplots(
        3, benchmark.num + 1,
        figsize=(partsize[0] * (benchmark.num+1), 3 * partsize[1]),
        sharex=True, sharey=True, dpi=dpi
    )
    
    # Original distributions, pushed and inverse pushed from G(Z)
    Z = Z_sampler.sample(plot_batchsize).detach()
    X = G(Z).detach()
    X.requires_grad_(True) 

    X_pca = pca.transform(X.cpu().detach().numpy())
    axes[1,-1].scatter(
        X_pca[:, 0], X_pca[:, 1], edgecolors='black', color='gold',
        label=r'$x\sim \mathbb{P}_{\xi}=G_{\xi}\sharp\mathbb{S}$',
    )
    X_push_sum = 0.
        
    for n in range(benchmark.num):
        # Input distributions
        Y = benchmark.samplers[n].sample(plot_batchsize)
        Y_pca = pca.transform(Y.cpu().detach().numpy())
        label = '$y\\sim \\mathbb{P}$' + f'$_{{{n}}}$'
        axes[0, n].scatter(
            Y_pca[:, 0], Y_pca[:, 1], edgecolors='black', color='lightblue',
            label=r'{}'.format(label),
        )
        
        # Maps from marginal distributions to the barycenters
        Y_push_inv = T_inv_list[n](Y).detach()
        Y_push_inv_pca = pca.transform(Y_push_inv.cpu().detach().numpy())
        pre_label = str(n) + '\\rightarrow\\xi'
        inv = 'inv'
        label = '$y\sim \widehat{T}$' + f'$^{{inv}}_{{{pre_label}}}$'
        axes[1, n].scatter(
            Y_push_inv_pca[:, 0], Y_push_inv_pca[:, 1], edgecolors='black', color='wheat',
            label=r'{}'.format(label),
        )

        # Maps to marginal distributions
        X_push = T_list[n](X).detach()
        X_push_pca = pca.transform(X_push.cpu().detach().numpy())
        with torch.no_grad():
            X_push_sum += benchmark.alphas[n] * X_push
        pre_label = '\\xi\\rightarrow' + str(n)
        label = '$y\\sim \\widehat{T}$' + f'$_{{{pre_label}}}$'
        axes[2, n].scatter(
            X_push_pca[:, 0], X_push_pca[:, 1], edgecolors='black', color='tan',
            label=r'{}'.format(label)
        )
    
    # Generator regression target
    X_push_sum_pca = pca.transform(X_push_sum.cpu().detach().numpy())
    axes[2, -1].scatter(
        X_push_sum_pca[:, 0], X_push_sum_pca[:, 1], edgecolors='black', color='red',
        label=r'$x\sim (\sum_n \alpha_{n}\widehat{T}_{\mathbb{P}_{\xi}\rightarrow\mathbb{P}_{n}})\sharp\mathbb{P}_{\xi}$'
    )
        
    # Ground truth barycenter
    Y = benchmark.bar_sampler.sample(plot_batchsize).cpu().detach().numpy()
    Y_pca = pca.transform(Y)
    axes[0, -1].scatter(
        Y_pca[:, 0], Y_pca[:, 1], edgecolors='black', color='green',
        label=r'$x\sim \overline{\mathbb{P}}$',
    )
    
    for ax in axes.flatten():
        ax.legend(fontsize=12, loc='lower right', framealpha=1)
            
    gc.collect()
    torch.cuda.empty_cache()
    fig.tight_layout(pad=0.01)
        
    return fig, axes