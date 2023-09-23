import os
import numpy as np
import torch
from scipy import io
from typing import List, Optional, Tuple
from models_wrappers.models_wrapper_base import ModelWrapper
from max_entropy_utils import max_entropy_pdf
from moments_calculations import Moments

import matplotlib
matplotlib.use('Agg')
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "cmu-serif",
    "mathtext.fontset": "cm",
    # "font.size": 18
})


def plot_subspace_corr(subspace_corr: Optional[List], outdir: str):
    if subspace_corr is not None:
        torch.save(subspace_corr, os.path.join(outdir, 'subspace_corr.pth'))

        ev1_convergence = [x[0, 0] for x in subspace_corr][1:]
        ev2_convergence = [x[1, 1] for x in subspace_corr][1:]
        ev3_convergence = [x[2, 2] for x in subspace_corr][1:]

        plt.figure(figsize=(4, 2))
        plt.plot(ev1_convergence, label=r'$\boldsymbol{v}_1$')
        plt.plot(ev2_convergence, label=r'$\boldsymbol{v}_2$', color='red', linestyle=(0, (5, 10)))
        plt.plot(ev3_convergence, label=r'$\boldsymbol{v}_3$', color='black', linestyle=(0, (2, 6)))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'evs_convergence.pdf'), dpi=500, bbox_inches='tight')
        plt.close()


def plot_eigvecs(model: ModelWrapper,
                 im,
                 nim: torch.Tensor, fullnim: torch.Tensor,
                 patch: torch.Tensor, npatch: torch.Tensor, rpatch: torch.Tensor,
                 eigvecs: torch.Tensor, eigvals: torch.Tensor,
                 name: str, outdir: str, model_name: str, path_name: str,
                 max_entropy_params: Optional[Tuple[List[np.array], np.array, np.array]] = None,
                 amount: int = 1,
                 subspace_corr: Optional[List] = None,
                 max_axis: float = 3,
                 delta: float = 0.01):

    n_ev = eigvecs.shape[0]

    plot_subspace_corr(subspace_corr, outdir)

    fig = plt.figure(figsize=(30, 5*n_ev))
    rowmult = 1 + (max_entropy_params is not None)
    nrows = max(2, n_ev * rowmult)

    save_im = model.save_im
    toim = model.toim

    # Cols = 1 For original image, 1 for patch + noisy patch, 1 for evs, 1 for the MMSE, and 2*amount for the +- images
    imax = plt.subplot2grid((nrows, 4 + 2 * amount), (0, 0), rowspan=nrows)
    imax.imshow(toim(im), cmap='gray')
    imax.axis('off')
    imax.set_title('Original Image')
    imax = plt.subplot2grid((nrows, 4 + 2 * amount), (0, 1), rowspan=nrows//2)
    imax.imshow(toim(patch), cmap='gray')
    imax.axis('off')
    imax.set_title('Original Patch')
    imax = plt.subplot2grid((nrows, 4 + 2 * amount), (nrows//2, 1), rowspan=nrows-nrows//2)
    imax.imshow(toim(npatch), cmap='gray')
    imax.axis('off')
    imax.set_title('Noisy Patch')

    for row in range(n_ev):
        norm_stretch = max(abs(eigvecs[row].min()), abs(eigvecs[row].max()))
        eigvecs_normed = eigvecs[row] / (2 * norm_stretch) + 0.5

        # show eigvec
        eigvecs_show = toim(eigvecs_normed)
        ax = plt.subplot2grid((nrows, 4 + 2 * amount), (row * rowmult, 2))
        ax.imshow(eigvecs_show, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        ax.set_title(f'EigVec, Eigval: {eigvals[row]:.2E}')
        save_im(eigvecs_show, os.path.join(outdir, f'pc{row + 1}_eigvec.png'))

        # plot MMSE
        ax = plt.subplot2grid((nrows, 4 + 2 * amount), (row * rowmult, 3 + amount))
        ax.imshow(toim(rpatch), cmap='gray')
        ax.axis('off')
        ax.set_title('Restored Patch')

        steps = np.linspace(0, max_axis, amount + 1)[1:]
        for i, step in enumerate(steps):
            evup = (rpatch + (step * eigvals[row].sqrt() * eigvecs[row]))
            evdown = (rpatch - (step * eigvals[row].sqrt() * eigvecs[row]))

            ax = plt.subplot2grid((nrows, 4 + 2 * amount), (row * rowmult, 3 + amount + 1 + i))
            ax.imshow(toim(evup), cmap='gray')
            ax.axis('off')
            ax.set_title(fr'+ {step:.2g} * PC \#{row + 1}')

            ax = plt.subplot2grid((nrows, 4 + 2 * amount), (row * rowmult, 3 + amount - i - 1))
            ax.imshow(toim(evdown), cmap='gray')
            ax.axis('off')
            ax.set_title(fr'- {step:.2g} * PC \#{row + 1}')

            save_im(toim(evdown), os.path.join(outdir, f'pc{row+1}_evdown_{step:.2g}.png'))
            save_im(toim(evup), os.path.join(outdir, f'pc{row+1}_evup_{step:.2g}.png'))

        # plot pdf if possible
        if max_entropy_params is not None:
            z = max_entropy_params[0][row]
            ax = plt.subplot2grid((nrows, 4 + 2 * amount), (row*rowmult + 1, 3), colspan=amount * 2 + 1)

            if z is not None:
                yup_max = (max_entropy_params[1][row] + max_axis * np.sqrt(max_entropy_params[2][row]))
                ydown_max = (max_entropy_params[1][row] - max_axis * np.sqrt(max_entropy_params[2][row]))
                bit = 0.5
                abit = bit * np.sqrt(max_entropy_params[2][row])
                xs = np.arange(ydown_max - abit, yup_max + abit, delta)

                unnormed_pdf = max_entropy_pdf(z, xs, max_entropy_params[1][row])
                pdf = unnormed_pdf / (delta * sum(unnormed_pdf))

                xs = np.linspace(-max_axis - bit, max_axis + bit, len(xs))
                ax.plot(xs, pdf)

                xs = np.concatenate([-1 * steps[::-1], np.array([0]), steps])
                scatter_xs = max_entropy_params[1][row] + xs * np.sqrt(max_entropy_params[2][row])
                scatter_ys = max_entropy_pdf(z, scatter_xs, max_entropy_params[1][row])

                # ax.scatter(xs, scatter_ys, c='k')
                ax.scatter(xs, scatter_ys / (delta * sum(unnormed_pdf)), c='k')
            else:
                ax.text(0.5, 0.5, 'Optimization problem didn\'t converge',
                        ha='center', va='center', fontsize=20, color='red')

    plt.suptitle(f'eigvecs for {name} using {model_name}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(outdir, '..', path_name + '.png'))
    save_im(toim(rpatch), os.path.join(outdir, 'rpatch.png'))
    save_im(toim(npatch), os.path.join(outdir, 'npatch.png'))
    save_im(toim(patch), os.path.join(outdir, 'ppatch.png'))
    save_im(toim(im), os.path.join(outdir, 'im.png'))
    save_im(toim(nim), os.path.join(outdir, 'nim.png'))
    save_im(toim(fullnim), os.path.join(outdir, 'fullnim.png'))

    plt.close(fig)


def save_moments(moments: Moments, outdir: str, use_poly: bool):
    n_ev = len(moments.vmu1)

    io.savemat(os.path.join(outdir, 'moments.mat' if not use_poly else 'poly_moments.mat'),
               {'n_ev': n_ev,
                'vmu1': moments.vmu1,
                'vmu2': moments.vmu2,
                'vmu3': moments.vmu3,
                'vmu4': moments.vmu4})  # << That's the difference

    io.savemat(os.path.join(outdir, 'bigger_c_moments.mat' if not use_poly else 'bigger_c_poly_moments.mat'),
               {'n_ev': n_ev,
                'vmu1': moments.vmu1,
                'vmu2': moments.vmu2,
                'vmu3': moments.vmu3,
                'vmu4': moments.vmu4_other})  # << That's the difference


def save_eigvecs(eigvecs: torch.Tensor, eigvals: torch.Tensor, outdir: str):
    io.savemat(os.path.join(outdir, 'eigvecs.mat'),
               {'eigvecs': eigvecs.cpu().numpy().astype(np.float64),
                'eigvals': eigvals.cpu().numpy().astype(np.float64)})
