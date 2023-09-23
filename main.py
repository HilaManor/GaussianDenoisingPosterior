import argparse
import os
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms as T
from interactive_utils import select_impatch
from models_loading import load_model
from plotting import plot_eigvecs, save_moments, save_eigvecs
from max_entropy_utils import calc_max_entropy_dist_params
from moments_calculations import get_eigvecs, calc_moments
import matplotlib
matplotlib.use('Agg')
# matplotlib.use("TkAgg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--amount', default=2, type=int,
                        help='Amount of sigmas to multiply the ev by (recommended 1-3)')
    parser.add_argument('-c', '--const', type=float, default=1e-6, help='Normalizing const for the power iterations')
    parser.add_argument('-e', '--n_ev', default=3, type=int, help='Number of eigenvectors to compute')
    parser.add_argument('-g', '--gpu_num', default=0, type=int, help='GPU device to use. -1 for cpu')
    parser.add_argument('-i', '--input', help='path to input file or input folder of files')
    parser.add_argument('-m', '--manual', nargs=4, default=None, type=int,
                        help='Choose a patch for uncertainty quantification in advanced, instead of choosing '
                             'interactively. Format: x1 x2 y1 y2.')
    parser.add_argument('-n', '--noise_std', type=float, default=None, help='Noise level to add to images')
    parser.add_argument('-o', '--outpath', default='Outputs', help='path to dump results')
    parser.add_argument('-p', '--padding', default=None, type=int,
                        help='The size of margin around the patch to insert to the model.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Set seed number')
    parser.add_argument('-t', '--iters', type=int, default=50, help='Amount of power iterations')
    parser.add_argument('-d', '--denoiser', help='name of the denoising model to use',
                        choices=['dncnn_15', 'dncnn_25', 'dncnn_50', 'dncnn_gray_blind', 'dncnn_color_blind',
                                 '004_grayDN_DFWB_s128w8_SwinIR-M_noise15', '004_grayDN_DFWB_s128w8_SwinIR-M_noise25',
                                 '004_grayDN_DFWB_s128w8_SwinIR-M_noise50', '005_colorDN_DFWB_s128w8_SwinIR-M_noise15',
                                 '005_colorDN_DFWB_s128w8_SwinIR-M_noise25', '005_colorDN_DFWB_s128w8_SwinIR-M_noise50',
                                 'ircnn_color', 'ircnn_gray',
                                 'N2V',
                                 'MNIST_n140.25',
                                 *[f'DDPM_FFHQ_{i}' for i in range(100, 500, 50)],
                                 # You can change the range to get whatever sigma you want.
                                 # I kept it (100,500,50) just to make sure the help menu isn't too big
                                 ],
                        default='dncnn_25')
    parser.add_argument('-v', '--marginal_dist', action='store_true',
                        help='Calc the marginal distribution along the evs (v\\mu_i)')
    parser.add_argument('--var_c', type=float, default=1e-5, help='Normalizing constant for 2rd moment approximation')
    parser.add_argument('--skew_c', type=float, default=1e-5, help='Normalizing constant for 3rd moment approximation')
    parser.add_argument('--kurt_c', type=float, default=1e-5, help='Normalizing constant for 4th moment approximation')
    parser.add_argument('--model_zoo', default='./KAIR/model_zoo', help='Directory of the models\' weights')
    parser.add_argument('--force_grayscale', action='store_true', help='Convert the image to gray scale')
    parser.add_argument('--low_acc', dest='double_precision', action='store_false',
                        help='Recomended when calculating only PCs (and not higher-order moments)')
    parser.add_argument('--use_poly', action='store_true',
                        help='Use a polynomial fit before calculating the derivatives for moments calculation')
    parser.add_argument('--poly_deg', type=int, default=6, help='The degree for the polynomial fit')
    parser.add_argument('--poly_bound', type=float, default=1, help='The bound around the MMSE for the polynomial fit')
    parser.add_argument('--poly_pts', type=int, default=30, help='The amount of points to use for the polynomial fit')
    parser.add_argument('--mnist_break_at', type=int, default=None, help='Stop iterating over MNIST at this index')
    parser.add_argument('--mnist_choose_one', type=int, default=None, help='Stop iterating over MNIST at this index')
    parser.add_argument('--fmd_choose_one', type=int, default=None, help='Choose a specific FOV from the FMD.')
    parser.add_argument('--old_noise_selection', action='store_true',
                        help='Deprecated. Only here to reproduce the paper\'s figures')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() and args.gpu_num != -1 else 'cpu')

    os.makedirs(args.outpath, exist_ok=True)

    model = load_model(args.denoiser, device, args.model_zoo, args.noise_std, args.double_precision)
    args.noise_std = model.set_std(noise_std=args.noise_std, denoiser_name=args.denoiser)

    if 'MNIST' in args.denoiser:
        from torchvision.datasets import MNIST
        test_set = MNIST(root='./MNIST/', train=False, download=True)
        # indexs = np.random.choice(len(test_set), 50)
        indexes = np.arange(len(test_set))
        orig_ims = [test_set.__getitem__(i)[0] for i in indexes]
        paths = [f'MNIST_{i}' for i in indexes]
        if args.mnist_choose_one is not None:
            for i in range(args.mnist_choose_one):  # For reproducing paper's graphs
                torch.randn((1, *orig_ims[0].size))
                torch.randn((args.n_ev, 1, *orig_ims[0].size), device=device, dtype=torch.double if args.double_precision else torch.float)
            orig_ims = [orig_ims[args.mnist_choose_one]]
            paths = [paths[args.mnist_choose_one]]
    elif model.is_FMD:
        noisy_ims = np.load(os.path.join(args.input, 'test_raw.npy')).astype(np.float64)
        orig_im = np.load(os.path.join(args.input, '..', 'gt', 'test_gt.npy')).astype(np.float64)[0]
        paths = [os.path.basename(os.path.abspath(os.path.join(args.input, '..'))) +
                 str(i) for i in range(len(noisy_ims))]
        if args.fmd_choose_one is not None:
            paths = [paths[args.fmd_choose_one]]
            noisy_ims = noisy_ims[args.fmd_choose_one][None, ...]
    else:
        if os.path.isdir(args.input):
            paths = [os.path.join(args.input, x) for x in os.listdir(args.input)]
        else:
            paths = [args.input]

    for i, path in enumerate(paths):
        print("**************************************\n"
              f"\t\t  {i}/{len(paths)}\n"
              "**************************************")
        if model.is_FMD:
            name = path
        elif 'MNIST' in args.denoiser:
            name = path
            orig_im = orig_ims[i]
        else:
            name = os.path.basename(path).split(os.extsep)[0]
            orig_im = Image.open(path)

            if args.force_grayscale:
                orig_im = ImageOps.grayscale(orig_im)

        outpath = os.path.join(args.outpath, f'{name}_s{args.seed}')

        selected_coords = select_impatch(orig_im, args.manual)
        p_rep = f"{int(np.floor(selected_coords['y1']))}-{int(np.ceil(selected_coords['y2']))}--" \
            f"{int(np.floor(selected_coords['x1']))}-{int(np.ceil(selected_coords['x2']))}"

        toTensor = T.ToTensor()

        # Add noise
        if args.old_noise_selection:
            im = toTensor(orig_im)
            sigma = (args.noise_std / 255.)
            im = im[:, selected_coords['y1']:selected_coords['y2'], selected_coords['x1']:selected_coords['x2']]

            nim = im + sigma * torch.randn_like(im)
            nim = nim.clip(0, 1)
            fullfull_nim = nim
            mask = torch.ones_like(im, device=device)
        else:
            if model.is_FMD:
                # The noise is natural
                sigma = None
                im = toTensor(orig_im[:, :, None]) / 255.
                nim = toTensor(noisy_ims[i][:, :, None]) / 255.
            else:
                im = toTensor(orig_im)
                if 'DDPM_FFHQ' in args.denoiser:
                    im = (im * 2) - 1
                    sigma = args.noise_std
                else:
                    sigma = (args.noise_std / 255.)
                nim = im + sigma * torch.randn_like(im)
                # if 'DDPM_FFHQ' not in args.denoiser:
                if 'MNIST' in args.denoiser:
                    nim = nim.clip(0, 1)

            fullfull_nim = nim
            mask = torch.zeros_like(im, device=device)
            mask[:, selected_coords['y1']:selected_coords['y2'], selected_coords['x1']:selected_coords['x2']] = 1

            if args.padding is None:
                sure = input(f"Are you sure you want to pass the entire {im.shape[1]}x{im.shape[2]} image through the "
                             "model? this might take a long time, depending on the model. (y/N) ")
                if sure != 'y':
                    exit(1)
            else:
                xmin = max(0, selected_coords['x1'] - args.padding // 2)
                ymin = max(0, selected_coords['y1'] - args.padding // 2)
                im = im[:, ymin:selected_coords['y2'] + args.padding // 2,
                        xmin:selected_coords['x2'] + args.padding // 2]
                nim = nim[:, ymin:selected_coords['y2'] + args.padding // 2,
                          xmin:selected_coords['x2'] + args.padding // 2]
                mask = mask[:, ymin:selected_coords['y2'] + args.padding // 2,
                            xmin:selected_coords['x2'] + args.padding // 2]

        os.makedirs(outpath, exist_ok=True)
        nim_full = nim

        if args.double_precision:
            nim = nim.to(torch.double)

        path_name = f'{name}_{args.denoiser}_n{args.noise_std}' \
                    f'_c{args.const:.1E}_2c{args.skew_c:.1E}_3c{args.kurt_c:.1E}_p{p_rep}_m{args.padding}'
        outdir = os.path.join(outpath, path_name)
        os.makedirs(outdir, exist_ok=True)

        # pylint: disable=unbalanced-tuple-unpacking
        eigvecs, eigvals, mmse, sigma, subspace_corr = get_eigvecs(model,
                                                                   nim,  # .unsqueeze(0),
                                                                   mask,  # .unsqueeze(0),
                                                                   args.n_ev,
                                                                   sigma,
                                                                   device,
                                                                   c=args.const, iters=args.iters,
                                                                   double_precision=args.double_precision)
        save_eigvecs(eigvecs, eigvals, outdir)

        max_entropy_params = None
        if args.marginal_dist:
            moments = calc_moments(model, nim, mask, sigma, device,
                                   mmse, eigvecs, eigvals,
                                   var_c=args.var_c, skew_c=args.skew_c, kurt_c=args.kurt_c,
                                   use_poly=args.use_poly, poly_deg=args.poly_deg,
                                   poly_bound=args.poly_bound, poly_pts=args.poly_pts,
                                   double_precision=args.double_precision)
            save_moments(moments, outdir, args.use_poly)
            max_entropy_params = calc_max_entropy_dist_params(moments)

        patch = im[mask.to(torch.bool).cpu()].reshape(
            mask.shape[0], int(np.floor(selected_coords['y2'])) - int(np.ceil(selected_coords['y1'])), -1)
        rpatch = mmse[mask.to(torch.bool)].reshape(
            mask.shape[0], int(np.floor(selected_coords['y2'])) - int(np.ceil(selected_coords['y1'])), -1)
        npatch = nim[mask.to(torch.bool).cpu()].reshape(
            mask.shape[0], int(np.floor(selected_coords['y2'])) - int(np.ceil(selected_coords['y1'])), -1)
        eigvecs = eigvecs[:, mask.to(torch.bool)].reshape(
            eigvecs.shape[0], mask.shape[0],
            int(np.floor(selected_coords['y2'])) - int(np.ceil(selected_coords['y1'])), -1)

        plot_eigvecs(model,
                     im, nim_full.cpu(), fullfull_nim.cpu(),
                     patch.cpu(), npatch.cpu(), rpatch.cpu(),
                     eigvecs.cpu(), eigvals.cpu(),
                     name, outdir, args.denoiser, path_name,
                     amount=args.amount,
                     max_entropy_params=max_entropy_params,
                     subspace_corr=subspace_corr)
