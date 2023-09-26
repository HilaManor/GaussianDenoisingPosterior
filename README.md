[![Python 3.8.0](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.24.3+-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.24.3/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.7.1)
[![Notebook](https://img.shields.io/badge/notebook-6.5.4+-green?logo=jupyter&logoColor=white)](https://pypi.org/project/notebook/6.5.4)
[![torch](https://img.shields.io/badge/torch-2.0.0+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15.1+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)

# On the Posterior Distribution in Denoising: Application to Uncertainty Quantification Official Implementation

[Project page](https://HilaManor.github.io/GaussianDenoisingPosterior) | [Arxiv](https://arxiv.org/abs/2309.13598) | [Supplementary materials](https://hilamanor.github.io/GaussianDenoisingPosterior/resources/supp.pdf)


https://github.com/HilaManor/GaussianDenoisingPosterior/assets/53814901/2b662ea6-ea42-4469-b53b-fb85729863bd


## Table of Contents

* [Requirements](#requirements)
* [Usage Example](#usage-example)
* [Citation](#citation)

## Requirements

```bash
python -m pip install -r requirements.txt
```

Note that for DDPM (faces), their code uses Open-MPI which sometimes have problems installing on machines with conda installed.

### Pre-trained Models

We support a number of pre-trained models, and one can clone their repo and download their checkpoints as necessary.

#### MNIST

As this is a simple network we built and trained, the checkpoint is already included in the repo.

#### KAIR

This repo contains the implementations for multiple denoisers that we support: `DnCNN`, `IRCNN`, `SwinIR`.

1. Clone their repo (provide this path to the `--model_zoo` parameter)

    ```bash
    git clone https://github.com/cszn/KAIR.git 
    ```

2. Follow the instructions in `KAIR/model_zoo/README.md` to download the wanted checkpoints or [here](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) for SwinIR.
    * We used `colorDN_DFWB_s128w8_SwinIR-M` models, but the interface *should* be able to use most versions)

#### Noise2Void

The original official implementation of Noise2Void is not in python. Nevertheless, the authors later published [Probabilistic Noise2Void](https://arxiv.org/abs/1906.00651), and with its [(python) GitHub implementation](https://github.com/juglab/pn2v) also included the python version of N2V.

This version of the code needs some fixes in their original code, and therefore is provided in the repo. Our trained checkpoint over the FMD data and the training notebooks are also included in the local pn2v repo.

1. Run the `GetData` notebook to download the [FMD dataset](https://github.com/yinhaoz/denoising-fluorescence/tree/master), extract the images and preprocess them for n2v.

#### DDPM (faces)

We use [Label-Efficient Semantic Segmentation with Diffusion Models](https://github.com/yandex-research/ddpm-segmentation)'s checkpoint for DDPM trained on the entire FFHQ dataset, and tested on celebA (as is usually done with the faces domain and diffusion models).

The relevant version of [guided_diffusion](https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924) used is already included in this repo, and therefore:

1. Follow their `download_checkpoint.sh` to download `ffhq.pt`, and place it in `DDPM_FFHQ`.
2. If needed, follow their `download_datasets.sh` to download celebA images.

## Usage Example

```bash
python main.py -e <number of eigenvectors> -p <context size around the patch> -t <subspace iters> -c <small constant> -o <output folder> -d <denoiser model> -i <input image path>
```

Use `--help` for more information on the parameters and other options, such as `low_acc` for finding EVs only quickly (without calculating the moments for the marginal distribution), or `use_poly` to try and fit a polynomial for the moments calculation.

Use `-v` to calculate the higher-order moments and estimate the density along the PCs.

## More Examples



https://github.com/HilaManor/GaussianDenoisingPosterior/assets/53814901/7d209056-83a7-4fbb-89de-af814cacd6a4



## [Citation](#citation)

If you use this code for your research, please cite our paper:

```
@article{manor2023posterior,
    title={On the Posterior Distribution in Denoising: Application to Uncertainty Quantification},
    author={Hila Manor and Tomer Michaeli}, journal={arXiv preprint arXiv:2305.10124},
    journal={arXiv preprint arXiv:2309.13598},
    year={2023},
}
```
