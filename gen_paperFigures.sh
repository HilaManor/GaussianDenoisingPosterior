#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py -e 3 -a 3 -d MNIST_n140.25 -c 1e-5 --var_c 1e-4 --skew_c 1e-1 --kurt_c 1e-1 -t 50  -o Outputs/DifferentNorm2 --model_zoo ./MNIST  -m 0 28 0 28 -p 100 -s 1 --mnist_choose_one 12 -v

CUDA_VISIBLE_DEVICES=0 python main.py -e 3 -p 1000 -t 50 -c 1e-5 -o Outputs/DifferentNorm2 -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/213.jpg" -m 70 180 60 240 -s 0
CUDA_VISIBLE_DEVICES=0 python main.py -e 4 -p 1000 -t 50 -c 1e-5 -o Outputs/DifferentNorm2 -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/514.jpg" -m 70 180 60 240 -s 8
CUDA_VISIBLE_DEVICES=0 python main.py -e 3 -p 1000 -t 50 -c 1e-5 -o Outputs/DifferentNorm2 -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/34.jpg" -m 70 180 60 240 -s 9
CUDA_VISIBLE_DEVICES=0 python main.py -e 3 -p 1000 -t 50 -c 1e-5 -o Outputs/DifferentNorm2 -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/227.jpg" -m 70 180 60 240 -s 1

CUDA_VISIBLE_DEVICES=0 python main.py -e 3 -d N2V -c 1e-5 --skew_c 1e-1 --kurt_c 1e-1 -t 100 -o Outputs/DifferentNorm2 --model_zoo "./FMD/TwoPhoton_MICE/raw/last_N2V_conv_twophoton_mice0.net" -m 80 140 260 320 -p 0 -i FMD/TwoPhoton_MICE/raw/ --fmd_choose_one 17 -v
CUDA_VISIBLE_DEVICES=0 python main.py -e 3 -d N2V -c 1e-5 --skew_c 1e-1 --kurt_c 1e-1 -t 100 -o Outputs/DifferentNorm2 --model_zoo "./FMD/Confocal_FISH/raw/last_N2V_conv_fish0.net" -m 40 120 160 240 -p 200 -i FMD/Confocal_FISH/raw/ --fmd_choose_one 4 -v -s 1

CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/CBSD68/227092.png   -e 4 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise25 -n 25 --skew_c 1e-2 --kurt_c 1e-2 -c 1e-5 -a 3 -t 50 -o Outputs/DifferentNorm2 -m 118 193 176 227 -s 111 -p 50 -v
CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/CBSD68/108005.png   -e 4 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise50 -n 50 --skew_c 1e-2 --kurt_c 1e-2 -c 1e-5 -a 3 -t 50 -o Outputs/DifferentNorm2 -m 180 260 97 189  -s 112 -p 50 -v 
CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/Kodak24/kodim23.png -e 5 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise25 -n 25 --skew_c 1e-2 --kurt_c 1e-2 -c 1e-5 -a 3 -t 50 -o Outputs/DifferentNorm2 -m 21 117 190 278  -s 113 --old_noise_selection -v


# SM

CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/ILSVRC2_2012_val/ILSVRC2012_val_00015913.jpeg -e 4 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise50 -n 50 -c 1e-5 --skew_c 1e-2 --kurt_c 1e-2 -a 3 -t 50 -o Outputs/DifferentNorm2_sup -m 300 370 55 125 -s 1 -p 50 -v
CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/ILSVRC2_2012_val/ILSVRC2012_val_00022113.jpeg -e 4 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise50 -n 50 -c 1e-5 --skew_c 1e-2 --kurt_c 1e-2 -a 3 -t 50 -o Outputs/DifferentNorm2_sup -m 350 450 50 150 -s 3 -p 50 -v
CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/Kodak24/kodim24.png                           -e 4 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise50 -n 50 -c 1e-5 --skew_c 1e-2 --kurt_c 1e-2 -a 3 -t 50 -o Outputs/DifferentNorm2_sup -m 180 280 150 250 -s 1 -p 50 -v
CUDA_VISIBLE_DEVICES=0 python main.py -i additional_testsets/McMaster/8.tif                                -e 4 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise50 -n 50 -c 1e-5 --skew_c 1e-2 --kurt_c 1e-2 -a 3 -t 50 -o Outputs/DifferentNorm3_sup -m 360 440 30 110  -s 1 -p 50 -v

CUDA_VISIBLE_DEVICES=0 python main.py -a 3 -e 4 -p 500 -t 50 -c 1e-5 -o Outputs/DiffNormSupFaces -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/199.jpg" -m 70 180 60 240 -s 7 --use_poly --poly_deg 6 -v
CUDA_VISIBLE_DEVICES=0 python main.py -a 3 -e 4 -p 500 -t 50 -c 1e-5 -o Outputs/DiffNormSupFaces -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/197.jpg" -m 70 180 60 240 -s 8 --use_poly --poly_deg 6 -v
CUDA_VISIBLE_DEVICES=0 python main.py -a 3 -e 4 -p 500 -t 50 -c 1e-5 -o Outputs/DiffNormSupFaces -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/194.jpg" -m 70 180 60 240 -s 3 --use_poly --poly_deg 6 -v
CUDA_VISIBLE_DEVICES=0 python main.py -a 3 -e 4 -p 500 -t 50 -c 1e-5 -o Outputs/DiffNormSupFaces -d DDPM_FFHQ_250 -i "DDPM_FFHQ/celeba_19/real/test/187.jpg" -m 70 180 60 240 -s 5 --use_poly --poly_deg 6 -v

# python main.py -i additional_testsets/CBSD68/108005.png   -e 6 -d 005_colorDN_DFWB_s128w8_SwinIR-M_noise50 -n 50 --skew_c 1e-2 --kurt_c 1e-2 -c 1e-5 -a 3 -t 50 -o Outputs/RebuttalFinal -m 180 260 97 189 -s 102  -p 50 -v 













