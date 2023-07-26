import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import insert_after
from ModelModifier.tools.quantization.utils import quantize_model_parameters_with_original_scale, \
    quantize_tensor_with_original_scale

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}}


def eval_one_epoch(model, dataloader, device):
    test_bar = tqdm(dataloader, desc='[testing benchmark datasets]')
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)

    saved_results = {'psnr': [], 'ssim': []}
    for item in results.values():
        psnr = np.array(item['psnr'])
        ssim = np.array(item['ssim'])
        if (len(psnr) == 0) or (len(ssim) == 0):
            psnr = 'No data'
            ssim = 'No data'
        else:
            psnr = psnr.mean()
            ssim = ssim.mean()
        saved_results['psnr'].append(psnr)
        saved_results['ssim'].append(ssim)

    psnr = np.mean(saved_results['psnr'])
    ssim = np.mean(saved_results['ssim'])
    return (psnr, ssim)


def get_quant_model(model, weight=32, bias=32, conv=32):
    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model,
                                                                                  weight_width=weight,
                                                                                  bias_width=bias,
                                                                                  by_channel=True)
    mapping = NodeInsertMapping()
    quantize_8bit_function_package = FunctionPackage(quantize_tensor_with_original_scale, {'width': conv})
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, quantize_8bit_function_package)
    mapping.add_config(conv2d_config)

    new = insert_after(model_input=quantized_by_parameters_model, insert_mapping=mapping)
    return new


def main():
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    model = Generator(UPSCALE_FACTOR).eval()
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        model = model.to(device)
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=device))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    plt.figure(figsize=(10, 12))
    plt.suptitle('Width of Weight vs PSNR (SRGAN)', fontsize=16)

    psnr_list = {}
    ssim_list = {}
    for weight_w in tqdm(range(1, 65)):
        model_quant = get_quant_model(model, weight=weight_w)
        psnr, ssim = eval_one_epoch(model_quant, test_loader, device)
        psnr_list.update({weight_w: psnr})

    plt.subplot(3, 2, 1)
    plt.plot(psnr_list.keys(), psnr_list.values())
    plt.xlabel('Weight Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    valuable_psnr_list = {k: v for k, v in psnr_list.items() if k in range(1, 33)}
    plt.subplot(3, 2, 2)
    plt.plot(valuable_psnr_list.keys(), valuable_psnr_list.values())
    plt.xlabel('Weight Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')
    plt.show()
    psnr_list = {}
    ssim_list = {}
    for bias_w in tqdm(range(1, 33)):
        model_quant = get_quant_model(model, bias=bias_w)
        psnr, ssim = eval_one_epoch(model_quant, test_loader, device)
        psnr_list.update({bias_w: psnr})

    plt.subplot(3, 2, 3)
    plt.plot(psnr_list.keys(), psnr_list.values())
    plt.xlabel('Bias Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    valuable_psnr_list = {k: v for k, v in psnr_list.items() if k in range(1, 11)}
    plt.subplot(3, 2, 4)
    plt.plot(valuable_psnr_list.keys(), valuable_psnr_list.values())
    plt.xlabel('Bias Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    psnr_list = {}
    ssim_list = {}
    for conv_w in tqdm(range(1, 65)):
        model_quant = get_quant_model(model, conv=conv_w)
        psnr, ssim = eval_one_epoch(model_quant, test_loader, device)
        psnr_list.update({conv_w: psnr})

    plt.subplot(3, 2, 5)
    plt.plot(psnr_list.keys(), psnr_list.values())
    plt.xlabel('Conv Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    valuable_psnr_list = {k: v for k, v in psnr_list.items() if k in range(1, 31)}
    plt.subplot(3, 2, 6)
    plt.plot(valuable_psnr_list.keys(), valuable_psnr_list.values())
    plt.xlabel('Conv Width/Bits')
    plt.ylabel('PSNR/dB')
    plt.grid(alpha=0.4, linestyle=':')

    plt.show()


if __name__ == '__main__':
    main()
