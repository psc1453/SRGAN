import argparse
import os
from math import log10

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import pytorch_ssim
from data_utils import TestDatasetFromFolder
from model import Generator

from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import insert_before
from ModelModifier.tools.hardware.bypass_bias import insert_bias_bypass, bypass_bias_adder
from ModelModifier.tools.quantization.utils import quantize_model_parameters_with_original_scale
from ModelModifier.tools.hardware.io import reshape_tensor_for_hardware_pe_input

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

# results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
#            'Urban100': {'psnr': [], 'ssim': []}}

results = {'BSD100': {'psnr': [], 'ssim': []}}


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


def get_quant_model(model, weight=8, bias=18, conv=18, width=18):
    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model,
                                                                                  weight_width=weight,
                                                                                  bias_width=bias)
    before_mapping = NodeInsertMapping()
    reshape_function_package = FunctionPackage(reshape_tensor_for_hardware_pe_input)
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
    before_mapping.add_config(conv2d_config)

    bypass_mapping = NodeInsertMapping()
    reshape_function_package = FunctionPackage(bypass_bias_adder, {'width': width})
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
    bypass_mapping.add_config(conv2d_config)

    new = insert_before(model_input=quantized_by_parameters_model, insert_mapping=before_mapping)
    # model = insert_after(model_input=model, insert_mapping=after_mapping)
    new = insert_bias_bypass(model_input=new, insert_mapping=bypass_mapping)
    return new


def main():
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    model = Generator(UPSCALE_FACTOR).eval()
    device = 'cpu'
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=device))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    plt.figure(figsize=(10, 6))
    plt.suptitle('Width of Accumulator vs PSNR (SRGAN)', fontsize=16)

    psnr_list = {}
    ssim_list = {}
    for accum in tqdm(range(1, 32)):
        model_quant = get_quant_model(model, width=accum)
        psnr, ssim = eval_one_epoch(model_quant, test_loader, device)
        psnr_list.update({accum: psnr})

        plt.subplot(1, 2, 1)
        plt.plot(psnr_list.keys(), psnr_list.values())
        plt.xlabel('Accumulator Width/Bits')
        plt.ylabel('PSNR/dB')
        plt.grid(alpha=0.4, linestyle=':')

        valuable_psnr_list = {k: v for k, v in psnr_list.items() if k in range(1, 33)}
        plt.subplot(1, 2, 2)
        plt.plot(valuable_psnr_list.keys(), valuable_psnr_list.values())
        plt.xlabel('Accumulator Width/Bits')
        plt.ylabel('PSNR/dB')
        plt.grid(alpha=0.4, linestyle=':')

        plt.show()


if __name__ == '__main__':
    main()
