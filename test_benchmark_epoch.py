import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import generate_quantized_module
from ModelModifier.tools.quantization import quantize_model_parameters_with_original_scale, \
    quantize_tensor_with_original_scale

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()


def get_quant_model(model, weight=32, bias=32, conv=32):
    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model,
                                                                                  weight_width=weight,
                                                                                  bias_width=bias)
    mapping = NodeInsertMapping()
    quantize_8bit_function_package = FunctionPackage(quantize_tensor_with_original_scale, {'width': conv})
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, quantize_8bit_function_package)
    mapping.add_config(conv2d_config)

    new = generate_quantized_module(model_input=quantized_by_parameters_model, insert_mapping=mapping)
    return new


if __name__ == '__main__':

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
               'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

    model = Generator(UPSCALE_FACTOR).eval()
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        model = model.to(device)
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=device))

    model = get_quant_model(model, weight=9, bias=32, conv=32)

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        lr_image = lr_image.to(device)
        hr_image = hr_image.to(device)

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)

    out_path = 'statistics/'
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

    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
