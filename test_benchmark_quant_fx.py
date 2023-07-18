import argparse
import os
import copy
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qconfig_mapping, quantize_fx
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')

if __name__ == '__main__':
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
               'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

    model_fp = Generator(UPSCALE_FACTOR).eval()
    model_fp.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location='cpu'))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    model_to_quantize = copy.deepcopy(model_fp)
    model_to_quantize.eval()

    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    print(test_loader)
    example_inputs = tuple(data[1] for data in test_loader)
    calibration_bar = tqdm(test_loader, desc='[calibrate quantized model]')

    model_prepared = quantize_fx.prepare_fx(model=model_to_quantize, qconfig_mapping=qconfig_mapping,
                                            example_inputs=example_inputs)

    for data in calibration_bar:
        image_name, lr_image, hr_restore_img, hr_image = data
        model_prepared(lr_image)

    model_quantized = quantize_fx.convert_fx(model_prepared)

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '_quant' + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]

        sr_image = model_quantized(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.squeeze(0)),
             display_transform()(sr_image.data.squeeze(0))])
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
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_quant_results.csv', index_label='DataSet')
