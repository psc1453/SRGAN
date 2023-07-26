import torch
import argparse
from model import Generator
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

def main():
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    model = Generator(UPSCALE_FACTOR).eval()
    device = 'cpu'
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=device))
    # TODO: named_parameters() comments example wrong
    parameter_dict = model.named_parameters()
    for name, parameter in parameter_dict:
        if 'weight' in name and 'conv' in name:
            print(name, ': ', parameter.shape)
            min_val = torch.min(parameter).item()
            max_val = torch.max(parameter).item()
            channel_num = parameter.shape[0]
            kernel_list = []
            if channel_num > 80:
                plt.figure(figsize=(8, 200))
            else:
                plt.figure(figsize=(8, 120))
            for kernel_index in range(channel_num):
                kernel = parameter[kernel_index].detach().numpy().flatten()
                kernel_list.append(kernel)
                plt.subplot(channel_num, 1, kernel_index + 1)
                plt.hist(kernel, range=(min_val, max_val), bins=40)

            plt.show()


if __name__ == '__main__':
    main()