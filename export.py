import onnx
import onnxsim
import torch
import netron

from model import Generator, Discriminator

generator_model_torch = Generator(scale_factor=4)
discriminator_model_torch = Discriminator()

torch.save(generator_model_torch, 'model_file/Generator_Pytorch.pth')
torch.save(discriminator_model_torch, 'model_file/Discriminator_Pytorch.pth')

input_name = ['input']
output_name = ['output']

input = torch.Tensor(torch.randn(1, 3, 240, 160))

torch.onnx.export(generator_model_torch, input, 'model_file/Generator_ONNX.onnx', input_names=input_name, output_names=output_name, verbose=True)
torch.onnx.export(discriminator_model_torch, input, 'model_file/Discriminator_ONNX.onnx', input_names=input_name, output_names=output_name, verbose=True)

generator_model_onnx = onnx.load('model_file/Generator_ONNX.onnx')
discriminator_model_onnx = onnx.load('model_file/Discriminator_ONNX.onnx')

(generator_model_onnx_simplified, check) = onnxsim.simplify(model=generator_model_onnx)
(discriminator_model_onnx_simplified, check) = onnxsim.simplify(model=discriminator_model_onnx)


onnx.save(generator_model_onnx_simplified, 'model_file/Generator_ONNX_simplified.onnx')
onnx.save(discriminator_model_onnx_simplified, 'model_file/Discriminator_ONNX_simplified.onnx')

# netron.start('model_file/Generator_ONNX_simplified.onnx')
# netron.start('model_file/Discriminator_ONNX_simplified.onnx')