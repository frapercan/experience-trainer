import torch
import torch.nn as nn

# Tensor de entrada
tensor = torch.rand([11, 2, 56, 56])

# Define una capa ConvTranspose1d para hacer la transformación
conv_transpose = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
pool = nn.AvgPool2d(kernel_size=11, stride=11)
# Aplica la capa al tensor
resized_tensor = conv_transpose(tensor)
print(resized_tensor.shape)
resized_tensor = pool(resized_tensor)

print(resized_tensor.shape)  # Debería imprimir torch.Size([11, 56,
