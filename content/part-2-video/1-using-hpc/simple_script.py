import torch
import torchvision as tv

img = tv.io.read_image("DTU_logo.png")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

tensor = img.to(device)

print(tensor.device)