import argparse
import os
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models.unet import UNet


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--ckpt', type=str)
parser.add_argument('--output', type=str, default='./outputs')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--source', type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = args.device
state_dict = torch.load(args.ckpt)
unet = UNet().to(device)
unet.load_state_dict(state_dict)

transform = transforms.ToTensor()

image = transform(Image.open(args.source).convert('RGB')).to(device).unsqueeze(0)
result = unet(image)
save_image(result, os.path.join(args.output, args.name+'.png'))
