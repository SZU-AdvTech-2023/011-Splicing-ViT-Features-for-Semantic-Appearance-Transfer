import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def config(name='exp'):
    parser = argparse.ArgumentParser(name)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--vit', type=str, default='dino', help='vit model: dino or clip. Default is dino.')
    parser.add_argument('--mode', type=str, default='base', help='vit mode: base, small or tiny. Default is base.')
    parser.add_argument('--patch_size', type=int, default=8, help='vit patch size. Default is 8 for dino, 16 for clip.')
    parser.add_argument('--layer', type=int, default=11, help='use the features from layer. Default is 11 (the last).')
    parser.add_argument('--facet', type=str, default='Keys',
                        help='vit facet: keys, values, queries, tokens. Default is keys.')
    parser.add_argument('--save_each_iter', type=int, default=100)

    return parser


def view(args):
    print('-' * 15, 'Args INFO', '-' * 15)
    dictionary = args.__dict__
    m = max([len(i) for i in dictionary.keys()])
    for k in dictionary.keys():
        print(k, ' ' * (m - len(k) + 1), ':\t', dictionary[k])
    print('-' * 41)


def save_config(args, output_dir):
    import os
    import json
    import datetime

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args_dict = args.__dict__
    dictionary = {'args': args_dict, 'running_time': current_time}
    with open(os.path.join(output_dir, 'args.json'), 'w', encoding='utf-8') as file:
        json.dump(dictionary, file, ensure_ascii=False, indent=4)


class Crop(nn.Module):
    def __init__(self, ratio: list, batch=4):
        super().__init__()
        self.ratio = ratio
        self.batch = batch

    def forward(self, image):
        _, _, h, w = image.shape
        n = int(np.random.uniform(*self.ratio) * h)
        image_cropped = []
        crop = transforms.RandomCrop(min(n, w))
        for i in range(self.batch):
            image_cropped.append(crop(image))

        return torch.cat(image_cropped, dim=0)


class Transform:
    def __init__(self, batch=4):
        self.imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.norm = transforms.Normalize(0.5, 0.5)
        self.resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        self.crop = transforms.Compose([
            Crop([0.95, 1.0], batch=batch),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.is_augment = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=.4, saturation=.2, hue=.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            self.crop
        ])
        self.it_augment = self.crop

    def augment(self, src_image, tgt_image):
        aug_src_image = self.is_augment(src_image)
        aug_tgt_image = self.it_augment(tgt_image)
        return aug_src_image, aug_tgt_image

    def vit_transform(self, image):
        image = self.imagenet_norm(self.resize(image))
        return image


def load_extractor(vit: str, mode: str, patch_size: int, device='cuda'):
    from extractors import DINOVitExtractor, CLIPVitExtractor
    model = {'dino': DINOVitExtractor, 'clip': CLIPVitExtractor}
    vit_extractor = model[vit.lower()](patch_size=patch_size, mode=mode, device=device)

    return vit_extractor


def load_image(image_path):
    image = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
    return image.unsqueeze(0)
