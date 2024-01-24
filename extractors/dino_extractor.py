from .base import VitExtractor


class DINOVitExtractor(VitExtractor):

    def load(self, pretrained):
        import torch
        self.model = torch.hub.load('facebookresearch/dino:main', model='dino_vitb8')

    def __init__(self, mode='base', patch_size=8, pretrained='./checkpoints/dino_vitbase8_pretrain.pth', device='cuda'):
        super().__init__(mode=mode, patch_size=patch_size, pretrained=pretrained, device=device)
