import clip
from .base import VitExtractor


class CLIPVitExtractor(VitExtractor):
    def load(self, pretrained):
        model, preprocess = clip.load(pretrained)
        # model, preprocess = clip.load('ViT-B/16')
        model = model.to(self.device)
        self.model = model.visual

    def __init__(self, mode='base', patch_size=16,
                 pretrained='./checkpoints/clip_vitbase16_pretrain.pt', device='cuda'):
        super().__init__(mode=mode, patch_size=patch_size, pretrained=pretrained, device=device)
