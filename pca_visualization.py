from argparse import ArgumentParser
import PIL
from PIL import Image
import tqdm
import os
import numpy as np
import torch
import torchvision.utils
from torchvision import transforms as T
from sklearn.decomposition import PCA
from util import load_extractor


parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--image_path", type=str, default='./data/0001.png')
parser.add_argument("--layers", type=str, default='2,5,11',
                    help='Transformer layers from which to extract the feature, between 0-11.')
parser.add_argument('--facets', type=str, default='k,q,v,t',
                    help='Transformer facets from which to extract the feature, supporting `k`, `q`, `v`, `t`.')
parser.add_argument("--model_name", type=str, default='clip',
                    help='DINO: dino base 8; CLIP: clip base 16')
parser.add_argument('--model_mode', type=str, default='base')
parser.add_argument('--model_patch', type=int, default=16)
parser.add_argument("--save_path", type=str, default='./outputs/pca')

args = parser.parse_args()
device = args.device

os.makedirs(args.save_path, exist_ok=True)
save_path = os.path.join(args.save_path, f'{args.model_name}_ssim_'+os.path.basename(args.image_path))
preprocess = T.Compose([
    T.Resize(224),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# load vit extractor
vit_extractor = load_extractor(vit=args.model_name, mode=args.model_mode, patch_size=args.model_patch, device=device)

modes = args.facets.split(',')
layers = [int(i) for i in args.layers.split(',')]
pca_images = []

# load the image
input_img = Image.open(args.image_path).convert('RGB')
shape = input_img.size
input_img = T.Compose([
    T.Resize(224),
    T.ToTensor()
])(input_img).unsqueeze(0).to(device)

for mode in tqdm.tqdm(modes):
    for layer in layers:
        # calculate self-sim
        ssim = {
            'k': vit_extractor.get_keys_self_sim_from_input,
            'q': vit_extractor.get_queries_self_sim_from_input,
            'v': vit_extractor.get_values_self_sim_from_input,
            't': vit_extractor.get_tokens_self_sim_from_input
        }

        with torch.no_grad():
            input_img_ = preprocess(input_img)
            self_sim = ssim[mode](input_img_, layer)
            cross_sim = vit_extractor.get_keys_cross_sim_from_input(input_img_, input_img_, layer)

        pca = PCA(n_components=3)
        pca.fit(self_sim[0].cpu().numpy())
        components = pca.transform(self_sim[0].cpu().numpy())

        patch_h_num = vit_extractor.get_height_patch_num(input_img.shape)
        patch_w_num = vit_extractor.get_width_patch_num(input_img.shape)
        components = components[1:, :]
        components = components.reshape(patch_h_num, patch_w_num, 3)
        comp = components
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        pca_pil = pca_pil.resize(shape, resample=PIL.Image.NEAREST)
        pca_image = T.ToTensor()(pca_pil)
        pca_images.append(pca_image)

torchvision.utils.save_image(pca_images, save_path, nrow=len(layers))
