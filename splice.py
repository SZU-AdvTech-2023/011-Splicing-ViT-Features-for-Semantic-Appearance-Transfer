import os
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from models.unet import UNet
from trainer import Trainer
from util import config, view, save_config, Transform, load_extractor, load_image


def init():
    parser = config('Splice training.')
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--target', type=str, default='./data/0019.png')
    parser.add_argument('--source', type=str, default='./data/0020.png')
    parser.add_argument('--app_wt', type=float, default=10.)
    parser.add_argument('--struct_wt', type=float, default=1.)
    parser.add_argument('--id_wt', type=float, default=1.)
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--add_raw_each', type=int, default=75)
    parser.add_argument('--lr', type=float, default=0.002)

    args = parser.parse_args()
    view(args)

    output_dir = os.path.join(args.output,
                              'src' + os.path.basename(args.source).split('.')[0]
                              + '_tgt' + os.path.basename(args.target).split('.')[0]
                              + '_iter' + str(args.num_iter))
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'temp'), exist_ok=True)
    save_config(args, output_dir)

    device = args.device
    source = load_image(args.source).to(device)
    target = load_image(args.target).to(device)
    save_image(source, os.path.join(output_dir, 'source.png'))
    save_image(target, os.path.join(output_dir, 'target.png'))
    image = {'src': source, 'tgt': target}

    transform = Transform(batch=args.batch)
    extractor = load_extractor(args.vit, args.mode, args.patch_size, device)

    return {'args': args, 'transform': transform, 'image': image, 'extractor': extractor}


def train(args, transform, image, extractor):
    device = args.device
    generator = UNet().to(device)
    source, target = image['src'], image['tgt']

    trainer = Trainer(args, transform, extractor, generator)

    pbar = tqdm(range(args.num_iter))
    for i in pbar:
        # augment
        aug_src_images, aug_tgt_images = transform.augment(source, target)

        loss_dict = trainer(aug_tgt_images, aug_src_images, args.layer)
        description = 'app: %.5f  struct: %.5f  id: %.5f  loss: %.5f' % (loss_dict['app'], loss_dict['struct'],
                                                                         loss_dict['id'], loss_dict['loss'])

        if i % args.add_raw_each == 0:
            _ = trainer(target, source, args.layer)

        if i % args.save_each_iter == 0:
            trainer.save_model(os.path.join(args.output_dir, 'temp', f'{i}'.zfill(4)+'.pt'))
            with torch.no_grad():
                predict = trainer.generator(source)
                predict_crop = trainer.generator(aug_src_images[0:1, ...])
                save_image(torch.cat([source, predict], dim=0),
                           os.path.join(args.output_dir, 'temp', f'{i}'.zfill(4)+'.png'))
                save_image(torch.cat([aug_src_images[0:1, ...], predict_crop], dim=0),
                           os.path.join(args.output_dir, 'temp', 'crop'+f'{i}'.zfill(4)+'.png'))

        pbar.set_description(description)

    predict = trainer.generator(source)
    save_image(torch.cat([source, predict], dim=0), os.path.join(args.output_dir, 'predict.png'))

    trainer.save_model(os.path.join(args.output_dir, 'generator.pt'))


if __name__ == '__main__':
    print('initializing ...')
    kwargs = init()
    print('start inversion.')
    train(**kwargs)
    print('Done.')
