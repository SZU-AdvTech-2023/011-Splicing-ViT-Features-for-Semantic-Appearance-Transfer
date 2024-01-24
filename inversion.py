import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from models.unet import UNet
from util import config, view, save_config, Transform, load_extractor, load_image, str2bool


def init():
    parser = config("Inversion")
    parser.add_argument('--use_cnn', type=str2bool, default=True,
                        help='optime a CNN if is True. Otherwise optimize noise.')
    parser.add_argument('--inv_type', type=str, default='cls', help='use `cls` or `ssim` feature for inversion.')
    parser.add_argument('--output', type=str, default='./outputs/inversion', help='results output directory.')
    parser.add_argument('--target', type=str, default='./data/0001.png', help='target image path')
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    view(args)

    output_dir = os.path.join(args.output,
                              args.vit + args.mode + str(args.patch_size) + '_'
                              + ('CLS' if args.inv_type.lower() == 'cls' else args.facet+'(ssim)')
                              + str(args.layer) + '_' + ('cnn' if args.use_cnn else 'noise')
                              + '_iter' + str(args.num_iter) + '_' + os.path.basename(args.target).split('.')[0])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'progress'), exist_ok=True)
    save_config(args, output_dir)

    torch.manual_seed(args.seed)
    device = args.device
    transform = Transform()

    image = load_image(args.target).to(device)
    save_image(image, os.path.join(output_dir, 'target.png'))

    vit_extractor = load_extractor(args.vit, args.mode, args.patch_size, device)

    get_features = {
        'keys': vit_extractor.get_keys_from_input,
        'tokens': vit_extractor.get_tokens_from_input,
        'values': vit_extractor.get_values_from_input,
        'queries': vit_extractor.get_queries_from_input
    }
    extractor = vit_extractor.get_cls_token_from_input if args.inv_type == 'cls' else get_features[args.facet.lower()]

    args.output_dir = output_dir
    return {'args': args, 'transform': transform, 'image': image, 'extractor': extractor}


def inverse(args, transform, image, extractor):
    device = args.device
    image_shape = image.shape
    layer = args.layer

    image_tf = transform.vit_transform(image)

    noise = torch.randn([1, args.depth, image_shape[-2], image_shape[-1]],
                        requires_grad=False if args.use_cnn else True, device=device)
    cnn = UNet(in_channels=args.depth).to(device)
    params = cnn.parameters() if args.use_cnn else [{'params': noise}]
    optimizer = optim.Adam(params=params, lr=args.lr)

    with torch.no_grad():
        target_feature = extractor(image_tf, layer_num=layer)

    def forward(x):
        return cnn(x) if args.use_cnn else F.sigmoid(x)

    loss_record = []
    pbar = tqdm(range(args.num_iter))
    for i in pbar:
        optimizer.zero_grad()

        predict_image = forward(noise)
        predict_image_tf = transform.vit_transform(predict_image)
        predict_feature = extractor(predict_image_tf, layer_num=layer)

        loss = F.mse_loss(predict_feature, target_feature)
        loss_record.append(loss.data)

        loss.backward()
        optimizer.step()

        if i % args.save_each_iter == 0:
            predict = forward(noise)
            save_image(predict, os.path.join(args.output_dir, 'progress', f'{i}'.zfill(4)+'.png'))
        pbar.set_description('%.6f' % loss)

    predict = forward(noise)
    save_image(predict, os.path.join(args.output_dir, 'inversion.png'))

    with open(os.path.join(args.output_dir, 'loss.txt'), 'w') as file:
        for v in loss_record:
            file.write('%.6f\n' % v)

    if args.use_cnn:
        torch.save(cnn.state_dict(), os.path.join(args.output_dir, 'unet.pt'))


if __name__ == '__main__':
    print('initializing ...')
    kwargs = init()
    print('start inversion.')
    inverse(**kwargs)
    print('Done.')
