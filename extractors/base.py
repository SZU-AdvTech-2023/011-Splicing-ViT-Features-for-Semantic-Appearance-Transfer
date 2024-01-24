import torch
from models import vision_transformer as vit


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


def cross_cos_sim(x, y, eps=1e-08):
    x, y = x[0], y[0]
    norm_x = x.norm(dim=2, keepdim=True)
    norm_y = y.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm_x @ norm_y.permute(0, 2, 1), min=eps)
    cross_sim_matrix = (x @ y.permute(0, 2, 1)) / factor
    return cross_sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]
    MODE = {'small': vit.vit_small, 'base': vit.vit_base, 'tiny': vit.vit_tiny}

    def __init__(self, mode, patch_size, pretrained=None, device='cuda'):
        self.device = device
        self.mode = mode
        self.patch_size = patch_size
        self.model = VitExtractor.MODE[mode](patch_size=patch_size)
        if pretrained is not None:
            print(f'load pretrained model from {pretrained}.')
            self.load(pretrained=pretrained)
        self.model.eval().to(device)

        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def load(self, pretrained):
        raise NotImplementedError

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return self.patch_size

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        return 6 if self.mode == 'small' else 12

    def get_embedding_dim(self):
        return 384 if self.mode == 'small' else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        # print(keys.shape)
        return keys

    def get_values_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        values = self.get_values_from_qkv(qkv_features, input_img.shape)
        return values

    def get_queries_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        queries = self.get_queries_from_qkv(qkv_features, input_img.shape)
        return queries

    def get_tokens_from_input(self, input_img, layer_num):
        tokens = self.get_feature_from_input(input_img)[layer_num]
        return tokens

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

    def get_values_self_sim_from_input(self, input_img, layer_num):
        values = self.get_values_from_input(input_img, layer_num=layer_num)
        h, t, d = values.shape
        concatenated_values = values.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_queries_self_sim_from_input(self, input_img, layer_num):
        queries = self.get_queries_from_input(input_img, layer_num=layer_num)
        h, t, d = queries.shape
        concatenated_values = queries.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_tokens_self_sim_from_input(self, input_img, layer_num):
        tokens = self.get_tokens_from_input(input_img, layer_num=layer_num)
        h, t, d = tokens.shape
        concatenated_values = tokens.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_attentions_from_input(self, input_img, layer_num, return_mean=True):
        attn_output_weights = self.get_attn_feature_from_input(input_img)[layer_num]
        attn_output_weights_mean = attn_output_weights.sum(dim=1) / self.get_head_num()
        return attn_output_weights, attn_output_weights_mean if return_mean else None

    def get_keys_cross_sim_from_input(self, source_img, target_img, layer_num):
        src_keys = self.get_keys_from_input(source_img, layer_num=layer_num)
        tgt_keys = self.get_keys_from_input(target_img, layer_num=layer_num)
        assert src_keys.shape == tgt_keys.shape
        h, t, d = src_keys.shape
        concatenated_src_keys = src_keys.transpose(0, 1).reshape(t, h*d)
        concatenated_tgt_keys = tgt_keys.transpose(0, 1).reshape(t, h*d)
        cross_sim_map = cross_cos_sim(concatenated_src_keys[None, None, ...], concatenated_tgt_keys[None, None, ...])
        return cross_sim_map

    def get_cls_token_from_input(self, input_img, layer_num):
        cls = self.get_feature_from_input(input_img)[layer_num][:, 0, :]
        return cls
