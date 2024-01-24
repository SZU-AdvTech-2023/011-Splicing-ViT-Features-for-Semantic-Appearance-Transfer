from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class Trainer(nn.Module):
    def __init__(self, args, transformer, extractor, generator):
        super().__init__()
        self.generator = deepcopy(generator)
        self.optimizer = optim.Adam(self.generator.parameters(), lr=args.lr, betas=(.0, .99))
        self.extractor = extractor
        self.transformer = transformer
        self.app_wt = args.app_wt
        self.struct_wt = args.struct_wt
        self.id_wt = args.id_wt

    def transform(self, x):
        y = self.transformer.vit_transform(x)
        return y

    def app_loss(self, target, predict, layer=11):
        loss, batch = 0., target.shape[0]

        for tgt, pre in zip(self.transform(target), self.transform(predict)):
            with torch.no_grad():
                cls_target = self.extractor.get_cls_token_from_input(tgt.unsqueeze(0), layer)
            cls_predict = self.extractor.get_cls_token_from_input(pre.unsqueeze(0), layer)
            loss += F.mse_loss(cls_target, cls_predict)

        return loss / batch

    def struct_loss(self, target, predict, layer=11):
        loss, batch = 0., target.shape[0]

        for tgt, pre in zip(self.transform(target), self.transform(predict)):
            with torch.no_grad():
                struct_target = self.extractor.get_keys_self_sim_from_input(tgt.unsqueeze(0), layer)
            struct_predict = self.extractor.get_keys_self_sim_from_input(pre.unsqueeze(0), layer)
            loss += F.mse_loss(struct_target, struct_predict)

        return loss / batch

    def id_loss(self, target, predict, layer=11):
        loss, batch = 0., target.shape[0]

        for tgt, pre in zip(self.transform(target), self.transform(predict)):
            with torch.no_grad():
                id_target = self.extractor.get_keys_from_input(tgt.unsqueeze(0), layer)
            id_predict = self.extractor.get_keys_from_input(pre.unsqueeze(0), layer)
            loss += F.mse_loss(id_target, id_predict)

        return loss / batch

    def forward(self, tgt, src, layer):

        tgt_predict = self.generator(tgt)
        src_predict = self.generator(src)

        self.optimizer.zero_grad()

        loss_app = self.app_loss(tgt, src_predict, layer=layer) * self.app_wt
        loss_struct = self.struct_loss(src, src_predict, layer=layer) * self.struct_wt
        loss_id = self.id_loss(tgt, tgt_predict, layer=layer) * self.id_wt
        loss = loss_app + loss_struct + loss_id
        loss.backward()
        loss_dict = {'app': loss_app, 'struct': loss_struct, 'id': loss_id, 'loss': loss}

        self.optimizer.step()

        return loss_dict

    def save_model(self, path):
        torch.save(self.generator.state_dict(), path)
