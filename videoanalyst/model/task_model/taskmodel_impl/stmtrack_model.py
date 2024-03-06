# -*- coding: utf-8 -*

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
from videoanalyst.model.utils.TransformerModel import Transformer, build_transformer
from videoanalyst.model.utils.transformer_utils import nested_tensor_from_tensor, build_position_encoding
torch.set_printoptions(precision=8)

bands_num_material = 6
class SELayerMaterial(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerMaterial, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, bands_num_material, bias=False),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x= torch.cat(x, dim=1)
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        w = self.fc(y).view(b, bands_num_material)
        return w
bands_num_hsi = 14
class SELayerHSI(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerHSI, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, bands_num_hsi, bias=False),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x= torch.cat(x, dim=1)
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        w = self.fc(y).view(b, bands_num_hsi)
        return w

param_dic = {}
param_dic['HIDDEN_DIM'] = 512
param_dic['DEC_LAYERS'] = 1
param_dic['DIM_FEEDFORWARD'] = 1024
param_dic['DIVIDE_NORM'] = False
param_dic['DROPOUT'] = 0.1
param_dic['ENC_LAYERS'] = 1
param_dic['NHEADS'] = 8
param_dic['PRE_NORM'] = False


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class STMTrack(ModuleBase):

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                corr_fea_output=False,
                                amp=False)

    support_phases = ["train", "memorize", "track"]

    def __init__(self, backbone_m, backbone_q, neck_m, neck_q, head, loss=None, debugTM='v1'):
        super(STMTrack, self).__init__()
        self.basemodel_m = backbone_m
        self.basemodel_q = backbone_q
        self.neck_m = neck_m
        self.neck_q = neck_q
        self.head = head
        self.loss = loss
        self.senet_layer_Material = SELayerMaterial(512*(bands_num_material))
        self.senet_layer_hsi = SELayerHSI(512*(bands_num_hsi))
        self.senet_layer_fuse = nn.Parameter(torch.ones(2))
        self.transformer = build_transformer(param_dic, debugTM=debugTM)
        self.position_embedding = build_position_encoding()
        self._phase = "train"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def memorize(self, im_crop, fg_bg_label_map):
        fm = self.basemodel_m(im_crop, fg_bg_label_map)
        fm = self.neck_m(fm)
        fm = fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous() 
        return fm

    def _split_Channel(self, feat_channel, mode='HSI'):
        if mode == 'HSI':
            res = []
            b,c,w,h = feat_channel.size()
            assert c-2 == bands_num_hsi
            for i in range(c-2):
                res.append(feat_channel[:,i:i+3,:,:])
            return res
        else:
            res = []
            b,c,w,h = feat_channel.size()
            assert c == bands_num_material
            for i in range(c):
                res.append(feat_channel[:,i:i+1,:,:].expand(b,3,w,h))
            return res
    
    def get_weight_feature(self, material, mode='HSI'): 
        if mode == 'HSI':
            weight = self.senet_layer_hsi(material) 
        elif mode == 'Material':
            weight = self.senet_layer_Material(material) 
        else:
            raise Exception

        batch_num, channel_num = weight.size()
        arr = material
        res = []
        for b in range(batch_num):
            mm = []
            for kk in range(channel_num): 
                mm.append(arr[kk][b:b+1,:,:,:])
            w = weight[b, :]; tmp = 0;
            for k in range(channel_num): tmp += mm[k]*w[k]
            res.append(tmp)
        material = torch.cat(res, dim=0)
        return material

    def getFeature(self, memory_img_arr, target_fg_bg_label_map=None, mode='template', method='Material'):
        temp_arr = []
        memory_img_arrArr = self._split_Channel(memory_img_arr, mode=method)

        for memory_img in memory_img_arrArr:
            if mode == 'template':
                fm_hsi = self.basemodel_m(memory_img, target_fg_bg_label_map)
                fm_hsi = self.neck_m(fm_hsi) 
            elif mode == 'search':
                fm_hsi = self.basemodel_q(memory_img) 
                fm_hsi = self.neck_q(fm_hsi) 
            temp_arr.append(fm_hsi)
        fea_fuse = temp_arr 
        return self.get_weight_feature(fea_fuse, mode=method)
    
    def cross_modality_fuse(self, zf_hsi, zf_material, zf_fc,  mask, pos, param=None):
        src_hsi = zf_hsi
        mask_hsi = mask
        pos_hsi = pos

        src_fc= zf_fc 
        mask_fc= mask
        pos_fc = pos
        b,c,w,h = src_fc.size()

        src_material = zf_material
        mask_material = mask
        pos_material = pos

        src_fc_flat = src_fc.flatten(2).permute(2, 0, 1) 
        mask_fc_flat = mask_fc.flatten(1) 
        pos_fc_flat = pos_fc.flatten(2).permute(2, 0, 1) 
        
        src_material_flat = src_material.flatten(2).permute(2, 0, 1) 
        mask_material_flat = mask_material.flatten(1) 
        pos_material_flat = pos_material.flatten(2).permute(2, 0, 1) 
        
        src_hsi_flat = src_hsi.flatten(2).permute(2, 0, 1) 
        mask_hsi_flat = mask_hsi.flatten(1) 
        pos_hsi_flat = pos_hsi.flatten(2).permute(2, 0, 1) 

        encoder_fc = self.transformer.encoder(src_fc_flat, src_key_padding_mask = mask_fc_flat, pos = pos_fc_flat)
        encoder_material = self.transformer.encoder(src_material_flat, src_key_padding_mask = mask_material_flat, pos = pos_material_flat)
        encoder_hsi = self.transformer.encoder(src_hsi_flat, src_key_padding_mask = mask_hsi_flat, pos = pos_hsi_flat)

        decoder_res_material_fc = self.transformer.decoder(encoder_fc, encoder_material, tgt_key_padding_mask=mask_fc_flat, memory_key_padding_mask = mask_material_flat, pos_enc=pos_material_flat, pos_dec=pos_fc_flat)
        decoder_res_hsi_fc = self.transformer.decoder(encoder_hsi, encoder_material, tgt_key_padding_mask=mask_hsi_flat, memory_key_padding_mask = mask_material_flat, pos_enc=pos_material_flat, pos_dec=pos_hsi_flat)

        decoder_res_hsi_fc = decoder_res_hsi_fc.squeeze(0).permute(1, 2, 0)
        decoder_res_material_fc = decoder_res_material_fc.squeeze(0).permute(1, 2, 0)

        decoder_res_hsi_fc = decoder_res_hsi_fc.view(b,c,w,h) 
        decoder_res_material_fc = decoder_res_material_fc.view(b,c,w,h) 

        mode = 'manual_weighted_fuse'
        if mode == 'append':
            cross_modality_res = []
            cross_modality_res.append(tfeat_fc)
            cross_modality_res.append(zf_material)
            cross_modality_res = torch.cat(cross_modality_res, dim=1)
        elif mode == 'auto_weighted_fuse':
            weight = self.senet_layer_fuse([decoder_res_hsi_fc, decoder_res_material_fc]) 
            batch_num, channel_num = weight.size()
            res = []
            for b in range(batch_num):
                tmp = decoder_res_hsi_fc[b:b+1,:,:,:] * weight[b, 0] + decoder_res_material_fc[b:b+1,:,:,:] * (1-weight[b, 0])
                res.append(tmp)
            cross_modality_res = torch.cat(res, dim=0)

        elif mode == 'manual_weighted_fuse':
            tmpw = F.softmax(self.senet_layer_fuse, dim=-1)
            cross_modality_res = decoder_res_hsi_fc*tmpw[0] + decoder_res_material_fc*tmpw[1]
        
        return cross_modality_res

    def getFCFeatureWithPos_Mask(self, ori_data, target_fg_bg_label_map, mode='template'):
        tensor, mask = nested_tensor_from_tensor(ori_data)
        if mode == 'template':
            feat = self.basemodel_m(tensor, target_fg_bg_label_map)
            feat = self.neck_m(feat)
        elif mode == 'search':
            feat = self.basemodel_q(tensor)
            feat = self.neck_q(feat)
        else:
            print ('======== no this mode========')
            raise Exception
        mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(None, tensorParam=feat, maskParam=mask).to(feat.dtype)
        return feat, mask, pos

    def train_forward(self, training_data):
        memory_img_fc = training_data["im_m"]['fc']
        memory_img_hsi = training_data["im_m"]['hsi']
        memory_img_material = training_data["im_m"]['material']
        query_img_fc = training_data["im_q"]['fc']
        query_img_hsi = training_data["im_q"]['hsi'] 
        query_img_material = training_data["im_q"]['material'] 
        # backbone feature
        B, C, H, W = memory_img_hsi.shape
        target_fg_bg_label_map = training_data["fg_bg_label_map"]
        
        fm_material = self.getFeature(memory_img_material, target_fg_bg_label_map, mode='template', method='Material')
        fm_hsi = self.getFeature(memory_img_hsi, target_fg_bg_label_map, mode='template', method='HSI')
        fm_fc, fm_mask, fm_pos = self.getFCFeatureWithPos_Mask(memory_img_fc, target_fg_bg_label_map, mode='template')

        fq_material = self.getFeature(query_img_material, mode='search', method='Material')
        fq_hsi = self.getFeature(query_img_hsi, mode='search', method='HSI')
        fq_fc, fq_mask, fq_pos = self.getFCFeatureWithPos_Mask(query_img_fc, None, mode='search')

        fm = self.cross_modality_fuse(fm_hsi, fm_material, fm_fc, fm_mask, fm_pos) 
        fq = self.cross_modality_fuse(fq_hsi, fq_material, fq_fc, fq_mask, fq_pos) 
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(fm, fq)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def forward(self, *args, phase=None):
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])
        elif phase == 'memorize':
            pass
        elif phase == 'template':
            data_fc_t, data_hsi_t, data_material_t, fg_bg_label_map = args
            ft_material = self.getFeature(data_material_t, fg_bg_label_map, mode='template', method='Material')
            ft_hsi = self.getFeature(data_hsi_t, fg_bg_label_map, mode='template', method='HSI')
            
            ft_fc, ft_mask, ft_pos = self.getFCFeatureWithPos_Mask(data_fc_t, fg_bg_label_map, mode='template')
            return ft_fc, ft_mask, ft_pos, ft_hsi, ft_material

        elif phase == 'track':
            search_fc, search_hsi, search_material, ft_fc, ft_mask, ft_pos, ft_hsi, ft_material= args
            fq_material = self.getFeature(search_material, mode='search', method='Material')
            fq_hsi = self.getFeature(search_hsi, mode='search', method='HSI')
            
            fq_fc, fq_mask, fq_pos = self.getFCFeatureWithPos_Mask(search_fc, None, mode='search')

            fm = self.cross_modality_fuse(ft_hsi, ft_material, ft_fc, ft_mask, ft_pos) 
            fq = self.cross_modality_fuse(fq_hsi, fq_material, fq_fc, fq_mask, fq_pos) 

            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                fm, fq, search_hsi.size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

            extra = dict()
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")
        return out_list

    def update_params(self):
        self._make_convs()
        self._initialize_conv()
        super().update_params()

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
