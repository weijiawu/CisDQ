# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from einops import repeat
from params import params

version = params['version']

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if version == 'baseline':
    # baseline
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                query_embed = torch.cat([query_embed, incre_query_embed], 0)
                output = torch.cat([output, incre_query_feat], 0)

            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                pad_outputs_mask = list(predictions_mask[-1].values())
                pad_outputs_mask = torch.cat(pad_outputs_mask,1)
                bs, l, _, _ = pad_outputs_mask.shape

                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6
                s = self.splits
                for k, v in predictions_class[-1].items():
                    cstep = int(k[4:])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, s[cstep]:s[cstep+1], :self.num_classes+1]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.mask_embed(decoder_output[:,s[i]:s[i+1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.incre_mask_embed_list[i - 1](decoder_output[:,s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'ab1':
    # ab1 ablation of incremental query
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            predictions_class = []
            predictions_mask = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class, 'predictions_mask':predictions_mask},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                pad_outputs_mask = list(predictions_mask[-1].values())
                pad_outputs_mask = torch.cat(pad_outputs_mask,1)
                bs, l, _, _ = pad_outputs_mask.shape

                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6
                s = self.splits
                st = 0
                for k, v in predictions_class[-1].items():
                    cstep = int(k[4:])
                    if cstep==0:
                        t = outputs_class[k].shape[1]
                        pad_outputs_class[:, st:st+t, :self.num_classes+1] = outputs_class[k][:, s[cstep]:s[cstep+1], :self.num_classes+1]
                        st = st+t
                    else:
                        t = outputs_class[k].shape[1]
                        pad_outputs_class[:, st:st+t, 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, st:st+t, self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]
                        st = st + t

                out = {
                    'kd_features': {'predictions_class': predictions_class, 'predictions_mask': predictions_mask},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output)
                    mask_embed = self.mask_embed(decoder_output)
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output)
                    mask_embed = self.mask_embed(decoder_output)
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, attn_masks[0]

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'ab2':
    # ab2 ablation of maskembed
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                query_embed = torch.cat([query_embed, incre_query_embed], 0)
                output = torch.cat([output, incre_query_feat], 0)

            # nq, bs, _ = output.shape
            # incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class, 'predictions_mask':predictions_mask},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                pad_outputs_mask = list(predictions_mask[-1].values())
                pad_outputs_mask = torch.cat(pad_outputs_mask,1)
                bs, l, _, _ = pad_outputs_mask.shape

                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6
                s = self.splits
                for k, v in predictions_class[-1].items():
                    cstep = int(k[-1])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, s[cstep]:s[cstep+1], :self.num_classes+1]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]

                out = {
                    'kd_features': {'predictions_class': predictions_class, 'predictions_mask': predictions_mask},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.mask_embed(decoder_output[:,s[i]:s[i+1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.mask_embed(decoder_output[:,s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'ab3':
    # ab3 attention mask
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],self.splits[i]:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                # self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                query_embed = torch.cat([query_embed, incre_query_embed], 0)
                output = torch.cat([output, incre_query_feat], 0)

            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                pad_outputs_mask = list(predictions_mask[-1].values())
                pad_outputs_mask = torch.cat(pad_outputs_mask,1)
                bs, l, _, _ = pad_outputs_mask.shape

                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6
                s = self.splits
                for k, v in predictions_class[-1].items():
                    cstep = int(k[4:])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, s[cstep]:s[cstep+1], :self.num_classes+1]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]

                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.mask_embed(decoder_output[:,s[i]:s[i+1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.incre_mask_embed_list[i - 1](decoder_output[:,s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'final':
    # ab3 attention mask
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],self.splits[i]:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                # self.incre_query_feat_list = nn.ModuleList()
                # for i in range(self.num_incre_steps):
                #     self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))



        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                # incre_query_feat = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_feat_list[:step]]
                # if len(incre_query_feat)>0:
                #     incre_query_feat = torch.cat(incre_query_feat,0)
                #     output = torch.cat([output, incre_query_feat], 0)
                incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps, self.inq, -1)[:step]
                incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)


                incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                output = torch.cat([output, incre_query_feat], 0)
                query_embed = torch.cat([query_embed, incre_query_embed], 0)


            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _,  h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, :, :self.num_classes+1]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'final2':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],self.splits[i]:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))



        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_embed_list[:step]]
                if len(incre_query_feat)>0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

                # incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps, self.inq, -1)[:step]
                # incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)


                # incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                # incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)
                #
                # output = torch.cat([output, incre_query_feat], 0)
                # query_embed = torch.cat([query_embed, incre_query_embed], 0)


            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _,  h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, :, :self.num_classes+1]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'final3':
    # group mask embed
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],self.splits[i]:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_embed_list[:step]]
                if len(incre_query_feat)>0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

                # incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps, self.inq, -1)[:step]
                # incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)


                # incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                # incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)
                #
                # output = torch.cat([output, incre_query_feat], 0)
                # query_embed = torch.cat([query_embed, incre_query_embed], 0)


            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _,  h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, :, :self.num_classes+1]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.incre_mask_embed_list[i-1](decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'pod':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)





            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

                # incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps, self.inq, -1)[:step]
                # incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                # incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                # incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)
                #
                # output = torch.cat([output, incre_query_feat], 0)
                # query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * 8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'freeze':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)




            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * self.num_heads).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6
                # pad_outputs_class[:,:,0] = 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                    mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                    mask_embed = self.incre_mask_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'freeze_old':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)




            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward_base(self, query_embed, output, mask_features, src, pos, size_list):

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[0])

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(
                                                                                                                               i + 1) % self.num_feature_levels])

                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            return predictions_class, predictions_mask, predictions_query

        def forward_novel(self, query_embed, output, mask_features, src, pos, size_list, step):
            bs = output.shape[1]

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            idx = step - 1

            # incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
            #                     self.incre_query_feat_list[:step]]
            # incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
            #                      self.incre_query_embed_list[:step]]
            #
            # incre_query_feat = torch.cat(incre_query_feat, 0)
            # output = torch.cat([output, incre_query_feat], 0)

            incre_query_feat = self.incre_query_feat_list[idx].weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)
            incre_query_embed = self.incre_query_embed_list[idx].weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

            output = torch.cat([output, incre_query_feat], 0)
            query_embed = torch.cat([query_embed, incre_query_embed], 0)


            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0],
                                                                                         step = step)

            outputs_class, outputs_mask = outputs_class[:, -self.inq:], outputs_mask[:, -self.inq:]
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output[:, -self.inq:])

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                               i + 1) % self.num_feature_levels],
                                                                                             step = step)
            outputs_class, outputs_mask = outputs_class[:, -self.inq:], outputs_mask[:, -self.inq:]

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output[:, -self.inq:])

            return predictions_class, predictions_mask, predictions_query

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            predictions_class = [dict()]*10
            predictions_mask = [dict()]*10
            predictions_query = [dict()]*10

            tmp1, tmp2, tmp3 = self.forward_base(query_embed, output, mask_features, src, pos, size_list)
            for i, (c, m, q) in enumerate(zip(tmp1, tmp2, tmp3)):
                predictions_class[i]['step0'] = c
                predictions_mask[i]['step0'] = m
                predictions_query[i]['step0'] = q

            if step!=0:
                for s in range(1, step+1):
                    tmp1, tmp2, tmp3 = self.forward_novel(query_embed, output, mask_features, src, pos, size_list, s)
                    for i, (c, m, q) in enumerate(zip(tmp1, tmp2, tmp3)):
                        predictions_class[i][f'step{s}'] = c
                        predictions_mask[i][f'step{s}'] = m
                        predictions_query[i][f'step{s}'] = q

            outputs_class = predictions_class[-1]
            outputs_mask = predictions_mask[-1]


            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6
                # pad_outputs_class[:,:,0] = 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size, step):
            idx = step-1
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.incre_class_embed_list[idx](decoder_output)
            mask_embed = self.incre_mask_embed_list[idx](decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'freeze_pod':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

                # incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps, self.inq, -1)[:step]
                # incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)

                # incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                # incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)
                #
                # output = torch.cat([output, incre_query_feat], 0)
                # query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * 8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'pod_mm':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)





            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            # incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * 8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            s = self.splits
            querys = dict()
            for i in range(step + 1):
                querys[f'step{i}'] = output[s[i]:s[i + 1]]

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(querys)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                s = self.splits
                querys = dict()
                for i in range(step + 1):
                    querys[f'step{i}'] = output[s[i]:s[i + 1]]


                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(querys)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'cisdq':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)





            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            # incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * 8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            s = self.splits
            querys = dict()
            for i in range(step + 1):
                querys[f'step{i}'] = output[s[i]:s[i + 1]]

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(querys)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                s = self.splits
                querys = dict()
                for i in range(step + 1):
                    querys[f'step{i}'] = output[s[i]:s[i + 1]]


                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(querys)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'cisdq_ade':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)





            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            # incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * 8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            s = self.splits
            querys = dict()
            for i in range(step + 1):
                querys[f'step{i}'] = output[s[i]:s[i + 1]]

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(querys)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                s = self.splits
                querys = dict()
                for i in range(step + 1):
                    querys[f'step{i}'] = output[s[i]:s[i + 1]]


                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(querys)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'cisdq_voc':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)





            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.splits = [0, self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1] + self.inq)

            self.incre_mask = torch.ones((self.splits[-1], self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i + 1], self.splits[i]:self.splits[i + 1]] = 0

            # learnable query features
            if self.inq * self.num_incre_steps != 0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))

        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                    self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1, 256).unsqueeze(1).repeat(1, bs, 1) for el in
                                     self.incre_query_embed_list[:step]]
                if len(incre_query_feat) > 0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

            nq, bs, _ = output.shape
            # incre_mask = repeat(self.incre_mask[:nq, :nq], 'a b -> n a b', n=bs * 8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])
            s = self.splits
            querys = dict()
            for i in range(step + 1):
                querys[f'step{i}'] = output[s[i]:s[i + 1]]

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(querys)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])
                s = self.splits
                querys = dict()
                for i in range(step + 1):
                    querys[f'step{i}'] = output[s[i]:s[i + 1]]


                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(querys)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _, h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep == 0:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], :self.num_classes + 1] = outputs_class[k][:, :,
                                                                                             :self.num_classes + 1]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep + 1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep + 1],
                        self.num_classes + 1 + self.inc * (cstep - 1):self.num_classes + 1 + self.inc * cstep] = \
                        outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep + 1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class': predictions_class,
                                    'predictions_mask': predictions_mask,
                                    'predictions_query': predictions_query,
                                    'mask_features': mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step + 1):
                if i == 0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i - 1](decoder_output[:, s[i]:s[i + 1]])
                mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear",
                                          align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks, 1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

elif version == 'plop':

    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps - 1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps * self.inc + num_classes

            self.incre_class_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step
            num_classes = self.num_classes + step*self.inc + 1

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            predictions_class = []
            predictions_mask = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=
                                                                                         size_list[0])


            predictions_class.append(outputs_class[...,:num_classes])
            predictions_mask.append(outputs_mask)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                             attn_mask_target_size=
                                                                                             size_list[(
                                                                                                                   i + 1) % self.num_feature_levels])


                predictions_class.append(outputs_class[...,:num_classes])
                predictions_mask.append(outputs_mask)

            assert len(predictions_class) == self.num_layers + 1

            out = {
                'kd_features': {'predictions_class': predictions_class,
                                'predictions_mask': predictions_mask,
                                'mask_features': mask_features},
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    [el for el in predictions_class] if self.mask_classification else None,
                    [el for el in predictions_mask]
                )
            }

            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output) # [B, Q, C]
            outputs_class_novel = [class_embed(decoder_output) for class_embed in self.incre_class_embed_list]# [B, Q, C_novel]
            outputs_class = torch.cat([outputs_class, *outputs_class_novel],-1)# [B, Q, C+C_novel]
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


else:
    # group mask embed
    @TRANSFORMER_DECODER_REGISTRY.register()
    class IncrementalMultiScaleMaskedTransformerDecoder(nn.Module):
        _version = 2

        def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            version = local_metadata.get("version", None)
            if version is None or version < 2:
                # Do not warn if train from scratch
                scratch = True
                logger = logging.getLogger(__name__)
                for k in list(state_dict.keys()):
                    newk = k
                    if "static_query" in k:
                        newk = k.replace("static_query", "query_feat")
                    if newk != k:
                        state_dict[newk] = state_dict[k]
                        del state_dict[k]
                        scratch = False

                if not scratch:
                    logger.warning(
                        f"Weight format of {self.__class__.__name__} have changed! "
                        "Please upgrade your models. Applying automatic conversion now ..."
                    )

        @configurable
        def __init__(
                self,
                in_channels,
                mask_classification=True,
                *,
                num_classes: int,
                hidden_dim: int,
                num_queries: int,
                num_incre_queries: int,
                num_incre_classes: int,
                num_steps: int,
                nheads: int,
                dim_feedforward: int,
                dec_layers: int,
                pre_norm: bool,
                mask_dim: int,
                enforce_input_project: bool,
        ):
            """
            NOTE: this interface is experimental.
            Args:
                in_channels: channels of the input features
                mask_classification: whether to add mask classifier or not
                num_classes: number of classes
                hidden_dim: Transformer feature dimension
                num_queries: number of queries
                nheads: number of heads
                dim_feedforward: feature dimension in feedforward network
                enc_layers: number of Transformer encoder layers
                dec_layers: number of Transformer decoder layers
                pre_norm: whether to use pre-LayerNorm or not
                mask_dim: mask feature dimension
                enforce_input_project: add input project 1x1 conv even if input
                    channels and hidden dim is identical
            """
            super().__init__()

            assert mask_classification, "Only support mask classification model"
            self.mask_classification = mask_classification

            # positional encoding
            N_steps = hidden_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

            # define Transformer decoder here
            self.num_heads = nheads
            self.num_layers = dec_layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(self.num_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.decoder_norm = nn.LayerNorm(hidden_dim)

            self.num_queries = num_queries
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # level embedding (we always use 3 scales)
            self.num_feature_levels = 3
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
            self.input_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                if in_channels != hidden_dim or enforce_input_project:
                    self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # output FFNs
            if self.mask_classification:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


            # Incremental
            self.num_steps = num_steps
            self.num_incre_steps = num_steps-1
            self.inc = num_incre_classes
            self.inq = num_incre_queries
            self.num_classes = num_classes
            self.num_classes_all = self.num_incre_steps*self.inc+num_classes

            self.splits = [0,self.num_queries]
            for i in range(self.num_incre_steps):
                self.splits.append(self.splits[-1]+self.inq)

            self.incre_mask = torch.ones((self.splits[-1],self.splits[-1]))
            for i in range(num_steps):
                self.incre_mask[self.splits[i]:self.splits[i+1],self.splits[i]:self.splits[i+1]] = 0

            # learnable query features
            if self.inq*self.num_incre_steps!=0:
                self.incre_query_feat_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_feat_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_feat = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)
                # learnable query p.e.
                self.incre_query_embed_list = nn.ModuleList()
                for i in range(self.num_incre_steps):
                    self.incre_query_embed_list.append(nn.Embedding(self.inq, hidden_dim))
                # self.incre_query_embed = nn.Embedding(self.inq*self.num_incre_steps, hidden_dim)

            self.incre_class_embed_list = nn.ModuleList()
            self.incre_mask_embed_list = nn.ModuleList()
            for i in range(self.num_incre_steps):
                self.incre_class_embed_list.append(nn.Linear(hidden_dim, self.inc + 1))
                self.incre_mask_embed_list.append(MLP(hidden_dim, hidden_dim, mask_dim, 3))


        @classmethod
        def from_config(cls, cfg, in_channels, mask_classification):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification

            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM

            # Incremental parameters
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["num_incre_queries"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_OBJECT_QUERIES
            ret["num_incre_classes"] = cfg.MODEL.MASK_FORMER.NUM_INCRE_CLASSES
            ret["num_steps"] = cfg.MODEL.MASK_FORMER.NUM_STEPS
            # Transformer parameters:
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

            # NOTE: because we add learnable query features which requires supervision,
            # we add minus 1 to decoder layers to be consistent with our loss
            # implementation: that is, number of auxiliary losses is always
            # equal to number of decoder layers. With learnable query features, the number of
            # auxiliary losses equals number of decoders plus 1.
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

            return ret

        def forward(self, x, mask_features, mask=None):

            # currunt training step
            step = self.step

            # x is a list of multi-scale feature
            assert len(x) == self.num_feature_levels
            src = []
            pos = []
            size_list = []

            # disable mask, it does not affect performance
            del mask

            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)

            _, bs, _ = src[0].shape

            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.inq * self.num_incre_steps != 0:
                incre_query_feat = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_feat_list[:step]]
                incre_query_embed = [el.weight.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1) for el in self.incre_query_embed_list[:step]]
                if len(incre_query_feat)>0:
                    incre_query_feat = torch.cat(incre_query_feat, 0)
                    output = torch.cat([output, incre_query_feat], 0)

                    incre_query_embed = torch.cat(incre_query_embed, 0)
                    query_embed = torch.cat([query_embed, incre_query_embed], 0)

                # incre_query_feat = self.incre_query_feat.weight.reshape(self.num_incre_steps, self.inq, -1)[:step]
                # incre_query_feat = incre_query_feat.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)


                # incre_query_embed = self.incre_query_embed.weight.reshape(self.num_incre_steps,self.inq,-1)[:step]
                # incre_query_embed = incre_query_embed.reshape(-1,256).unsqueeze(1).repeat(1, bs, 1)
                #
                # output = torch.cat([output, incre_query_feat], 0)
                # query_embed = torch.cat([query_embed, incre_query_embed], 0)


            nq, bs, _ = output.shape
            incre_mask = repeat(self.incre_mask[:nq,:nq],'a b -> n a b',n = bs*8).to(output).to(torch.bool)

            predictions_class = []
            predictions_mask = []
            predictions_query = []

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                         attn_mask_target_size=size_list[0])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(output)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=incre_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.incre_forward_prediction_heads(output, mask_features,
                                                                                       attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(output)

            assert len(predictions_class) == self.num_layers + 1

            if self.training:
                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': predictions_class[-1][f'step{step}'],
                    'pred_masks': predictions_mask[-1][f'step{step}'],
                    'aux_outputs': self._set_aux_loss(
                        [el[f'step{step}'] for el in predictions_class] if self.mask_classification else None,
                        [el[f'step{step}'] for el in predictions_mask]
                    )
                }
            else:
                l = sum([el.shape[1] for el in outputs_mask.values()])
                bs, _,  h, w = outputs_mask['step0'].shape
                pad_outputs_mask = -torch.ones([bs, l, h, w]).to(predictions_mask[-1]['step0']) * 1e6
                pad_outputs_class = -torch.ones([bs, l, self.num_classes_all + 1]).to(pad_outputs_mask) * 1e6

                s = self.splits
                for k, v in outputs_mask.items():
                    cstep = int(k[4:])
                    if cstep==0:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], :self.num_classes+1] = outputs_class[k][:, :, :self.num_classes+1]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]
                    else:
                        pad_outputs_class[:, s[cstep]:s[cstep+1], 0] = outputs_class[k][:, :, 0]
                        pad_outputs_class[:, s[cstep]:s[cstep+1], self.num_classes + 1 + self.inc * (cstep-1) :self.num_classes + 1 + self.inc * cstep] = outputs_class[k][:, :, 1:]
                        pad_outputs_mask[:, s[cstep]:s[cstep+1]] = outputs_mask[k]

                out = {
                    'kd_features': {'predictions_class':predictions_class,
                                    'predictions_mask':predictions_mask,
                                    'predictions_query':predictions_query,
                                    'mask_features':mask_features},
                    'pred_logits': pad_outputs_class,
                    'pred_masks': pad_outputs_mask,
                }


            return out

        def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

        def incre_forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)

            step = self.step
            outputs_classs = dict()
            outputs_masks = dict()
            attn_masks = []
            s = self.splits
            for i in range(step+1):
                if i==0:
                    outputs_classs[f'step{i}'] = self.class_embed(decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.mask_embed(decoder_output[:, s[i]:s[i + 1]])
                else:
                    outputs_classs[f'step{i}'] = self.incre_class_embed_list[i-1](decoder_output[:,s[i]:s[i+1]])
                    mask_embed = self.incre_mask_embed_list[i-1](decoder_output[:, s[i]:s[i + 1]])
                outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
                outputs_masks[f'step{i}'] = outputs_mask
                # NOTE: prediction is of higher-resolution
                # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                                 1) < 0.5).bool()
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)

            return outputs_classs, outputs_masks, torch.cat(attn_masks,1)

        @torch.jit.unused
        def _set_aux_loss(self, outputs_class, outputs_seg_masks):
            # this is a workaround to make torchscript happy, as torchscript
            # doesn't support dictionary with non-homogeneous values, such
            # as a dict having both a Tensor and a list.
            if self.mask_classification:
                return [
                    {"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]
            else:
                return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]