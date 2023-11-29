import os
import torch
import copy
import argparse


# name = [
# "sem_seg_head.pixel_decoder.adapter_1.norm.{bias, weight}",
# "sem_seg_head.pixel_decoder.adapter_1.weight",
# "sem_seg_head.pixel_decoder.input_proj.0.0.{bias, weight}
# "sem_seg_head.pixel_decoder.input_proj.0.1.{bias, weight}
# "sem_seg_head.pixel_decoder.input_proj.1.0.{bias, weight}
# "sem_seg_head.pixel_decoder.input_proj.1.1.{bias, weight}
# "sem_seg_head.pixel_decoder.input_proj.2.0.{bias, weight}
# "sem_seg_head.pixel_decoder.input_proj.2.1.{bias, weight}
# "sem_seg_head.pixel_decoder.layer_1.norm.{bias, weight}
# "sem_seg_head.pixel_decoder.layer_1.weight
# "sem_seg_head.pixel_decoder.mask_features.{bias, weight}
# "sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear1.{bias, weight}
# "sem_seg_head.pixel_decoder.transformer.encoder.layers.0.linear2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.0.norm2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.attention_weights.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.output_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.sampling_offsets.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.0.self_attn.value_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.linear2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.norm2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.attention_weights.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.output_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.sampling_offsets.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.1.self_attn.value_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.linear2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.norm2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.attention_weights.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.output_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.sampling_offsets.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.2.self_attn.value_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.linear2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.norm2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.attention_weights.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.output_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.sampling_offsets.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.3.self_attn.value_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.linear2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.norm2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.attention_weights.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.output_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.sampling_offsets.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.4.self_attn.value_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.linear2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm1.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.norm2.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.attention_weights.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.output_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.sampling_offsets.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.encoder.layers.5.self_attn.value_proj.{bias, weight}
# sem_seg_head.pixel_decoder.transformer.level_embed"]
# fix the mismatch bug of mask2former checkpoint
def fix_ckpoint(in_ckpoint,out_ckpoint):
    if not in_ckpoint.endswith('pth'):
        logpath = os.path.join(in_ckpoint,'log.txt')
        if os.path.exists(logpath):
            with open(logpath, 'r') as fr:
                file = fr.read()
        pths = [os.path.join(in_ckpoint,el) for el in os.listdir(in_ckpoint) if el.endswith('pth')]
        in_ckpoint = sorted(pths)[-1]
    print(f'load from {in_ckpoint}')
    file = torch.load(in_ckpoint)
    remove_list = []
    new_file = copy.deepcopy(file)
    for k,v in file['model'].items():
        print(k)
        # layer_1.norm
        name = ["transformer.encoder","mask_features","layer_1.norm","layer_1.weight","transformer.level_embed","input_proj","adapter_1"]
        flag = 0 
        for i in name:
            if i in k:
                nk = k.replace("sem_seg_head","sem_seg_head.pixel_decoder")
                print(nk)
                new_file['model'][nk] = v
                remove_list.append(k)
                break
#         elif 'teacher_model' in k:
#             remove_list.append(k)
    for k in remove_list:
        del new_file['model'][k]
    torch.save(new_file,out_ckpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i')
    parser.add_argument('--o', default='cache/tmp.pth')
    args = parser.parse_args()
    fix_ckpoint(args.i,args.o)
