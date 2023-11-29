import os
import torch
import copy
import argparse

# fix the mismatch bug of mask2former checkpoint
def fix_ckpoint(in_ckpoint,out_ckpoint):
    
    if not in_ckpoint.endswith('pth'):
        print("no pth")
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
    print(file.keys())
    for k,v in file['model'].items():
        print(k)
        if 'pixel_decoder' in k:
            
            nk = k.replace("pixel_decoder.","")
            new_file['model'][nk] = v
            remove_list.append(k)
#         elif 'criterion.empty_weight' in k:
#             nk = k.replace("criterion.empty_weight","criterion.base_criterion.empty_weight")
#             new_file['model'][nk] = v
            
#             nk = k.replace("criterion.empty_weight","criterion.novel_criterion.empty_weight")
#             new_file['model'][nk] = v
#             remove_list.append(k)

        elif 'teacher_model' in k:
            remove_list.append(k)
    for k in remove_list:
        del new_file['model'][k]
    torch.save(new_file,out_ckpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i')
    parser.add_argument('--o', default='cache/tmp.pth')
    args = parser.parse_args()
    fix_ckpoint(args.i,args.o)
