import os
import json
from mask2former.data.datasets.register_ade20k_incremental import tasks_ade, ADE20K_SEM_SEG
from mask2former.data.datasets.register_voc_incremental import tasks_voc, VOC_SEM_SEG
from mask2former.data.datasets.register_ade20k_incremental_instance import tasks_ade_ins, ADE_CATEGORIES, ids
from mask2former.data.datasets.register_coco_incremental_instance import tasks_coco_ins, COCO_CATEGORIES
from collections import OrderedDict
import copy
import cv2
import tqdm
import numpy as np


def incremental_eval(results, cfg=None):
    if cfg is None:
        name = 'voc_incremental_19-1_1'
    else:
        name = cfg.DATASETS.TRAIN[0]
    if 'ade20k' in name:
        if 'incrementalins' in name:
            classes = ADE_CATEGORIES
            tasks = tasks_ade_ins
        else:
            classes = ADE20K_SEM_SEG
            tasks = tasks_ade
    elif 'voc' in name:
        classes = VOC_SEM_SEG
        tasks = tasks_voc
    elif 'coco' in name:
        classes = COCO_CATEGORIES
        tasks = tasks_coco_ins
    else:
        classes = ADE20K_SEM_SEG
        tasks = tasks_ade
    if isinstance(results, str):
        with open(results,'r') as fr:
            results = json.load(fr)
    else:
        cachename = os.path.join('cache', 'eval.json')
        with open(cachename,'w') as fw:
            json.dump(results,fw)

    results = dict(results)
    update_results = dict()

    dataset, _, task, step = name.split('_')
    for domain, res in results.items():
        if domain == 'sem_seg':
            incre_res = dict()
            steps = tasks[task]
            for step,training_classes in steps.items():
                training_classes_str = [classes[id] for id in training_classes]
                assert len(training_classes_str) == len(training_classes)
                IoU_res = [(k, v) for (k, v) in res.items() if
                           k.startswith('IoU') and k.split('-')[-1] in training_classes_str]
                for el in IoU_res:
                    incre_res[f'step{step}_{el[0]}'] = el[1]
                incre_res[f'step{step}_mIoU of {len(IoU_res)} classes'] = sum([el[1] for el in IoU_res]) / len(IoU_res)
            update_results['incremental_results'] = incre_res
        elif domain == 'segm':
            incre_res = dict()
            steps = tasks[task]
            for step, training_ids in steps.items():
                training_classes_str = [[cat for cat in classes if cat['id']==i][0]['name'] for i in training_ids]
                assert len(training_classes_str) == len(training_ids)
                IoU_res = [(k, v) for (k, v) in res.items() if
                           k.startswith('AP') and k.split('-')[-1] in training_classes_str]
                for el in IoU_res:
                    incre_res[f'step{step}_{el[0]}'] = el[1]
                incre_res[f'step{step}_mAP of {len(IoU_res)} classes'] = sum([el[1] for el in IoU_res]) / len(IoU_res)
            for key, value in incre_res.items():
                print(f'{key} : {value}')
            update_results['incremental_segm_results'] = incre_res
            # print(results['incremental_results'])
            # for key, value in results['incremental_results'].items():
            #     if '_IoU' in key:
            #         print(f'{key} : {value}')
            # for key, value in results['incremental_results'].items():
            #     if '_mIoU' in key:
            #         print(f'{key} : {value}')
        # elif domain == 'bbox':
        #     incre_res = dict()
        #     steps = tasks[task]
        #     for step, training_ids in steps.items():
        #         training_classes_str = [[cat for cat in ADE_CATEGORIES if cat['id']==i][0]['name'] for i in training_ids]
        #         assert len(training_classes_str) == len(training_ids)
        #         IoU_res = [(k, v) for (k, v) in res.items() if
        #                    k.startswith('AP') and k.split('-')[-1] in training_classes_str]
        #         for el in IoU_res:
        #             incre_res[f'step{step}_{el[0]}'] = el[1]
        #         incre_res[f'step{step}_mAP of {len(IoU_res)} classes'] = sum([el[1] for el in IoU_res]) / len(IoU_res)
        #     update_results['incremental_bbox_results'] = incre_res
    results.update(update_results)
    results = OrderedDict(results)
    return results


# fix the mismatch bug of mask2former checkpoint
def fix_ckpoint(in_ckpoint,out_ckpoint='cache/fixed.pth'):
    import torch
    file = torch.load(in_ckpoint)
    remove_list = []
    new_file = copy.deepcopy(file)
    for k,v in file['model'].items():
        if 'sem_seg_head.pixel_decoder' in k:
            nk = 'sem_seg_head'+k[26:]
            new_file['model'][nk] = v
            remove_list.append(k)
    for k in remove_list:
        del new_file['model'][k]
    torch.save(new_file,out_ckpoint)


def voc_label_gen(path = 'datasets/PascalVOC12/'):
    txt = 'datasets/PascalVOC12/splits/val.txt'
    with open(txt, 'r') as fr:
        training_lst = fr.readlines()
    val_lst = [os.path.basename(el.split(' ')[0])[:-4] for el in training_lst]

    segs = os.path.join(path,'SegmentationClassAug/')
    segs = [os.path.join(segs,el) for el in os.listdir(segs) if el.endswith('png')]
    save = os.path.join(path, 'SegmentationClassAugMod')
    if not os.path.exists(save):
        os.mkdir(save)
    classes = list(range(20))+[200,255]
    for seg in tqdm.tqdm(segs):
        if os.path.basename(seg)[:-4] in val_lst:
            tmp = cv2.imread(seg, cv2.IMREAD_GRAYSCALE)
            tmp = tmp - 1
            tmp[tmp == 254] = 255  # ignore
        else:
            tmp = cv2.imread(seg,cv2.IMREAD_GRAYSCALE)
            tmp = tmp-1
            tmp[tmp==255] = 200 # background
            tmp[tmp==254] = 255 # ignore
        clst = np.unique(tmp).tolist()
        try:
            for c in clst:
                assert c in classes
        except:
            print(clst)
        # assert tmp.max()<=19 and tmp.min()>=-1
        # tmp[tmp<0] = 255
        cv2.imwrite(os.path.join(save,os.path.basename(seg)),tmp)

if __name__ == '__main__':
    incremental_eval('cache/eval.json')
    #voc_label_gen()
