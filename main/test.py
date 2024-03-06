# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip
import os

import argparse
import os.path as osp

from loguru import logger

import torch
from glob import glob
import cv2
import numpy as np

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg

def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')
    parser.add_argument('-r',
                        '--root_dir',
                        default='',
                        type=str,
                        help='root dir')
    parser.add_argument('-s',
                        '--snapshot',
                        default='',
                        type=str,
                        help='snapshot path')
    parser.add_argument('-d',
                        '--debugTM',
                        default='v10',
                        type=str,
                        help='debugTM mode')

    

    return parser

def X2Cube(img):

    B = [4, 4]
    skip = [4, 4]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    img = out.reshape(M//4, N//4, 16)
    return img

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def build_stmtrack_tester(task_cfg, debugTM='v1'):
    model = model_builder.build("track", task_cfg.model, debugTM)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers

def get_gt_txt(gt_name):
    f = open(gt_name,'r')
    gt_arr = []
    gt_res = f.readlines()
    for gt in gt_res:
        kk = gt.split('\t')[:-1]
        x = list(map(int, kk))
        gt_arr.append(x)
    return gt_arr

def getDivisor(num):
    res = []
    for kk in range(1, num+1):
        if num % kk == 0: res.append(kk)
    if len(res) % 2 == 1: return res[len(res)//2], res[len(res)//2]
    else: return res[len(res)//2], res[len(res)//2-1]

def X2CubeNew(img, modeMatlib): 
    div1, div2 = getDivisor(int(modeMatlib[1:])) 
    h,w,c = img.shape[0] // div2, img.shape[1] // div1, (div1*div2)
    assert div1*div2 == c and div1*div2 == int(modeMatlib[1:])
    resImg = np.zeros((h, w, c))
    for i in range(div2):
        for j in range(div1):
            resImg[:,:,i*div1+j] = img[i*h:(i+1)*h,j*w:(j+1)*w]
    return resImg

def get_frames_material(video_name, img_size):
    modeMatlib = video_name.split('/')[-3]
    modeMatlib = modeMatlib.split('-')[-1]
    images = glob(os.path.join(video_name, '*.png*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    materialArr = []
    for img in images:
        resImg = cv2.imread(img, 0)
        frame = X2CubeNew(resImg, modeMatlib)
        materialArr.append(frame)
    return materialArr

def get_frames_hsi(video_name):
    images = glob(os.path.join(video_name, '*.png*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    hsiArr = []
    for img in images:
        frame1 = cv2.imread(img, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        frame = X2Cube(frame1)
        hsiArr.append(frame)
    return hsiArr

def get_frames_falseColor(video_name):
    video_name = video_name[:-3]+'img'
    images = glob(os.path.join(video_name, '*.jp*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    frameArr = []
    for img in images:
        frame = cv2.imread(img)
        frameArr.append(frame)
    return frameArr

def getMultiModalData(video_path_name):
    gt_path = video_path_name + '/groundtruth_rect.txt'
    gtArr = get_gt_txt(gt_path)
    hsi_res = get_frames_hsi(os.path.join(video_path_name, 'HSI'))
    fc_res = get_frames_falseColor(os.path.join(video_path_name.replace('test_HSI','testFalseColor'), 'img'))
    material_res = get_frames_material(os.path.join(video_path_name.replace('test_HSI','testMaterial-R6'), 'img'), fc_res[0].shape)
    return fc_res, hsi_res, material_res, gtArr

if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    ## python main/test.py --config experiments/stmtrack/test/otb/stmtrack-googlenet-otb.yaml
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg) ## 'track', ..
    task_cfg['model']['task_model']['STMTrack']['pretrain_model_path'] = parsed_args.snapshot
    task_cfg.freeze()
    
    torch.multiprocessing.set_start_method('spawn', force=True)
    if task == 'track':
        testers = build_stmtrack_tester(task_cfg, parsed_args.debugTM)
    else:
        raise ValueError('Undefined task: {}'.format(task))
    rootDir = parsed_args.root_dir
    video_dir_arr = os.listdir(rootDir)
    video_dir_arr.sort()
    video_dir_arr = [os.path.join(rootDir, vi_name) for vi_name in video_dir_arr]

    tester = testers[0]
    detArr = []
    gtArr = []
    for i in range(len(video_dir_arr)):
        video_name = video_dir_arr[i].split('/')[-1].split('.')[0]
        print ('video_name = ', video_name)
        saveModelPath = parsed_args.snapshot.split('-')[-1]
        saveModelPath = 'epoch_'+saveModelPath[:-4]
        save_det_path = os.path.join('demo/', saveModelPath, video_name+'_det.txt')
        if not os.path.exists(os.path.join('demo/', saveModelPath)):
            os.makedirs(os.path.join('demo/', saveModelPath))
        f = open(save_det_path,'w')

        fc_res, hsi_res, material_res, gtBBoxes = getMultiModalData(video_dir_arr[i])

        detRes = tester.test(fc_res, hsi_res, material_res, gtBBoxes)
        for data_tmp in detRes:
            for tmp in data_tmp:
                f.write(str(tmp)+'\t')
            f.write('\n')
        f.close()
