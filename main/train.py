# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import os.path as osp
import sys

import cv2
from loguru import logger

import torch

import random
import os
import numpy as np

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.engine.monitor.monitor_impl.tensorboard_logger import TensorboardLogger
import torch.nn as nn

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='path to experiment configuration')
    parser.add_argument(
        '-r',
        '--resume',
        default="",
        help=r"completed epoch's number, latest or one model path")
    parser.add_argument(
        '-v',
        '--validation',
        default="15",
        help=r"Epoch's number to start to evaluate the model on the validation set")
    parser.add_argument(
        '-g',
        '--gpu_num',
        default="0,1,2",
        help=r"s")
    parser.add_argument(
        '-d',
        '--debugTM',
        default='v10',
        type=str,
        help=r"debugTM mode")

    return parser

def freeze_model(model):
    for name, value in model.senet_layer_Material.named_parameters():
        value.requires_grad = True
    for m in model.senet_layer_Material.modules():
        m.train()

    for name, value in model.senet_layer_hsi.named_parameters():
        value.requires_grad = True
    for m in model.senet_layer_hsi.modules():
        m.train()

    for name, value in model.transformer.named_parameters():
        value.requires_grad = True
    for m in model.transformer.modules():
        m.train()
    
    ## backbone
    for param in model.basemodel_m.parameters():
        param.requires_grad = False
    for m in model.basemodel_m.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for param in model.basemodel_q.parameters():
        param.requires_grad = False
    for m in model.basemodel_q.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    ## neck
    for param in model.neck_q.parameters():
        param.requires_grad = False
    for m in model.neck_q.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for param in model.neck_m.parameters():
        param.requires_grad = False
    for m in model.neck_m.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    ## head
    for param in model.head.parameters():
        param.requires_grad = False
    for m in model.head.modules():
       if isinstance(m, nn.BatchNorm2d):
           m.eval()
    return model


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



if __name__ == '__main__':
    set_seed(1000000007)
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    # root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg.train)
    task_cfg.freeze()
    # log config
    log_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(log_dir)
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=osp.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )
    # backup config
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info(
        "Merged with root_cfg imported from videoanalyst.config.config.cfg")
    cfg_bak_file = osp.join(log_dir, "%s_bak.yaml" % task_cfg.exp_name)
    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())
    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    # device config
    # if task_cfg.device == "cuda":
    #     world_size = task_cfg.num_processes
    #     assert torch.cuda.is_available(), "please check your devices"
    #     assert torch.cuda.device_count(
    #     ) >= world_size, "cuda device {} is less than {}".format(
    #         torch.cuda.device_count(), world_size)
    #     # devs = ["cuda:{}".format(i) for i in range(world_size)]
    #     devs = ["cuda:{}".format(1)]
    # else:
    #     devs = ["cpu"]


    if parsed_args.gpu_num.find(',') == -1:
        devs = ["cuda:{}".format(parsed_args.gpu_num)]
    else:
        devs = ["cuda:{}".format(int(i)) for i in parsed_args.gpu_num.split(',')]
    print ('devs = ', devs)
    model = model_builder.build(task, task_cfg.model, parsed_args.debugTM)

    if len(devs) == 1:
        model = model.cuda() 
    else:
        model.set_device(devs[0])
    model = freeze_model(model)
    # load data
    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader, devs)
    if len(devs) > 1:
        trainer.set_device(devs)
    # trainer.resume(parsed_args.resume)
    # trainer = trainer.cuda()
    
    # for k,v in model.named_parameters():
    #     print ('k = ', k, ' , v = ', v)
    # raise Exception

    # Validator initialization
    # root_cfg.test.track.freeze()
    # pipeline = pipeline_builder.build("track", root_cfg.test.track.pipeline, model)
    # testers = tester_builder("track", root_cfg.test.track.tester, "tester", pipeline)
    # epoch_validation = int(parsed_args.validation)
    # logger.info("Start to evaluate the model on the validation set after the epoch #{}".format(epoch_validation))

    logger.info("Training begins.")
    while not trainer.is_completed():
        trainer.train()
        trainer.save_snapshot()
        if False and trainer._state['epoch'] >= epoch_validation:
            logger.info('Validation begins.')
            model.eval()
            for tester in testers:
                res = tester.test()
                benchmark = '{}/{}/{}'.format(tester.__class__.__name__,
                                              tester._hyper_params['subsets'][0],
                                              'AO')
                logger.info('{}: {}'.format(benchmark, res['main_performance']))
                tb_log = {benchmark: res['main_performance']}
                for mo in trainer._monitors:
                    if isinstance(mo, TensorboardLogger):
                        mo.update(tb_log)
                torch.cuda.empty_cache()
            logger.info('Validation ends.')

    # export final model
    trainer.save_snapshot(model_param_only=True)
    logger.info("Training completed.")
