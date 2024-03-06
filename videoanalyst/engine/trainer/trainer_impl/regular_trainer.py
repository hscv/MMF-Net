# -*- coding: utf-8 -*
import copy
from collections import OrderedDict

from loguru import logger
from tqdm import tqdm

import torch
from torch import nn

from videoanalyst.utils import (Timer, ensure_dir, move_data_to_device,
                                unwrap_model)

from ..trainer_base import TRACK_TRAINERS, TrainerBase


@TRACK_TRAINERS.register
class RegularTrainer(TrainerBase):
    r"""
    Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    devices: List[str]
        list of string
    """
    extra_hyper_params = dict(
        minibatch=1,
        nr_image_per_epoch=1,
        max_epoch=1,
        snapshot="",
    )

    def __init__(self, optimizer, dataloader, devs, monitors=[]):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        optimizer: ModuleBase
            including optimizer, model and loss
        dataloder: DataLoader
            PyTorch dataloader object. 
            Usage: batch_data = next(dataloader)
        """
        super(RegularTrainer, self).__init__(optimizer, dataloader, monitors)
        # update state
        self._state["epoch"] = -1  # uninitialized
        self._state["initialized"] = False
       
        if len(devs) > 1:
            self.single_gpu_flag = False
            cuda_name = devs[0] ## cuda:0
            # print ('cuda_name = ', cuda_name)
            self._state["devices"] = torch.device(cuda_name)
        else:
            self.single_gpu_flag = True
            cuda_name = devs[0]
            # print ('cuda_name = ', cuda_name)
            self._state["devices"] = torch.device(cuda_name)
        # print ('single_gpu_flag = ', self.single_gpu_flag)
        # print ('cuda_name = ', cuda_name)
    def init_train(self, ):
        torch.cuda.empty_cache()
        # move model & loss to target devices
        # devs = self._state["devices"]
        # self._model.train()
        # load from self._state["snapshot_file"]
        # print ('snapshot = ', self._state["snapshot_file"])
        self.load_snapshot()
        # raise Exception
        # parallelism with Data Parallel (DP)
        # if len(self._state["devices"]) > 1:
            # self._model = nn.DataParallel(self._model, device_ids=devs)
            # logger.info("Use nn.DataParallel for data parallelism")
        super(RegularTrainer, self).init_train()
        logger.info("{} initialized".format(type(self).__name__))

    def train(self):
        # print ('do --- regular_trainer- t')
        
        if not self._state["initialized"]:
            self.init_train()
        self._state["initialized"] = True

        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        num_iterations = self._hyper_params["num_iterations"]
        

        # udpate engine_state
        self._state["max_epoch"] = self._hyper_params["max_epoch"]
        self._state["max_iteration"] = num_iterations

        self._optimizer.modify_grad(epoch)
        # self._optimizer.zero_grad()
        # raise Exception
        pbar = tqdm(range(num_iterations))
        self._state["pbar"] = pbar
        self._state["print_str"] = ""
        # cnt = 0
        time_dict = OrderedDict()
        for iteration, _ in enumerate(pbar):
            # cnt += 1
            # print ('===================cnt=============', cnt)
            # if cnt > 10: return
            self._state["iteration"] = iteration
            with Timer(name="data", output_dict=time_dict):
                training_data = next(self._dataloader)
            # raise Exception
            if self.single_gpu_flag:
                training_data = move_data_to_device(training_data, None, single_gpu=self.single_gpu_flag)
            else:
                training_data = move_data_to_device(training_data, self._state["devices"][0], single_gpu=self.single_gpu_flag)
            

            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()
            # raise Exception
            ## loss_name =  cls
            #  loss_name =  ctr
            #  loss_name =  reg
            
            # forward propagation
            with Timer(name="fwd", output_dict=time_dict):
                # print ('training_data = ', training_data)
                ## ['im_m', 'im_q', 'bbox_m', 'bbox_q', 'cls_gt', 'ctr_gt', 'box_gt', 'fg_bg_label_map', 'is_negative_pair']
                # print ('training_data.keys() = ', training_data.keys())
                predict_data = self._model(training_data)
                training_losses, extras = OrderedDict(), OrderedDict()
                for loss_name, loss in self._losses.items():
                    training_losses[loss_name], extras[loss_name] = loss(
                        predict_data, training_data)
                total_loss = sum(training_losses.values())

            # backward propagation
            with Timer(name="bwd", output_dict=time_dict):
                if self._optimizer.grad_scaler is not None:
                    self._optimizer.grad_scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
            self._optimizer.modify_grad(epoch, iteration)
            with Timer(name="optim", output_dict=time_dict):
                self._optimizer.step()

            trainer_data = dict(
                schedule_info=schedule_info,
                training_losses=training_losses,
                extras=extras,
                time_dict=time_dict,
            )

            for monitor in self._monitors:
                monitor.update(trainer_data)
            del training_data
            print_str = self._state["print_str"]
            pbar.set_description(print_str)


RegularTrainer.default_hyper_params = copy.deepcopy(
    RegularTrainer.default_hyper_params)
RegularTrainer.default_hyper_params.update(RegularTrainer.extra_hyper_params)
