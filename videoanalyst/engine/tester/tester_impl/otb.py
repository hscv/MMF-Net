# -*- coding: utf-8 -*
import copy
import os.path as osp

from loguru import logger

import torch
import torch.multiprocessing as mp

from videoanalyst.evaluation.got_benchmark.experiments import ExperimentOTB

from ..tester_base import TRACK_TESTERS, TesterBase
from .utils.got_benchmark_helper import PipelineTracker


@TRACK_TESTERS.register
class OTBTester(TesterBase):
    r"""OTB tester
    
    Hyper-parameters
    ----------------
    device_num: int
        number of gpus. If set to non-positive number, then use cpu
    data_root: str
        path to got-10k root
    subsets: List[str]
        list of subsets name (val|test)
    """
    extra_hyper_params = dict(
        device_num=1,
        data_root="datasets/OTB/OTB2015",
        subsets=["2015"],  # (2013|2015)
    )

    def __init__(self, *args, **kwargs):
        super(OTBTester, self).__init__(*args, **kwargs)
        # self._experiment = None

    def update_params(self):
        # set device state
        num_gpu = self._hyper_params["device_num"]
        if num_gpu > 0:
            all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        else:
            all_devs = [torch.device("cpu")]
        self._state["all_devs"] = all_devs

    def test(self, fc_images, hsi_images, material_images, gtArr, ):
        print ('self._hyper_params = ', self._hyper_params)
        tracker_name = self._hyper_params["exp_name"]
        all_devs = self._state["all_devs"]
        nr_devs = len(all_devs)
        root_dir = self._hyper_params["data_root"]
        dataset_name = "GOT-Benchmark" 
        dev = all_devs[0]
        self._pipeline.set_device(dev)
        pipeline_tracker = PipelineTracker(tracker_name, self._pipeline)
        boxes, times = pipeline_tracker.track(fc_images, hsi_images, material_images,
                                 gtArr,
                                 visualize=False)
        return boxes


    def worker(self, dev_id, dev, subset, slicing_quantile):
        logger.debug("Worker starts: slice {} at {}".format(
            slicing_quantile, dev))
        tracker_name = self._hyper_params["exp_name"]

        pipeline = self._pipeline
        pipeline.set_device(dev)
        pipeline_tracker = PipelineTracker(tracker_name, pipeline)

        root_dir = self._hyper_params["data_root"]
        dataset_name = "GOT-Benchmark"  # the name of benchmark toolkit, shown under "repo/logs" directory
        save_root_dir = osp.join(self._hyper_params["exp_save"], dataset_name)
        result_dir = osp.join(save_root_dir, "result")
        report_dir = osp.join(save_root_dir, "report")

        experiment = ExperimentOTB(root_dir,
                                   version=subset,
                                   result_dir=result_dir,
                                   report_dir=report_dir)
        experiment.run(pipeline_tracker, slicing_quantile=slicing_quantile)
        logger.debug("Worker ends: slice {} at {}".format(
            slicing_quantile, dev))


OTBTester.default_hyper_params = copy.deepcopy(OTBTester.default_hyper_params)
OTBTester.default_hyper_params.update(OTBTester.extra_hyper_params)
