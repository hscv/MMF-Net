# -*- coding: utf-8 -*
import time
from typing import List

# from PIL import Image
import cv2
import numpy as np

from videoanalyst.evaluation.got_benchmark.utils.viz import show_frame
from videoanalyst.pipeline.pipeline_base import PipelineBase


class PipelineTracker(object):
    def __init__(self,
                 name: str,
                 pipeline: PipelineBase,
                 is_deterministic: bool = True):
        """Helper tracker for comptability with 
        
        Parameters
        ----------
        name : str
            [description]
        pipeline : PipelineBase
            [description]
        is_deterministic : bool, optional
            [description], by default False
        """
        self.name = name
        self.is_deterministic = is_deterministic
        self.pipeline = pipeline

    def init(self, fc_image, hsi_image, material_image, box):
        """Initialize pipeline tracker
        
        Parameters
        ----------
        image : np.array
            image of the first frame
        box : np.array or List
            tracking bbox on the first frame
            formate: (x, y, w, h)
        """
        self.pipeline.init(fc_image, hsi_image, material_image, box)

    def update(self, fc_image, hsi_image, material_image):
        """Perform tracking
        
        Parameters
        ----------
        image : np.array
            image of the current frame
        
        Returns
        -------
        np.array
            tracking bbox
            formate: (x, y, w, h)
        """
        return self.pipeline.update(fc_image, hsi_image, material_image)

    def track(self, fc_imagesArr, hsi_imagesArr, material_imagesArr, gtArr, visualize: bool = False):
        """Perform tracking on a given video sequence
        
        Parameters
        ----------
        fc_imagesArr : List
            list of falsecolor images (no path) of the sequence
        box : np.array or List
            box of the first frame
        visualize : bool, optional
            Visualize or not on each frame, by default False
        
        Returns
        -------
        [type]
            [description]
        """
        box = gtArr[0]
        frame_num = len(fc_imagesArr)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f in range(len(fc_imagesArr)):
            fc_image = fc_imagesArr[f]
            hsi_image = hsi_imagesArr[f]
            material_image = material_imagesArr[f]

            start_time = time.time()
            if f == 0:
                self.init(fc_image, hsi_image, material_image, box)
                print ('--',f+1,'-- gt = ', gtArr[f], 'det = ', gtArr[f])
            else:
                boxes[f, :] = self.update(fc_image, hsi_image, material_image)
                print ('--',f+1,'-- gt = ', gtArr[f], 'det = ', boxes[f, :])
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times
