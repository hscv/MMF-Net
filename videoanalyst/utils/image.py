# -*- coding: utf-8 -*-
import glob
import os
import os.path as osp

import cv2
import numpy as np
from loguru import logger
from PIL import Image

_RETRY_NUM = 10

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
    img = np.asarray(img, dtype=np.float32)
    return img

def getDivisor(num):
    res = []
    for kk in range(1, num+1):
        if num % kk == 0: res.append(kk)
    if len(res) % 2 == 1: return res[len(res)//2], res[len(res)//2]
    else: return res[len(res)//2], res[len(res)//2-1]

def X2CubeNew(img, modeMatlib): ## img = (h*div2, w*div1), modeMatlib=R8
    div1, div2 = getDivisor(int(modeMatlib[1:])) ## div1 > div2
    h,w,c = img.shape[0] // div2, img.shape[1] // div1, (div1*div2)
    assert div1*div2 == c and div1*div2 == int(modeMatlib[1:])
    resImg = np.zeros((h, w, c))
    for i in range(div2):
        for j in range(div1):
            resImg[:,:,i*div1+j] = img[i*h:(i+1)*h,j*w:(j+1)*w]
    return resImg

def load_image(img_file: str) -> np.array:
    """Image loader used by data module (e.g. image sampler)
    
    Parameters
    ----------
    img_file: str
        path to image file
    Returns
    -------
    np.array
        loaded image
    
    Raises
    ------
    FileExistsError
        invalid image file
    RuntimeError
        unloadable image file
    """
    if not osp.isfile(img_file):
        logger.info("Image file %s does not exist." % img_file)

    if img_file.find('.png') != -1:
        if img_file.find('Material') != -1:
            model_material = img_file.split('/')[-2] ## Material-R6
            model_material = model_material.split('-')[-1] ## R6
            resImg = cv2.imread(img_file, 0)
            img = X2CubeNew(resImg, model_material)
            # print ('img.shape = ', img.shape)
            # for dd in range(img.shape[-1]):
            #     savename = 'mater_%d.jpg' % dd
            #     cv2.imwrite(savename, img[:, :, dd])
            # raise Exception
        else:
            # print ('img_file = ', img_file)
            img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = X2Cube(img)
    elif img_file.find('.jpg') != -1:
        img = cv2.imread(img_file)
    elif img_file.find('.npy') != -1:
        raise  Exception
        img = np.load(img_file)
    else:
        # read with OpenCV
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img is None:
            # retrying
            for ith in range(_RETRY_NUM):
                logger.info("cv2 retrying (counter: %d) to load image file: %s" %
                            (ith + 1, img_file))
                img = cv2.imread(img_file, cv2.IMREAD_COLOR)
                if img is not None:
                    break
        # read with PIL
        if img is None:
            logger.info("PIL used in loading image file: %s" % img_file)
            img = Image.open(img_file)
            img = np.array(img)
            img = img[:, :, [2, 1, 0]]  # RGB -> BGR
        if img is None:
            logger.info("Fail to load Image file %s" % img_file)

    return img


def save_image(image, name):
    save_dir = './logs/STMTrack_debug/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, name+'.jpg')
    cv2.imwrite(path, image)

class ImageFileVideoStream:
    r"""Adaptor class to be compatible with VideoStream object
        Accept seperate video frames
    """
    def __init__(self, video_dir, init_counter=0):
        self._state = dict()
        self._state["video_dir"] = video_dir
        self._state["frame_files"] = sorted(glob.glob(video_dir))
        self._state["video_length"] = len(self._state["frame_files"])
        self._state["counter"] = init_counter  # 0

    def isOpened(self, ):
        return (self._state["counter"] < self._state["video_length"])

    def read(self, ):
        frame_idx = self._state["counter"]
        frame_file = self._state["frame_files"][frame_idx]
        frame_img = load_image(frame_file)
        self._state["counter"] += 1
        return frame_idx, frame_img

    def release(self, ):
        self._state["counter"] = 0


class ImageFileVideoWriter:
    r"""Adaptor class to be compatible with VideoWriter object
        Accept seperate video frames
    """
    def __init__(self, video_dir):
        self._state = dict()
        self._state["video_dir"] = video_dir
        self._state["counter"] = 0
        logger.info("Frame results will be dumped at: {}".format(video_dir))

    def write(self, im):
        frame_idx = self._state["counter"]
        frame_file = osp.join(self._state["video_dir"],
                              "{:06d}.jpg".format(frame_idx))
        if not osp.exists(self._state["video_dir"]):
            os.makedirs(self._state["video_dir"])
        cv2.imwrite(frame_file, im)
        self._state["counter"] += 1

    def release(self, ):
        self._state["counter"] = 0
