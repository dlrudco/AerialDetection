# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch

from grad_cam import GradCAM, GradCamPlusPlus
from skimage import io
from torch import nn

from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import numpy as np
from tqdm import tqdm
import DOTA_devkit.polyiou as polyiou
import math
import pdb

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
        


# constants
WINDOW_NAME = "COCO detections"

def get_last_conv_name(net):

    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """

        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        output = self.net.inference([inputs])
        score = output[0]['instances'].scores[index]
        score.backward()

        return inputs['image'].grad  # [3,H,W]


def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    mask = cv2.resize(mask, (image.shape[1],image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)/255.
    return norm_image(cam), norm_image(heatmap), np.uint8(255 * mask)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network='frcnn', output_dir='./results'):
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}.png'.format(input_image_name, key)), image)


def get_parser():
    parser = argparse.ArgumentParser(description="MMDetection demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="../configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="../work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


def _prepare_data(img, img_transform, cfg, device):
    from mmdet.datasets import to_tensor
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])

def ispure(x):
    return not('instance' in x)

def main(args):
    #####
    #TODO:build model & load weight
    from mmdet.datasets.transforms import ImageTransform
    from tqdm import tqdm
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    cfg = Config.fromfile(config_file)
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint_file, device=device)
    print(model)
    
    ######
    # Grad-CAM
    # layer_name = get_last_conv_name(model)
    layer_name = 'backbone.layer4.2.conv3'

    folder = '/EHDD1/ADD/data/iSAID_Devkit/preprocess/dataset/iSAID_patches/val/images/'
    dst_folder = '/EHDD1/ADD/data/iSAID_Devkit/preprocess/dataset/iSAID_patches/val/cam'
    os.makedirs(dst_folder, exist_ok=True)
    os.makedirs(dst_folder+'++', exist_ok=True)
    imlist_total = os.listdir(folder)
    imlist = list(filter(ispure, imlist_total))
    #####
    for image in tqdm(imlist[1::2]):
        #TODO : prepare input
        grad_cam = GradCAM(model, layer_name)
        grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
        img = mmcv.imread(os.path.join(folder, image))
        if img.shape[0] != img.shape[1]:
            print(image)
            continue
        img_transform = ImageTransform(size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
        data = _prepare_data(img, img_transform, model.cfg, device)
        #######
        image_dict = {}
        mask = grad_cam(data)  # cam mask
        grad_cam.remove_handlers()
        image_dict['overlay'], image_dict['heatmap'], image_dict['mask'] = gen_cam(img, mask)
        save_image(image_dict, image.split('.')[0], output_dir=dst_folder)
        # # Grad-CAM++
        # grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
        image_dict = {}
        mask_plus_plus = grad_cam_plus_plus(data)  # cam mask
        image_dict['overlay'], image_dict['heatmap'], image_dict['mask'] = gen_cam(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()
        save_image(image_dict, image.split('.')[0], output_dir=dst_folder+'++')
        torch.cuda.empty_cache()

if __name__ == "__main__":
    """
    Usage:export KMP_DUPLICATE_LIB_OK=TRUE
    python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
      --input ./examples/pic1.jpg \
      --opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
    """
    mp.set_start_method("spawn", force=True)
    arguments = get_parser().parse_args()
    main(arguments)
