# -*- coding: utf-8 -*-
"""
 @File    : grad_cam.py
 @Time    : 2020/3/14 下午4:06
 @Author  : yizuotian
 @Description    :
"""
import cv2
import numpy as np
import torch

class GradCAM(object):

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                # print(name)
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        x = self.net.extract_feat(inputs['img'][0])
        # score1, _= self.net.rpn_head.forward_single(x[0])
        # score2, _= self.net.rpn_head.forward_single(x[1])
        # score3, _= self.net.rpn_head.forward_single(x[2])
        # score4, _= self.net.rpn_head.forward_single(x[3])
        score5, _= self.net.rpn_head.forward_single(x[4])
        score = torch.mean(score5)# + torch.mean(score2) + torch.mean(score3) + torch.mean(score4) + torch.mean(score5)
        score.backward()

        gradient = self.gradient[0].detach().cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].detach().cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        del self.gradient
        del self.feature
        return cam


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=0):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        :param index: 第几个边框
        :return:
        """
        self.net.zero_grad()
        x = self.net.extract_feat(inputs['img'][0])
        # score1, _= self.net.rpn_head.forward_single(x[0])
        # score2, _= self.net.rpn_head.forward_single(x[1])
        # score3, _= self.net.rpn_head.forward_single(x[2])
        # score4, _= self.net.rpn_head.forward_single(x[3])
        score5, _= self.net.rpn_head.forward_single(x[4])
        score = torch.mean(score5)# + torch.mean(score2) + torch.mean(score3) + torch.mean(score4) + torch.mean(score5)
        score.backward()

        gradient = self.gradient[0].detach().cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].detach().cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        del self.gradient
        del self.feature
        return cam
