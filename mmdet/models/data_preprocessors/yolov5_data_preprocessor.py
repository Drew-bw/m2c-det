# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union
import torch
from mmengine.structures import BaseDataElement
from .data_preprocessor import DetDataPreprocessor
from mmdet.registry import MODELS

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


@MODELS.register_module()
class YOLOv5DetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)
        # import numpy as np
        # import cv2
        # img1 = inputs[0, :3].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
        # for bboxes in data_samples['bboxes_labels']:
        #     if bboxes[0] == 0:
        #         x1, y1, x2, y2 = float(bboxes[2]), float(bboxes[3]), float(bboxes[4]), float(bboxes[5])
        #     cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # cv2.imwrite('a.jpg', img1)
        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            if self.multispectral and self.training:
                inputs = inputs[:, [2, 1, 0, 5, 4, 3], ...]  # type: ignore
            else:
                inputs = inputs[:, [2, 1, 0], ...]

        if self._enable_normalize:
            if self.multispectral and self.training:
                inputs = (inputs - self.mean) / self.std
            else:
                inputs = (inputs - self.mean[:3]) / self.std[:3]

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']
        if 'texts' in data_samples:
            data_samples_output['texts'] = data_samples['texts']
        if 'keypoints' in data_samples:
            data_samples_output['keypoints'] = data_samples['keypoints']
            data_samples_output['keypoints_visible'] = data_samples[
                'keypoints_visible']

        return {'inputs': inputs, 'data_samples': data_samples_output}