# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union
from torch import Tensor
from mmdet.structures import SampleList
from ..language_models import CLIPTextModel
from .yolo_detector import YOLODetector
from mmdet.registry import MODELS


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):

    def __init__(self,
                 language_model,
                 *args,
                 **kwargs) -> None:
        self.language_model = language_model
        super().__init__(*args, **kwargs)
        self.language_model = CLIPTextModel(**self.language_model)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_feats = self.language_model.forward_yolov5(batch_data_samples.pop('texts'))
        x, text_feats = self.extract_feat(batch_inputs, text_feats)
        losses = self.bbox_head.loss(x, text_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        text_feats = self.language_model(batch_data_samples)
        x, text_feats = self.extract_feat(batch_inputs, text_feats)
        results_list = self.bbox_head.predict(
            x, text_feats, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def extract_feat(self, batch_inputs: Tensor, text_feats: Tensor):

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x, text_feats)
        return x, text_feats