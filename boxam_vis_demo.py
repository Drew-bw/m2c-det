# Copyright (c) OpenMMLab. All rights reserved.
"""This script is in the experimental verification stage and cannot be
guaranteed to be completely correct. Currently Grad-based CAM and Grad-free CAM
are supported.

The target detection task is different from the classification task. It not
only includes the AM map of the category, but also includes information such as
bbox and mask, so this script is named bboxam.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import os.path
import warnings
import cv2
import numpy as np
from functools import partial
import mmcv
from mmengine import Config, DictAction
from mmengine.utils import ProgressBar
from mmengine.runner import Runner

try:
    from pytorch_grad_cam import AblationCAM, EigenCAM
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      'pytorch_grad_cam package.')

from boxam_utils import (BoxAMDetectorVisualizer,
                         BoxAMDetectorWrapper, DetAblationLayer,
                         DetBoxScoreTarget, GradCAM,
                         GradCAMPlusPlus, reshape_transform)

GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
}

GRAD_BASED_METHOD_MAP = {'gradcam': GradCAM, 'gradcam++': GradCAMPlusPlus}

ALL_SUPPORT_METHODS = list(GRAD_FREE_METHOD_MAP.keys() # 是否基于梯度：不基于梯度的是不算loss，基于梯度会算loss
                           | GRAD_BASED_METHOD_MAP.keys())

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Box AM')
    # img = '/home/jxl/mmProjects/datasets/DSEC/test/thun_01_b/events_frame/left/distorted/000000.png'
    config = 'configs/dsec/dfine_s_event.py'
    # ck = '/home/lenovo/mmProjects/model/spacenet_sar/cm_gfl_x1_bs8_lr005/freqkd/0-0-0&0.125-0.5-1&&0&20/epoch_12.pth'
    ck = '/home/jxl/dsec/m2c/epoch_12.pth'
    out = '../tmm/feat/base'
    parser.add_argument(
        '--img', default=None, help='Image path, include image file, dir and URL.')
    parser.add_argument('--config', default=config, help='Config file')
    parser.add_argument('--checkpoint', default=ck, help='Checkpoint file')
    parser.add_argument(
        '--method',
        default='gradcam++', # 
        choices=ALL_SUPPORT_METHODS,
        help='Type of method to use, supports '
        f'{", ".join(ALL_SUPPORT_METHODS)}.')
    parser.add_argument( # 对某一层可视化
        '--target-layers',
        default=['neck.convs[2]'],
        # default=['neck.lateral_convs[1]'],
        # default=['backbone.stages[3]'],
        nargs='+',
        type=str,
        help='The target layers to get Box AM, if not set, the tool will '
        'specify the neck.out_layers[2]')
    parser.add_argument(
        '--out-dir', default=out, help='Path to output file')
    parser.add_argument(
        '--show', action='store_true', help='Show the CAM results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument(
        '--topk',
        type=int,
        default=-1,
        help='Select topk predict resutls to show. -1 are mean all.')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=-1,
        help='max shapes. Its purpose is to save GPU memory. '
        'The activation map is scaled and then evaluated. '
        'If set to -1, it means no scaling.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--norm-in-bbox', action='store_true', help='Norm in bbox of am image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
        'The parameter controls how many channels should be ablated')

    args = parser.parse_args()
    return args


def init_detector_and_visualizer(args, cfg):
    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [args.max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    model_wrapper = BoxAMDetectorWrapper(
        cfg, args.checkpoint, args.score_thr, device=args.device)

    if args.preview_model:
        print(model_wrapper.detector)
        print('\n Please remove `--preview-model` to get the BoxAM.')
        return None, None

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(
                eval(f'model_wrapper.detector.{target_layer}'))
        except Exception as e:
            print(model_wrapper.detector)
            raise RuntimeError('layer does not exist', e)

    ablationcam_extra_params = {
        'batch_size': args.batch_size,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': args.ratio_channels_to_ablate
    }

    if args.method in GRAD_BASED_METHOD_MAP:
        method_class = GRAD_BASED_METHOD_MAP[args.method]
        is_need_grad = True
    else:
        method_class = GRAD_FREE_METHOD_MAP[args.method]
        is_need_grad = False

    boxam_detector_visualizer = BoxAMDetectorVisualizer(
        method_class,
        model_wrapper,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=ablationcam_extra_params)
    return model_wrapper, boxam_detector_visualizer


def main():
    args = parse_args()

    # hard code
    ignore_loss_params = None

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if not os.path.exists(args.out_dir) and not args.show:
        os.makedirs(args.out_dir)

    model_wrapper, boxam_detector_visualizer = init_detector_and_visualizer(
        args, cfg)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.out_dir
    cfg.test_dataloader.batch_size = 1
    runner = Runner.from_cfg(cfg)
    test_dataloader = runner.test_dataloader

    progress_bar = ProgressBar(len(test_dataloader))
    for idx, data_batch in enumerate(test_dataloader):
        image = np.array(data_batch['inputs'][0]).transpose(1, 2, 0)
        model_wrapper.set_input_data(image)

        # forward detection results
        result = model_wrapper()[0]

        pred_instances = result.pred_instances
        # Get candidate predict info with score threshold
        pred_instances = pred_instances[pred_instances.scores > args.score_thr]

        if len(pred_instances) == 0:
            warnings.warn('empty detection results! skip this')
            continue

        if args.topk > 0:
            pred_instances = pred_instances[:args.topk]

        targets = [
            DetBoxScoreTarget(
                pred_instances,
                device=args.device,
                ignore_loss_params=ignore_loss_params)
        ]

        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(True)
            model_wrapper.set_input_data(image, data_batch)
            boxam_detector_visualizer.switch_activations_and_grads(
                model_wrapper)

        # get box am image
        grayscale_boxam = boxam_detector_visualizer(image, targets=targets)

        # draw cam on image
        pred_instances = pred_instances.numpy()
        image_with_bounding_boxes = boxam_detector_visualizer.show_am(
            image.copy(),
            pred_instances,
            grayscale_boxam,
            with_norm_in_bboxes=args.norm_in_bbox)

        image_path = data_batch['data_samples'][0].img_path
        filename = os.path.basename(image_path)
        # out_file = None if args.show else os.path.join(args.out_dir, filename)
        out_file = image_path.replace('DSEC', 'DSEC_m2c')

        if out_file:
            mmcv.imwrite(image_with_bounding_boxes, out_file)
        else:
            cv2.namedWindow(filename, 0)
            cv2.imshow(filename, image_with_bounding_boxes)
            cv2.waitKey(0)

        # switch
        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(False)
            boxam_detector_visualizer.switch_activations_and_grads(
                model_wrapper)

        progress_bar.update()

    if not args.show:
        print(f'All done!'
              f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
