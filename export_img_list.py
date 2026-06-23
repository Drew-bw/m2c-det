import argparse
import os
import os.path as osp
from copy import deepcopy
from mmengine import Config
from mmengine.runner import Runner
from mmdet.engine.hooks.utils import trigger_visualization_hook

def parse_args():
    parser = argparse.ArgumentParser(description='Export image paths in visualized order')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--out', default='vis_img_list.txt', help='Output log file')
    parser.add_argument('--work-dir', help='Optional work_dir')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--show-dir', default=None, help='Optional show_dir for visualization hook')
    parser.add_argument('--show', action='store_true', help='show prediction results')  # ✅ 新增
    parser.add_argument('--wait-time', type=float, default=2, help='the interval of show (s)')  # ✅ 新增
    return parser.parse_args()



def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # --- work_dir 处理，保证 Runner.from_cfg 不报错 ---
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Build runner
    runner = Runner.from_cfg(cfg)

    # 如果希望可视化 hook 激活，需要指定 show_dir
    if args.show_dir:
        cfg.show_dir = args.show_dir
        cfg = trigger_visualization_hook(cfg, args)

    # 获取 test dataloader
    dataloader = runner.test_dataloader
    dataset = dataloader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset  # 处理 MultiDataset / Nested Dataset

    print(f"Total dataset samples: {len(dataset)}")

    # 遍历 dataloader 顺序导出 img_path
    img_list = []
    for batch in dataloader:
        # batch 是 dict / list，根据你的 dataloader pipeline
        if isinstance(batch, dict):
            data_samples = batch.get('data_samples', [])
        elif isinstance(batch, list):
            data_samples = batch
        else:
            data_samples = []

        for data in data_samples:
            # data 里通常有 img_id / img_path
            img_id = getattr(data, 'img_id', None)
            img_path = getattr(data, 'img_path', None)
            if img_path is None and img_id is not None:
                # fallback to dataset info
                info = dataset.get_data_info(img_id)
                img_path = info.get('img_path', info.get('filename', None))
            img_list.append((img_id, img_path))

    # 保存到文件
    with open(args.out, 'w') as f:
        for idx, (img_id, img_path) in enumerate(img_list):
            f.write(f"{idx}\t{img_id}\t{img_path}\n")

    print(f"Done. Exported {len(img_list)} images to {args.out}")

if __name__ == '__main__':
    main()
