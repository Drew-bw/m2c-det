import torch
import re
import pickle

# data = pickle.load(open('../results.pkl', 'rb'))
# for d in data:
#     if '000000000139' in d['img_path']:
#         a = 1
# a = 1
def load_model_weights(weight_path):
    """加载模型权重，返回state_dict"""
    weights = torch.load(weight_path, map_location="cpu")
    if "state_dict" in weights:
        weights = weights["state_dict"]
    return weights

def convert_keys(mmdet_weights, mapping):
    """根据映射规则转换key名称"""
    detr_weights = {}
    for old_key, value in mmdet_weights.items():
        if 'backbone' in old_key:
            new_key = old_key
        elif ('decoder.layers' in old_key) or ('decoder.lqe_layers' in old_key):
            new_key = 'decoder.' + old_key
            if 'self_attn' in new_key:
                new_key = new_key.replace('self_attn.attn', 'self_attn')
            if 'norms.0' in new_key:
                new_key = new_key.replace('norms.0', 'norm1')
            if 'norms.1' in new_key:
                new_key = new_key.replace('norms.1', 'gateway.norm')
            if 'norms.2' in new_key:
                new_key = new_key.replace('norms.2', 'norm3')
            if 'ffn' in new_key:
                new_key = new_key.replace('ffn.layers.0.0', 'linear1').replace(
                    'ffn.layers.1', 'linear2')
        elif 'dn_query_generator' in old_key:
            new_key = 'decoder.denoising_class_embed.weight'
        elif 'decoder.ref_point_head' in old_key:
            new_key = old_key.replace('ref_point_head', 'query_pos_head')
        elif 'memory_trans_fc' in old_key:
            new_key = old_key.replace('memory_trans_fc', 'decoder.enc_output.proj')
        elif 'memory_trans_norm' in old_key:
            new_key = old_key.replace('memory_trans_norm', 'decoder.enc_output.norm')

        # =====================different scale=====================
        elif 'bbox_head.reg_branches.0' in old_key:
            new_key = old_key.replace(
                'bbox_head.reg_branches.0', 'decoder.dec_bbox_head.0.layers').replace(
                '2.weight', '1.weight').replace('4.weight', '2.weight').replace(
                '2.bias', '1.bias').replace('4.bias', '2.bias')
        elif 'bbox_head.reg_branches.1' in old_key:
            new_key = old_key.replace(
                'bbox_head.reg_branches.1', 'decoder.dec_bbox_head.1.layers').replace(
                '2.weight', '1.weight').replace('4.weight', '2.weight').replace(
                '2.bias', '1.bias').replace('4.bias', '2.bias')
        elif 'bbox_head.reg_branches.2' in old_key:
            new_key = old_key.replace(
                'bbox_head.reg_branches.2', 'decoder.dec_bbox_head.2.layers').replace(
                '2.weight', '1.weight').replace('4.weight', '2.weight').replace(
                '2.bias', '1.bias').replace('4.bias', '2.bias')
        # extra
        elif 'bbox_head.cls_branches.0' in old_key:
            new_key = old_key.replace(
                'bbox_head.cls_branches.0', 'decoder.dec_score_head.0')
        elif 'bbox_head.cls_branches.1' in old_key:
            new_key = old_key.replace(
                'bbox_head.cls_branches.1', 'decoder.dec_score_head.1')
        elif 'bbox_head.cls_branches.2' in old_key:
            new_key = old_key.replace(
                'bbox_head.cls_branches.2', 'decoder.dec_score_head.2')
        elif 'bbox_head.reg_branches.4' in old_key:
            new_key = old_key.replace(
                'bbox_head.reg_branches.4', 'decoder.pre_bbox_head.layers').replace(
                '2.weight', '1.weight').replace('4.weight', '2.weight').replace(
                '2.bias', '1.bias').replace('4.bias', '2.bias')
        # enc
        elif 'bbox_head.cls_branches.3.weight' in old_key:
            new_key = 'decoder.enc_score_head.weight'
        elif 'bbox_head.cls_branches.3.bias' in old_key:
            new_key = 'decoder.enc_score_head.bias'
        elif 'bbox_head.reg_branches.3' in old_key:
            new_key = old_key.replace(
                'bbox_head.reg_branches.3', 'decoder.enc_bbox_head.layers').replace(
                '2.weight', '1.weight').replace('4.weight', '2.weight').replace(
                '2.bias', '1.bias').replace('4.bias', '2.bias')
        # =====================different scale=====================

        # encoder
        elif 'neck.convs' in old_key:
            new_key = old_key.replace('neck.convs', 'encoder.input_proj').replace('bn', 'norm')

        elif 'encoder.transformer_blocks' in old_key:
            new_key = old_key.replace('encoder.transformer_blocks', 'encoder.encoder')
            if 'self_attn' in new_key:
                new_key = new_key.replace('self_attn.attn', 'self_attn')
            if 'norms.0' in new_key:
                new_key = new_key.replace('norms.0', 'norm1')
            if 'norms.1' in new_key:
                new_key = new_key.replace('norms.1', 'norm2')
            if 'ffn' in new_key:
                new_key = new_key.replace('ffn.layers.0.0', 'linear1').replace(
                    'ffn.layers.1', 'linear2')
        elif 'encoder.fpn.reduce_layers' in old_key:
            new_key = old_key.replace('encoder.fpn.reduce_layers', 'encoder.lateral_convs').replace('bn', 'norm')
        elif 'encoder.fpn.top_down_blocks' in old_key:
            new_key = old_key.replace('encoder.fpn.top_down_blocks', 'encoder.fpn_blocks').replace(
                'bn', 'norm').replace('main_conv', 'conv1').replace('short_conv', 'conv2').replace(
                '.blocks.', '.bottlenecks.').replace('rbr_dense', 'conv1').replace('rbr_1x1', 'conv2')
        elif 'encoder.fpn.bottom_up_blocks' in old_key:
            new_key = old_key.replace('encoder.fpn.bottom_up_blocks', 'encoder.pan_blocks').replace(
                'bn', 'norm').replace('main_conv', 'conv1').replace('short_conv', 'conv2').replace(
                '.blocks.', '.bottlenecks.').replace('rbr_dense', 'conv1').replace('rbr_1x1', 'conv2')
        elif 'encoder.fpn.downsamples' in old_key:
            new_key = old_key.replace('encoder.fpn.downsamples', 'encoder.downsample_convs').replace('bn', 'norm').replace(
                '.0.0.', '.0.0.cv1.').replace('.0.1.', '.0.0.cv2.').replace(
                '.1.0.', '.1.0.cv1.').replace('.1.1.', '.1.0.cv2.')
        else:
            new_key = old_key
        detr_weights[new_key] = value
        print(f'convert {old_key} to {new_key}')
    return detr_weights

def save_converted_weights(detr_weights, save_path):
    """保存转换后的权重"""
    torch.save({"model": detr_weights}, save_path)
    print(f"转换完成，权重已保存至：{save_path}")


def mmdet2detr_converter(mmdet_weight_path, detr_save_path):
    """主转换函数"""
    # 加载原始权重
    mmdet_weights = load_model_weights(mmdet_weight_path)
    print(f"成功加载mmdet权重，共{len(mmdet_weights)}个参数")

    # 转换key
    detr_weights = convert_keys(mmdet_weights)
    print(f"转换后得到{len(detr_weights)}个参数")

    # 保存权重
    save_converted_weights(detr_weights, detr_save_path)


if __name__ == "__main__":
    # 配置路径
    MMDET_WEIGHT_PATH = "../epoch_10.pth"  # 输入mmdet模型权重路径
    DETR_SAVE_PATH = "../ov_dfine_s_convert.pth"  # 输出detr模型权重路径

    # 执行转换
    mmdet2detr_converter(MMDET_WEIGHT_PATH, DETR_SAVE_PATH)