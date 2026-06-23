import os
import json

ann_file_path = '/repository/panshouan/dataset/HardEventDSEC-DET/annotations/test_coco.json'
output_file_path = '/repository/panshouan/dataset/HardEventDSEC-DET/annotations/test_event.json'

export_event=True
# open and read the original annotation file
with open(ann_file_path, 'r') as f:
    annotations = json.load(f)
    # 删除info和events字段
    if 'info' in annotations:
        del annotations['info']
    if export_event and 'events' in annotations:
        del annotations['images']
        # 修改events字段名称为images
        annotations['images'] = annotations.pop('events')
    else:
        del annotations['events']
# write the modified annotations to a new file
with open(output_file_path, 'w') as f:
    json.dump(annotations, f)
print(f"Modified annotations saved to {output_file_path}")