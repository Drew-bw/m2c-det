import json

file = '/home/jxl/mmProjects/datasets/DSEC/test_event.json'
new_file = '/home/jxl/mmProjects/datasets/DSEC/test_event.json'
fp = json.load(open(file))
new_fp = {
    'categories': fp['categories'],
    'annotations': fp['annotations'],
    'images': [],
}

# for item in fp['annotations']:
#     if item['category_id'] == 8:
#         raise 'error'
# new_fp['categories'].pop(7)

for item in fp['images']:
    item['file_name'] = item['file_name'].replace('train/', '')
    new_fp['images'].append(item)
json_fp = open(new_file, 'w')
json_fp.write(json.dumps(new_fp))
json_fp.close()