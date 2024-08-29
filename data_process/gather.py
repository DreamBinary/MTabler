# -*- coding:utf-8 -*-
# @FileName : gather.py
# @Time : 2024/8/17 10:40
# @Author : fiv
import json
import os.path

from tqdm import tqdm

if __name__ == "__main__":
    out_dir = "../data/gather_data/"
    tgt_image_dir = out_dir + "images"
    tgt_json_path = out_dir + "data.json"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(tgt_image_dir):
        os.mkdir(tgt_image_dir)

    image_dir_list = [
        "../data/ComTQA/images/",
        "../data/TableVQA-Bench/images/",
        "../data/MMTab/images/"
    ]
    for dir in tqdm(image_dir_list):
        cmd = f"cp {dir}/* {tgt_image_dir}"
        os.system(cmd)

    json_path_list = [
        "../data/ComTQA/ComTQA.json",
        "../data/TableVQA-Bench/TableVQA.json",
        "../data/MMTab/MMTab.json"
    ]
    json_data = []
    for path in json_path_list:
        with open(path) as f:
            data = json.load(f)
            json_data.extend(data)

    with open(tgt_json_path, 'w') as f:
        json.dump(json_data, f)

    print(len(json_data))
