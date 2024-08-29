# -*- coding:utf-8 -*-
# @FileName : ComTQA.py
# @Time : 2024/8/15 15:22
# @Author : fiv

import json
import os

import fitz  # PyMuPDF
import jsonlines
from PIL import Image
from tqdm import tqdm


def load_ComTQA(path):
    data_PubTab1M = []
    data_FinTabNet = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        dataset = item["dataset"]
        if dataset == 'PubTab1M':
            data_PubTab1M.append(item)
        elif dataset == 'FinTabNet':
            data_FinTabNet.append(item)
    return data, data_FinTabNet, data_PubTab1M


def load_FinTabNet(path):
    new_data = {}
    with open(path) as f:
        data = jsonlines.Reader(f)
        data = list(data)
    for item in data:
        new_data[item["table_id"]] = {
            "filename": item["filename"],
            "bbox": item["bbox"],
        }
    return new_data


def crop_FinTabNet(target, source, pdf_dir, output_dir):
    for tgt in tqdm(target):
        table_id = tgt["table_id"]
        fin = source[int(table_id)]
        filepath = os.path.join(pdf_dir, fin['filename'])

        doc = fitz.open(filepath)
        zoom_x, zoom_y = 10, 10
        mat = fitz.Matrix(zoom_x, zoom_y)

        page = doc[0]
        page_height = page.rect.height
        bbox = fin["bbox"]

        x0, y0, x1, y1 = bbox[0], page_height - bbox[3], bbox[2], page_height - bbox[1]  # 翻转
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.save(f"{output_dir}/{table_id}.png")


def load_PubTab1M(anno, source_dir, target_dir):
    for item in tqdm(anno):
        img_name = item["image_name"]
        cmd = f"cp {source_dir}/{img_name} {target_dir}"
        os.system(cmd)


def process_json(data):
    new_data = []
    for item in data:
        q = item["question"]
        q = q[0].lower() + q[1:]
        conv = [{
            "from": "human",
            "value": f"<image>\n This is a table image. Please first understand the information in the table."
        }, {
            "from": "gpt",
            "value": f"I have a general understanding of the information in this table."
        }, {
            "from": "human",
            "value": f"Based on the provided table, {q}"
        }, {
            "from": "gpt",
            "value": f"{item['answer']}"
        }]
        if 'table_id' in item.keys():
            image = f"{item['table_id']}.png"
        else:
            image = f"{item['image_name']}"
        new_data.append({
            "image": image,
            "conversations": conv
        })
    return new_data


if __name__ == "__main__":
    # image
    # data_FinTabNet = load_FinTabNet('../data/ComTQA/fintabnet/FinTabNet_1.0.0_cell_test.jsonl')
    # anno, anno_FinTabNet, anno_PubTab1M = load_ComTQA("../data/ComTQA/annotation.json")
    # pdf_path = '../data/ComTQA/fintabnet/pdf'
    # output_dir = "../data/ComTQA/images/"
    # crop_FinTabNet(anno_FinTabNet, data_FinTabNet, pdf_path, output_dir)
    # anno, anno_FinTabNet, anno_PubTab1M = load_ComTQA("../data/ComTQA/annotation.json")
    # load_PubTab1M(anno_PubTab1M, "../data/ComTQA/PubTables-1M-Structure_Images", output_dir)

    # json
    with open("../data/ComTQA/annotation.json") as f:
        data = json.load(f)
    data = process_json(data)
    with open("../data/ComTQA/ComTQA.json", 'w') as f:
        json.dump(data, f)
        # json.dump(data, f, indent=4, separators=(',', ':'))
