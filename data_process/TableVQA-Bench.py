# -*- coding:utf-8 -*-
# @FileName : TableVQA-Bench.py
# @Time : 2024/8/16 18:31
# @Author : fiv

import io
import json
import os
from collections import defaultdict

import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm


def html2shape(html):
    # 解析HTML
    soup = BeautifulSoup(html, 'html.parser')

    # 找到所有的表格行
    rows = soup.find_all('tr')

    # 用于跟踪每一行需要填充的列数（由上方跨行单元格占据）
    col_fill = []

    # 计算行数和最大列数
    row_count = 0
    max_col_count = 0

    for row in rows:
        while len(col_fill) <= row_count:
            col_fill.append(0)  # 初始化col_fill列表以跟踪所有行的填充

        col_count = 0
        cells = row.find_all(['th', 'td'])

        for cell in cells:
            # 考虑上方跨行单元格占据的列
            col_count += col_fill[row_count]
            col_fill[row_count] = 0  # 当前行被填充完成，重置填充状态

            # 计算每个单元格的列跨度
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))

            # 累加当前行的列数
            col_count += colspan

            # 如果有跨行的情况，记录下跨行需要填充的列数
            if rowspan > 1:
                for j in range(1, rowspan):
                    if row_count + j < len(col_fill):
                        col_fill[row_count + j] += colspan
                    else:
                        col_fill.append(colspan)

        # 更新最大列数
        max_col_count = max(max_col_count, col_count)
        row_count += 1
    return row_count, max_col_count


def bytes2img(bytes, filename, img_dir):
    img = Image.open(io.BytesIO(bytes))
    img.save(os.path.join(img_dir, filename))


def postprocess(data):
    new_data = []
    for item in data:
        conv = []
        conv.append({
            "from": "human",
            "value": f"<image>\n This is a table image. "
        })
        conv.append({
            "from": "gpt",
            "value": f"I have a general understanding of the information in this table."
        })
        conv.append({
            "from": "human",
            "value": f"Based on the provided table, what is its shape? Answer with two positive integers for rows and columns, separated by a comma:"
        })
        conv.append({
            "from": "gpt",
            "value": f"{item['rows'], item['cols']}"
        })
        conv.append({
            "from": "human",
            "value": f"Based on the provided table, {item['question']}"
        })
        conv.append({
            "from": "gpt",
            "value": f"{item['ans']}"
        })
        new_data.append({
            "id": item["id"],
            "image": item["image"],
            "conversation": conv
        })
    return new_data


def process_dataset(path, img_dir):
    df = pd.read_parquet(path)
    # dataset = path.split('/')[-1].split('-')[0]
    data = []
    for index, row in tqdm(df.iterrows()):
        bytes = row.image['bytes']
        path = row.image["path"]
        html = row.text_html_table
        ques = row.question
        ans = row.loc['gt']
        bytes2img(bytes, path, img_dir)
        r, c = html2shape(html)
        data.append({
            "image": path,
            "question": ques,
            "ans": ans,
            "rows": r,
            "cols": c
        })
    return data


def process_judge_dataset(path, img_dir):
    df = pd.read_parquet(path)
    data = []
    for index, row in tqdm(df.iterrows()):
        bytes = row.image['bytes']
        path = row.image["path"]
        html = row.text_html_table
        ques = f"judge whether '{row.question}' is True or False"
        ans = str(True if row.loc['gt'] == "1" else False)
        bytes2img(bytes, path, img_dir)
        r, c = html2shape(html)
        data.append({
            "image": path,
            "question": ques,
            "ans": ans,
            "rows": r,
            "cols": c
        })
    return data


def qa_by_image(data):
    image_qa = defaultdict(list)
    image_shape = defaultdict(tuple)
    for item in tqdm(data):
        image = item["image"]
        image_qa[image].append((item["question"], item["ans"]))
        image_shape[image] = (item["rows"], item["cols"])
    new_data = []
    for k, v in image_qa.items():
        shape = image_shape[k]
        conv = [{
            "from": "human",
            "value": f"<image>\n This is a table image. Please first understand the information in the table."
        }, {
            "from": "gpt",
            "value": f"I have a general understanding of the information in this table."
        }, {
            "from": "human",
            "value": f"Based on the provided table, what is its shape? Answer with two positive integers for rows and columns, separated by a comma:"
        }, {
            "from": "gpt",
            "value": f"{shape[0]}, {shape[1]}"
        }]
        for qa in v:
            conv.append({
                "from": "human",
                "value": f"Based on the provided table, {qa[0]}"
            })
            conv.append({
                "from": "gpt",
                "value": f"{qa[1]}"
            })
        new_data.append({
            "image": k,
            "conversations": conv
        })
    return new_data


if __name__ == "__main__":
    data_dir = "../data/TableVQA-Bench/data"
    json_path = "../data/TableVQA-Bench/TableVQA.json"
    img_dir = "../data/TableVQA-Bench/images"
    files = os.listdir(data_dir)
    datas = []
    for f in files:
        path = os.path.join(data_dir, f)
        if "vtabfact" in path:  # judge True or False
            data = process_judge_dataset(path, img_dir)
        else:
            data = process_dataset(path, img_dir)

        datas.extend(data)
        # print(data)
    datas = qa_by_image(datas)
    print(datas.__len__())
    with open(json_path, 'w') as file:
        json.dump(datas, file)
