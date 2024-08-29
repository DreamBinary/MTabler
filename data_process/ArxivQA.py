# -*- coding:utf-8 -*-
# @FileName : ArxivQA_process.py
# @Time : 2024/8/14 20:32
# @Author : fiv


import re

categories = {
    'physics': "Physics",  #  物理学
    'astro-ph': "Physics",
    'hep-ph': "Physics",
    'quant-ph': "Physics",
    'cond-mat': "Physics",
    'nlin': "Physics",
    'gr-qc': "Physics",
    'hep-th': "Physics",
    'nucl-ex': "Physics",
    'hep-ex': "Physics",
    'hep-lat': "Physics",
    'nucl-th': "Physics",
    'chao-dyn': "Physics",
    'math': "Mathematics",  # 数学
    'cs': "ComputerScience",  # 计算机科学
    'q-bio': "QuantitativeBiology",  # 量化生物学
    'q-fin': "QuantitativeFinance",  # 量化金融
    'stat': "Statistics",  # 统计学
    'eess': "ElectricalEngineeringandSystemsScience",  # 电气工程和系统科学
    'econ': "Economics",  # 经济学
    'patt-sol': "-1",  # xxx 少
    'adap-org': "-1",  # xxx 少
    'mtrl-th': "-1",  # xxx 少
    'supr-con': "-1",  # xxx 少
    'chem-ph': "-1",  # xxx 少
    'math-ph': "-1",  # 数学物理学 xxx
    'cmp-lg': "-1",  # 计算机语言 xxxx 少
}

def options_process(options: list):
    new_options = []
    for op in options:
        new_options.append(re.sub(r'^[A-Z][\.\)]\s*', '', op))
    return new_options

def id_process(id: str):
    '-'.join(id.split("-")[:-1])


def process(data):
    label2idx = {chr(i): i - 65 for i in range(65, 91)}
    idx2label = [chr(i) for i in range(65, 91)]
    new_data = []
    for item in data:
        label = item["label"]
        if len(label) >= 1 and label[0].isalpha():
            # process options
            options = options_process(item["options"])
            label = label[0].upper()
            if label2idx[label] < len(options):
                new_data.append({
                    "image": item["image"],
                    "question": item["question"],
                    "options": options,
                    "label": label
                })
