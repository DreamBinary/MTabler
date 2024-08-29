# -*- coding:utf-8 -*-
# @FileName : MMTab.py
# @Time : 2024/8/17 17:45
# @Author : fiv

import json
import os

from tqdm import tqdm


def data_by_type(type):
    """
    type in eval dataset
    'table size detection': 1250,   √
    'table cell locating': 1250,
    'row and column extraction': 1250,
    'merged cell detection': 1000,
    'table cell extraction': 1250,
    'table recognition with HTML': 780, x
    'table recognition with Latex': 106,    x
    'table recognition with Markdown': 114, x
    'TABMWP_free_text': 5694,  x
    'TABMWP_multi_choice': 1992,    √
    'WTQ with the brief answer format': 4344,  x
    'FeTaQA with the long answer format': 2003,  x
    'HiTab with look-up': 1376,
    'HiTab with numerical reasoning': 200,  √
    'TAT-QA_span': 171,  √
    'TAT-QA_arithmetic': 497,   √
    'TAT-QA_multi-span': 92,    √
    'TAT-QA_count': 12, √
    'TabFact simple statement': 2105,   √
    'TabFact complex statement': 4740,
    'InfoTabs_test_alpha1': 1800,
    'InfoTabs_test_alpha2': 1800,
    'InfoTabs_test_alpha3': 1800,
    'table-to-text with highlighted cells': 7700,
    'table-to-text with highlighted cells and operators': 1584,
    'table-to-text for game summary': 334,  x
    'table-to-text for biography': 1000,    x
    'TabMCQ multi-choice question': 1029,   √
    'AIT-QA_KPI-driven': 145,
    'AIT-QA_Table-driven': 366,
    'PubHealthTab': 1942
    :param type:
    :return:
    """
    with open("../data/MMTab/MMTab-eval_test_data_49K.json") as f:
        data = json.load(f)
    data_by_type = []
    for item in data:
        if item['original_query_type'] == type:
            data_by_type.append(item)
    return data_by_type


def cp_eval_img(image):
    source = f"../data/MMTab/MMTab-eval_table_images_23K/all_test_image/{image}"
    cmd = f"cp '{source}' {img_tgt_dir}"
    os.system(cmd)


def process_shape_data(data):
    shape_data = []
    for item in tqdm(data, desc="Shape data"):
        image_id = item['image_id']
        image = image_id + ".jpg"
        cp_eval_img(image)

        shape = item['answer_list'][0]
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
        shape_data.append({
            "image": image,
            "conversations": conv
        })
    return shape_data


def process_data(data, desc="Data"):
    choice_data = []
    for item in tqdm(data, desc=desc):
        image_id = item['image_id']
        image = image_id + ".jpg"
        cp_eval_img(image)

        input = item['input']
        output = item['output']
        conv = [{
            "from": "human",
            "value": f"<image>\n This is a table image. Please first understand the information in the table."
        }, {
            "from": "gpt",
            "value": f"I have a general understanding of the information in this table."
        }, {
            "from": "human",
            "value": input
        }, {
            "from": "gpt",
            "value": output
        }]
        choice_data.append({
            "image": image,
            "conversations": conv
        })
    return choice_data


img_tgt_dir = "../data/MMTab/images"
json_tgt_path = "../data/MMTab/MMTab.json"

if __name__ == '__main__':
    all_data = []
    shape_data = data_by_type('table size detection')
    shape_data = process_shape_data(shape_data)
    all_data.extend(shape_data)

    choice_data = data_by_type('TABMWP_multi_choice')
    choice_data = process_data(choice_data, "TABMWP_multi_choice")
    all_data.extend(choice_data)

    choice_data = data_by_type('TabMCQ multi-choice question')
    choice_data = process_data(choice_data, "TabMCQ multi-choice question")
    all_data.extend(choice_data)

    arith_data = data_by_type('TAT-QA_arithmetic')
    arith_data = process_data(arith_data, "TAT-QA_arithmetic")
    all_data.extend(arith_data)

    num_data = data_by_type('HiTab with numerical reasoning')
    num_data = process_data(num_data, "HiTab with numerical reasoning")
    all_data.extend(num_data)

    cnt_data = data_by_type('TAT-QA_count')
    cnt_data = process_data(cnt_data, "TAT-QA_count")
    all_data.extend(cnt_data)

    span_data = data_by_type('TAT-QA_span')
    span_data = process_data(span_data, "TAT-QA_span")
    all_data.extend(span_data)

    span_data = data_by_type('TAT-QA_multi-span')
    span_data = process_data(span_data, "TAT-QA_multi-span")
    all_data.extend(span_data)

    statement_data = data_by_type('TabFact simple statement')
    statement_data = process_data(statement_data, "TabFact simple statement")
    all_data.extend(statement_data)
    print(len(all_data))

    with open(json_tgt_path, "w") as f:
        json.dump(all_data, f, indent=4)
