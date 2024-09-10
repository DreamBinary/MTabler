#%%
q_prefix = "Based on the table, caption and html structure, "


def rewrite():
    import os
    import json
    NEW_IMG_DIR = "new_images"
    os.makedirs(NEW_IMG_DIR, exist_ok=True)
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data_t = json.load(f)
        data_t = list(data_t)[:10]
    data = []
    for d in data_t:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        # w_path = os.path.join(NEW_IMG_DIR, d["image_path"])
        question = d["question"]
        question = question[0].lower() + question[1:]
        q3 = f"""{q_prefix}{question}
A) {d["options"][0]}
B) {d["options"][1]}
C) {d["options"][2]}
D) {d["options"][3]}
"""
        data.append({
            "r_path": r_path,
            # "w_path": w_path,
            "image_path": d["image_path"],
            "caption": d["caption"],
            "q3": q3,
        })

    with open('data.json', 'w') as f:
        json.dump(data, f)


import multiprocessing

import logging

multiprocessing.log_to_stderr(logging.INFO)
logger = multiprocessing.get_logger()
logging.basicConfig(filename='sgl_unitable4.log', level=logging.INFO)

p = multiprocessing.Process(target=rewrite)
p.start()
#%%
pkgs_path = "/bohr/pkgs-7x29/v23/pkgs"
model_path = "Qwen/Qwen2-VL-7B-Instruct"
cache_path = "/bohr/cach-rxl3/v11/cache"
table_model_dir = "/bohr/ocrr-zlwd/v1/ch_ppstructure_openatom_SLANetv2_infer"
table_char_dict_path = "/bohr/ocrr-zlwd/v1/table_structure_dict.txt"

import os

# os.system(f"pip3 install {pkgs_path}/* --ignore-installed")
# # 提交时可能不能联网，设置成离线模式防止联网失败报错
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
device = "cuda"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
#%%
import json
from collections import defaultdict
import cv2
import numpy as np
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer
import warnings
import re
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from swift.utils import seed_everything
import torch

warnings.filterwarnings("ignore")
# import logging
#
# multiprocessing.log_to_stderr(logging.INFO)
# logger = multiprocessing.get_logger()
#%%
l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()
#%%
class OCR(TableStructurer):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, img):
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds["structure_probs"] = outputs[1]
        preds["loc_preds"] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result["structure_batch_list"][0]
        # bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = structure_str_list[0]
        # structure_str_list = (
        #         ["<html>", "<body>", "<table>"]
        #         + structure_str_list
        #         + ["</table>", "</body>", "</html>"]
        # )\
        structure_str_list = ["<table>"] + structure_str_list + ["</table>"]
        return structure_str_list
#%%
def count_rows_and_columns(html_tags):
    rows = 0
    max_columns = 0
    current_columns = 0
    rowspan_columns = {}
    index = 0
    columns_cnt = defaultdict(int)
    while index < len(html_tags):
        tag = html_tags[index]

        if tag == '<tr>':
            rows += 1
            current_columns = 0

            # Account for any ongoing rowspans from previous rows
            for col, span in rowspan_columns.items():
                if span > 1:
                    current_columns += 1
                    rowspan_columns[col] -= 1

        elif tag.startswith('<td'):
            colspan = 1
            rowspan = 1

            # Check if 'colspan' and 'rowspan' are in the subsequent strings
            if index + 1 < len(html_tags) and 'colspan="' in html_tags[index + 1]:
                colspan = int(html_tags[index + 1].strip().split('colspan="')[1].split('"')[0])
                index += 1  # Skip the colspan string
            if index + 1 < len(html_tags) and 'rowspan="' in html_tags[index + 1]:
                rowspan = int(html_tags[index + 1].strip().split('rowspan="')[1].split('"')[0])
                index += 1  # Skip the rowspan string

            # Increment columns count
            current_columns += colspan

            # Track rowspans for subsequent rows
            if rowspan > 1:
                for _ in range(colspan):
                    rowspan_columns[current_columns - _] = rowspan

        elif tag == '</tr>':
            # print(f"Row {rows} has {current_columns} columns")
            columns_cnt[current_columns] += 1
            max_columns = max(max_columns, current_columns)

        index += 1
    columns = max(columns_cnt, key=columns_cnt.get)
    return rows, columns
#%%
q2 = f"""{q_prefix}which subject is most relevant to the table or caption?
A) Physics
B) Mathematics
C) Computer Science
D) Quantitative Biology
E) Quantitative Finance
F) Statistics
G) Electrical Engineering and Systems Science
H) Economics
"""


def clean_out(self, o, r2, r3):
    # print(s, flush=True)
    img_path, rows, cols = o
    category = ""
    answer = -1
    try:
        match = re.search(r'[A-Za-z]', r2)
        if match:
            category = match.group(0).upper()
            category = sub_list[l2i[category]]
    except:
        category = ""
    try:
        match = re.search(r'[A-Za-z]', r3)
        if match:
            answer = match.group(0).upper()
            answer = l2i[answer]
    except:
        answer = -1
    sub_item = {
        "image_path": img_path,
        "category": category,
        "cols": cols,
        "rows": rows,
        "answer": answer,
    }
    return sub_item


class Worker:
    def __init__(self):
        self.llm = None
        self.processor = None
        self.batch_size = 8
        self.ocr_data = multiprocessing.Queue()

    def run(self):
        ocr_process = multiprocessing.Process(target=self.ocr)
        ocr_process.start()
        self.process()

    def ocr(self):
        args = type("Args", (), {
            "table_model_dir": table_model_dir,
            "table_char_dict_path": table_char_dict_path,
            "use_gpu": False,
            # "gpu_id": 0,
            # "gpu_mem": 500,
            "use_npu": False,
            "use_mlu": False,
            "use_xpu": False,
            "precision": "fp32",
            "benchmark": False,
            "use_tensorrt": False,
            "use_onnx": False,
            "table_max_len": 1024,
            "enable_mkldnn": True,
            "table_algorithm": "SLANet",
            "merge_no_span_structure": True,
            "cpu_threads": 16,
        })()
        engine = OCR(args)
        with open("data.json", 'r') as f:
            data = json.load(f)
            # data = data[:10]
        for item in data:
            img = cv2.imread(item["r_path"])
            html = engine(img)
            rows, cols = count_rows_and_columns(html)
            q1 = f'<image>\nThis is a table image. The caption of the table is "{item["caption"]}". The structure of the table in html format is as follows: {html}.'
            self.ocr_data.put(((item["image_path"], rows, cols), (item["r_path"], q1, item["q3"])))
            # print(f"Put {item['image_path']} into queue", flush=True)
        self.ocr_data.put(None)

    def process(self):
        model_type = ModelType.qwen2_vl_7b_instruct
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(
            model_type, torch.bfloat16, model_id_or_path=model_path, model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 2
        template = get_template(template_type, tokenizer)
        seed_everything(42)
        system = "You are a helpful assistant. Provide only an label ([A-H] or [A-D]) of the correct answer for multiple-choice questions."
        submission = []
        while True:
            item = self.ocr_data.get()
            if item is None:
                break
            output, input = item
            path, q1, q3 = input
            history = [
                ('user', q1),
                ('assistant', f'I have a general understanding of the information in this table.')
            ]
            r2, history = inference(model, template, q2, history, system, [path])
            r3, history = inference(model, template, q3, history, system, [path])
            sub_item = clean_out(output, r2, r3)
            logger.info(sub_item)
            submission.append(sub_item)

        # if len(submission) != 5360:
        #     raise Exception(f"Submission length is {len(submission)}")
        with open('submission.json', 'w') as f:
            json.dump(submission, f)
#%%
worker = Worker()
p.join()
worker.run()