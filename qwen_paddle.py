from tqdm import tqdm

pkgs_path = "/bohr/pkgs-7x29/v21/pkgs"
# llava_lib_path = "/bohr/libb-bg5b/v3/llava"
# tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"
model_path = "Qwen/Qwen2-VL-7B-Instruct"
cache_path = "/bohr/cach-rxl3/v11/cache"
table_model_dir = "/bohr/ocrr-zlwd/v1/ch_ppstructure_openatom_SLANetv2_infer"
table_char_dict_path = "/bohr/ocrr-zlwd/v1/table_structure_dict.txt"
vllm_path = "/bohr/vllm-iq98/v1/vllm"

# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"
# vllm_path = "/personal/vllm"


import os

os.system(f"pip3 install {pkgs_path}/* --ignore-installed")
# os.system(f"cp -r {llava_lib_path} .")
os.system(f"cp -r {vllm_path} .")
# # 提交时可能不能联网，设置成离线模式防止联网失败报错
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
device = "cuda"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import json
from collections import defaultdict
from PIL import Image
import cv2
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import numpy as np
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer
import warnings
import torch
import multiprocessing
import re
import math

warnings.filterwarnings("ignore")
import logging

multiprocessing.log_to_stderr(logging.INFO)
logger = multiprocessing.get_logger()

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class TSR(TableStructurer):
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
        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
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
        bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = structure_str_list[0]
        # structure_str_list = (
        #         ["<html>", "<body>", "<table>"]
        #         + structure_str_list
        #         + ["</table>", "</body>", "</html>"]
        # )\
        structure_str_list = ["<table>"] + structure_str_list + ["</table>"]
        return structure_str_list, bbox_list

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

class Worker:
    def __init__(self):
        self.llm = None
        self.processor = None
        self.batch_size = 8
        self.submission = []
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
        tsr = TSR(args)
        if os.environ.get('DATA_PATH_B'):  # 提交时会选择隐藏的测试数据集路径（A+B榜），数据集的格式与A榜数据相同，但数目不同（5360张）
            base_dir = os.environ.get('DATA_PATH_B')
        else:
            base_dir = '/bohr/form-recognition-train-b6y2/v4'  # 示例，把A榜测试数据集路径作为测试集路径，仅开发时挂载A榜数据用于debug   # 示例，把A榜测试数据集路径作为测试集路径，仅开发时挂载A榜数据用于debug
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            data = list(json.load(f))
            data = data[:10]
        # for item in data:
        for item in tqdm(data):
            path = os.path.join(base_dir, "test_images", item["image_path"])
            img = cv2.imread(path)
            structure_res = tsr(img)
            structure_str_list, bbox_list = structure_res
            # boxes = np.array(bbox_list)
            # for box in boxes.astype(int):
            #     x1, y1, x2, y2 = box
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # output_path = os.path.join(img_output_dir, item["image_path"])
            # cv2.imwrite(output_path, img)
            rows, cols = count_rows_and_columns(structure_str_list)
            question = item["question"]
            question = question[0].lower() + question[1:]
            q3 = f"""Based on the table, caption and html structure, {question}
A) {item["options"][0]}
B) {item["options"][1]}
C) {item["options"][2]}
D) {item["options"][3]}
"""
            self.ocr_data.put(((item["image_path"], rows, cols), (path, item["caption"], structure_str_list, q3)))
            # print(f"Put {item['image_path']} into queue", flush=True)
        self.ocr_data.put(None)

    def process(self):
        flag = True
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1},
            dtype="float16"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        while flag:
            item = self.ocr_data.get()
            if item is None:
                break
            size = min(self.ocr_data.qsize(), self.batch_size)
            batch_items = [item]
            for _ in range(size):
                item = self.ocr_data.get()
                batch_items.append(item)
            if batch_items[-1] is None:
                batch_items.pop()
                flag = False
            batch_inputs = []
            outputs = []
            q3s = []
            for output, input in batch_items:
                outputs.append(output)
                # batch_images.append({
                #     "img_path": input[0],
                #     "caption": input[1],
                #     "tsr": input[2],
                #     "q3": input[3],
                # })
                msg = [
                    {"role": "system",
                     "content": "You are a helpful assistant. Provide only an label ([A-H] or [A-D]) of the correct answer for multiple-choice questions."},
                    {"role": "user", "content": [
                        {"type": "image", "image": input[0]},
                        {"type": "text",
                         "text": f'This is a table image. The caption of the table is "{input[1]}". The structure of the table in html format is as follows: {input[2]}.'}
                    ]},
                    {"role": "assistant",
                     "content": "I have a general understanding of the information in this table."},
                    {"role": "user", "content": """Based on the table, caption and html structure, which subject is most relevant to the table and caption?
A) Physics
B) Mathematics
C) Computer Science
D) Quantitative Biology
E) Quantitative Finance
F) Statistics
G) Electrical Engineering and Systems Science
H) Economics
"""
                     }]
                prompt = self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                img = self.fetch_image(input[0])
                batch_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": img},
                })
                q3s.append(input[3])
            self.batch(outputs, batch_inputs)

        with open('submission.json', 'w') as f:
            json.dump(self.submission, f)

    def batch(self, msgs, q3s, outputs, batch_inputs):
        results = self.vllm(batch_inputs)
        ans = []
        for i, r in enumerate(results):
            text = r.outputs[0].text
            msgs[i].append({
                "role": "assistant",
                "content": text,
            })
            msgs[i].append({
                "role": "user",
                "content": q3s[i],
            })
            prompt = self.processor.apply_chat_template(
                msgs[i],
                tokenize=False,
                add_generation_prompt=True,
            )
            ans.append({"subject": text})
            batch_inputs[i]["prompt"] = prompt
        results = self.vllm(batch_inputs)
        for i, r in enumerate(results):
            text = r.outputs[0].text
            ans[i]["option"] = text
            self.clean_out(outputs[i], ans[i])

    def vllm(self, inputs):
        return self.llm.generate(inputs, sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1,
            repetition_penalty=1.05,
            max_tokens=2,
            stop_token_ids=[],
        ))

    def clean_out(self, o, s):
        # print(s, flush=True)
        img_path, rows, cols = o
        category = ""
        answer = -1
        try:
            subject = s["subject"]
            match = re.search(r'[A-Za-z]', subject)
            if match:
                category = match.group(0).upper()
                category = sub_list[l2i[category]]
        except:
            category = ""
        try:
            option = s["option"]
            match = re.search(r'[A-Za-z]', option)
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
        logger.info(sub_item)
        # print(sub_item, flush=True)
        self.submission.append(sub_item)

    def fetch_image(self, img_path, size_factor: int = IMAGE_FACTOR) -> Image.Image:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        img = img.resize((resized_width, resized_height))
        return img

worker = Worker()
worker.run()