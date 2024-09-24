import json
import multiprocessing
import os
import re
import warnings
from collections import defaultdict

import cv2
import math
import numpy as np
import torch
from PIL import Image
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

model_path = "/bohr/cach-rxl3/v17/cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f"
# raw_cache_path = "/bohr/cach-rxl3/v11/cache"
cache_path = "/bohr/cach-rxl3/v17/cache"
table_model_dir = "/bohr/ocrr-zlwd/v1/ch_ppstructure_openatom_SLANetv2_infer"
table_char_dict_path = "/bohr/ocrr-zlwd/v1/table_structure_dict.txt"
# os.system(f"cp -r {raw_cache_path} .")
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
device = "cuda"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
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
        # bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = structure_str_list[0]
        # structure_str_list = (
        #         ["<html>", "<body>", "<table>"]
        #         + structure_str_list
        #         + ["</table>", "</body>", "</html>"]
        # )\
        structure_str_list = ["<table>"] + structure_str_list + ["</table>"]
        return structure_str_list


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
            columns_cnt[current_columns] += 1
            max_columns = max(max_columns, current_columns)

        index += 1
    columns = max(columns_cnt, key=columns_cnt.get)
    return rows, columns


batch_size = 8
submission = []
data = multiprocessing.Queue()


def ocr():
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
    # NEW_IMG_DIR = "new_images"
    # os.makedirs(NEW_IMG_DIR, exist_ok=True)
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            data_t = json.load(f)
            # data_t = list(json.load(f))[:10]
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            data_t = list(json.load(f))[:10]
    q_prefix = "Based on the latex table, caption and html structure, "
    shapes = []
    paths = []
    q2s = []
    q3s = []
    for d in data_t:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        # w_path = os.path.join(NEW_IMG_DIR, d["image_path"])
        img = cv2.imread(r_path)
        html = tsr(img)
        q1 = f'This is a table image. The caption of the table is "{d["caption"]}". The structure of the table in html format is as follows: {html}.'
        q2 = f"""{q1}{q_prefix}which subject is most relevant to the table or caption?
A) Physics
B) Mathematics
C) Computer Science
D) Quantitative Biology
E) Quantitative Finance
F) Statistics
G) Electrical Engineering and Systems Science
H) Economics
"""
        question = d["question"]
        question = question[0].lower() + question[1:]
        q3 = f"""{q1}{q_prefix}{question}
A) {d["options"][0]}
B) {d["options"][1]}
C) {d["options"][2]}
D) {d["options"][3]}
"""
        rows, cols = count_rows_and_columns(html)
        shapes.append((d["image_path"], rows, cols))
        paths.append(r_path)
        q2s.append(q2)
        q3s.append(q3)
        if len(shapes) == batch_size:
            data.put((shapes, paths, q2s, q3s))
            shapes, paths, q2s, q3s = [], [], [], []
    if len(shapes) > 0:
        data.put((shapes, paths, q2s, q3s))
    data.put(None)


def process():
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
    )
    processor = AutoProcessor.from_pretrained(model_path)

    def create_msgs(paths, questions):
        return [
            processor.apply_chat_template([
                {"role": "system",
                 "content": "You are a helpful assistant. Provide only a label ([A-H] or [A-D]) of the correct answer for multiple-choice questions."},
                {"role": "user", "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": q}
                ]}
            ], tokenize=False, add_generation_prompt=True)
            for path, q in zip(paths, questions)
        ]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=16,
        stop_token_ids=[],
    )

    while True:
        item = data.get()
        if item is None:
            break
        shapes, paths, q2s, q3s = item
        batch_inputs = []
        imgs = [fetch_image(path) for path in paths]

        msgs_q2 = create_msgs(paths, q2s)
        msgs_q3 = create_msgs(paths, q3s)
        for msgs in [msgs_q2, msgs_q3]:
            for i, m in zip(imgs, msgs):
                batch_inputs.append({
                    "prompt": m,
                    "multi_modal_data": {
                        "image": i
                    }
                })

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        # print("-->> OUTPUTS")
        # print(outputs)
        ans = [output.outputs[0].text for output in outputs]
        # print("-->> ANSWERS")
        # print(ans)
        clean_out(shapes, ans)
    if len(submission) != 5360:
        with open('error.json', 'w') as f:
            json.dump(submission, f)
        raise Exception(f"Submission length is {len(submission)}")
    with open('submission.json', 'w') as f:
        json.dump(submission, f)


def clean_out(shapes, ans):
    l = len(shapes)
    for i in range(l):
        image_path, rows, cols = shapes[i]
        subject = ans[i]
        option = ans[i + l]
        category = ""
        answer = -1
        try:
            match = re.search(r'[A-Za-z]', subject)
            if match:
                category = match.group(0).upper()
                category = sub_list[l2i[category]]
        except:
            category = ""
        try:
            match = re.search(r'[A-Za-z]', option)
            if match:
                answer = match.group(0).upper()
                answer = l2i[answer]
        except:
            answer = -1
        sub_item = {
            "image_path": image_path,
            "category": category,
            "cols": cols,
            "rows": rows,
            "answer": answer,
        }
        # print(sub_item)
        submission.append(sub_item)


def fetch_image(img_path, size_factor: int = IMAGE_FACTOR) -> Image.Image:
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


ocr_process = multiprocessing.Process(target=ocr)
ocr_process.start()

process()
