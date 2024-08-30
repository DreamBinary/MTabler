import multiprocessing
import re

pkgs_path = "/bohr/pkgs-7x29/v15/pkgs"
# llava_lib_path = "/bohr/libb-bg5b/v3/llava"
# tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"
table_model_dir = "/bohr/ocrr-zlwd/v1/ch_ppstructure_openatom_SLANetv2_infer"
table_char_dict_path = "/bohr/ocrr-zlwd/v1/table_structure_dict.txt"
# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"


import os

# os.system(f"pip install {pkgs_path}/* --ignore-installed")
# os.system(f"cp -r {llava_lib_path} .")
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
import multiprocessing as mp
from collections import defaultdict
from typing import Optional

from sglang.lang.chat_template import get_chat_template
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import allocate_init_ports
from sglang import RuntimeEndpoint
import cv2
import numpy as np
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer

import warnings
import sglang as sgl
import torch

warnings.filterwarnings("ignore")

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()


# disable_torch_init()


class TSR(TableStructurer):
    def __init__(self, args):
        # python table/predict_structure.py - -table_model_dir =../ inference / slanet_lcnetv2_infer / --table_char_dict_path =../ ppocr / utils / dict / table_structure_dict.txt - -image_dir = docs / table / table.jpg - -output =../ output / table_slanet_lcnetv2 - -use_gpu = False - -benchmark = True - -enable_mkldnn = True - -table_max_len = 512

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


class Runtime(sgl.srt.server.Runtime):
    def __init__(
            self,
            log_level: str = "error",
            model_overide_args: Optional[dict] = None,
            *args,
            **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port,
            self.server_args.additional_ports,
            self.server_args.dp_size,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        self.pid = None
        # logger.info("Launching server...")
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, model_overide_args, pipe_writer),
        )
        # logger.info("Waiting for server to launch...")
        proc.start()
        self.pid = proc.pid
        # logger.info("Waiting for server to launch...")
        # pipe_writer.close()
        # timeout = 60
        # import time
        # start_time = time.time()
        #
        # while True:
        #     logger.info("Waiting for initialization state...", flush=True)
        #     if pipe_reader.poll(timeout=1):
        #         logger.info("Waiting for initialization state...", flush=True)
        #         init_state = pipe_reader.recv()
        #         break
        #     if time.time() - start_time > timeout:
        #         raise TimeoutError("Timeout while waiting for initialization state")
        # try:
        #     init_state = pipe_reader.recv()
        # except EOFError:
        #     init_state = ""
        init_state = pipe_reader.recv()

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        self.endpoint = RuntimeEndpoint(self.url)


@sgl.function
def one_image(s, img_path, caption, q3, tsr):
    q2 = """
Based on the table, caption and html structure, which subject is most relevant to the table?
A) Physics
B) Mathematics
C) Computer Science
D) Quantitative Biology
E) Quantitative Finance
F) Statistics
G) Electrical Engineering and Systems Science
H) Economics
"""
    s += sgl.system("You are a helpful assistant. Please only provide the letter corresponding to the correct answer for multiple-choice questions, without any additional information.")
    s += sgl.user(
        sgl.image(img_path) +
        f'This is a table image. The caption of the table is "{caption}". The structure of the table in html format is as follows: {tsr}.')
    s += sgl.assistant("I have a general understanding of the information in this table.")
    s += sgl.user(q2)
    s += sgl.assistant(
        sgl.gen("subject",
                # choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                max_tokens=4, temperature=0.0, top_p=1))
    s += sgl.user(q3)
    s += sgl.assistant(sgl.gen("option",
                               # choices=["A", "B", "C", "D"],
                               max_tokens=4, temperature=0.0, top_p=1))


def count_rows_and_columns(html_tags):
    rows = 0
    max_columns = 0
    current_columns = 0
    rowspan_columns = {}

    for tag in html_tags:
        if tag.startswith('<tr>'):
            rows += 1
            current_columns = 0
            for col, span in rowspan_columns.items():
                if span > 1:
                    current_columns += 1
                    rowspan_columns[col] -= 1
            max_columns = max(max_columns, current_columns)
        elif tag.startswith('<td'):
            current_columns += 1
            if 'rowspan' in tag:
                rowspan_value = int(tag.split('rowspan="')[1].split('"')[0])
                rowspan_columns[current_columns] = rowspan_value
        elif tag == '</tr>':
            max_columns = max(max_columns, current_columns)

    return rows, max_columns


class Worker:
    def __init__(self):
        self.batch_size = 8
        self.submission = []
        self.ocr_data = multiprocessing.Queue()

    def run(self):
        ocr_process = multiprocessing.Process(target=self.ocr)
        ocr_process.start()

        model_overide_args = {
            "attn_implementation": "eager",
            "multimodal": True,
            "overwrite_config": {
                "image_aspect_ratio": "anyres_max_9"
            }
        }
        runtime = Runtime(
            model_path=model_path,
            model_overide_args=model_overide_args,
        )
        runtime.endpoint.chat_template = get_chat_template("qwen")
        sgl.set_default_backend(runtime)
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
            data = json.load(f)
        for item in data:
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
        self.ocr_data.put(None)

    def process(self):
        flag = True
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
            batch_images = []
            outputs = []
            for output, input in batch_items:
                outputs.append(output)
                batch_images.append({
                    "img_path": input[0],
                    "caption": input[1],
                    "tsr": input[2],
                    "q3": input[3],
                })
            self.batch(outputs, batch_images)

        with open('submission.json', 'w') as f:
            json.dump(self.submission, f)

    def batch(self, outputs, batch_images):
        states = one_image.run_batch(batch_images)
        for o, s in zip(outputs, states):
            self.clean_out(o, s)

    def clean_out(self, o, s):
        print(s)
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
        print(sub_item)
        self.submission.append(sub_item)


worker = Worker()
worker.run()
