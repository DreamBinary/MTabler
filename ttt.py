# %%
import json
import os

q_prefix = "Based on the table, caption and html, "
NEW_IMG_DIR = "new_images"


def rewrite():
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data_t = json.load(f)
        data_t = list(data_t)[:10]
        # write path to json
    # new_data = []
    data = []
    for d in data_t:
        path = os.path.join(NEW_IMG_DIR, d["image_path"])
        # image = Image.open(path).convert("RGB")
        question = d["question"]
        question = question[0].lower() + question[1:]
        q3 = f"""{q_prefix}{question}
A) {d["options"][0]}
B) {d["options"][1]}
C) {d["options"][2]}
D) {d["options"][3]}
"""
        data.append({
            "path": path,
            "image_path": d["image_path"],
            "caption": d["caption"],
            # "image": image,
            "q3": q3,
        })

    with open('data.json', 'w') as f:
        json.dump(data, f)


import multiprocessing

# import logging
#
# multiprocessing.log_to_stderr(logging.INFO)
# logger = multiprocessing.get_logger()
# logging.basicConfig(filename='sgl_unitable4.log', level=logging.INFO)

p = multiprocessing.Process(target=rewrite)
p.start()
# %%
pkgs_path = "/bohr/pkgs-7x29/v18/pkgs"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"
# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"
OCR_BASE_DIR = "/bohr/ocrr-zlwd/v2/OCRCache"

# os.system(f"pip3 install {pkgs_path}/* --ignore-installed")
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
# %%
from collections import defaultdict
from sglang import Runtime
from sglang.lang.chat_template import get_chat_template
import cv2
import numpy as np
from paddleocr.paddleocr import parse_args
from paddleocr.ppstructure.table.predict_table import TableSystem
import warnings
import sglang as sgl
import torch
import re

warnings.filterwarnings("ignore")
# %%
l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()


# %%
# def count_rows_and_columns(html_tags):
#     rows = 0
#     max_columns = 0
#     current_columns = 0
#     rowspan_columns = {}
#     index = 0
#     columns_cnt = defaultdict(int)
#     while index < len(html_tags):
#         tag = html_tags[index]
# 
#         if tag == '<tr>':
#             rows += 1
#             current_columns = 0
# 
#             # Account for any ongoing rowspans from previous rows
#             for col, span in rowspan_columns.items():
#                 if span > 1:
#                     current_columns += 1
#                     rowspan_columns[col] -= 1
# 
#         elif tag.startswith('<td'):
#             colspan = 1
#             rowspan = 1
# 
#             # Check if 'colspan' and 'rowspan' are in the subsequent strings
#             if index + 1 < len(html_tags) and 'colspan="' in html_tags[index + 1]:
#                 colspan = int(html_tags[index + 1].strip().split('colspan="')[1].split('"')[0])
#                 index += 1  # Skip the colspan string
#             if index + 1 < len(html_tags) and 'rowspan="' in html_tags[index + 1]:
#                 rowspan = int(html_tags[index + 1].strip().split('rowspan="')[1].split('"')[0])
#                 index += 1  # Skip the rowspan string
# 
#             # Increment columns count
#             current_columns += colspan
# 
#             # Track rowspans for subsequent rows
#             if rowspan > 1:
#                 for _ in range(colspan):
#                     rowspan_columns[current_columns - _] = rowspan
# 
#         elif tag == '</tr>':
#             print(f"Row {rows} has {current_columns} columns")
#             columns_cnt[current_columns] += 1
#             max_columns = max(max_columns, current_columns)
# 
#         index += 1
#     columns = max(columns_cnt, key=columns_cnt.get)
#     return rows, columns
# %%
# class Runtime(sgl.srt.server.Runtime):
#     def __init__(
#             self,
#             log_level: str = "error",
#             model_overide_args: Optional[dict] = None,
#             *args,
#             **kwargs,
#     ):
#         """See the arguments in server_args.py::ServerArgs"""
#         self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)
# 
#         # Pre-allocate ports
#         self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
#             self.server_args.port,
#             self.server_args.additional_ports,
#             self.server_args.dp_size,
#         )
# 
#         self.url = self.server_args.url()
#         self.generate_url = (
#             f"http://{self.server_args.host}:{self.server_args.port}/generate"
#         )
# 
#         self.pid = None
#         # logger.info("Launching server...")
#         pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)
#         proc = multiprocessing.Process(
#             target=launch_server,
#             args=(self.server_args, model_overide_args, pipe_writer),
#         )
#         # logger.info("Waiting for server to launch...")
#         proc.start()
#         self.pid = proc.pid
#         # logger.info("Waiting for server to launch...")
#         # pipe_writer.close()
#         # timeout = 60
#         # import time
#         # start_time = time.time()
#         #
#         # while True:
#         #     logger.info("Waiting for initialization state...", flush=True)
#         #     if pipe_reader.poll(timeout=1):
#         #         logger.info("Waiting for initialization state...", flush=True)
#         #         init_state = pipe_reader.recv()
#         #         break
#         #     if time.time() - start_time > timeout:
#         #         raise TimeoutError("Timeout while waiting for initialization state")
#         # try:
#         #     init_state = pipe_reader.recv()
#         # except EOFError:
#         #     init_state = ""
#         init_state = pipe_reader.recv()
# 
#         if init_state != "init ok":
#             self.shutdown()
#             raise RuntimeError(
#                 "Initialization failed. Please see the error messages above."
#             )
#         self.endpoint = RuntimeEndpoint(self.url)
# %%
@sgl.function
def one_image(s, path, q1, q3):
    q2 = f"""{q_prefix}which subject is most relevant to the table and caption?
A) Physics
B) Mathematics
C) Computer Science
D) Quantitative Biology
E) Quantitative Finance
F) Statistics
G) Electrical Engineering and Systems Science
H) Economics
"""
    s += sgl.system(
        "You are a helpful assistant. Provide only an label ([A-H] or [A-D]) of the correct answer for multiple-choice questions.")
    # s += sgl.user(
    #     sgl.image(img_path) +
    #     f'This is a table image. The caption of the table is "{caption}". The OCR recognition result of the table in HTML format is {tsr}, which can be used as a reference but no standard answer')
    s += sgl.user(sgl.image(path) + q1)
    s += sgl.assistant("I have a general understanding of the information in this table.")
    s += sgl.user(q2)
    s += sgl.assistant(
        sgl.gen_string("subject",
                       # choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                       max_tokens=2, temperature=0.0, top_p=1
                       ))
    s += sgl.user(q3)
    s += sgl.assistant(
        sgl.gen_string("option",
                       # choices=["A", "B", "C", "D"],
                       max_tokens=2, temperature=0.0, top_p=1
                       ))


# %%
class OCR(TableSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)

        params.structure_version = "PP-StructureV2"
        params.use_gpu = False
        params.mode = "structure"

        params.det_model_dir = os.path.join(OCR_BASE_DIR, "whl", "det", "en", "en_PP-OCRv3_det_infer")
        params.rec_model_dir = os.path.join(OCR_BASE_DIR, "whl", "rec", "en", "en_PP-OCRv4_rec_infer")
        params.table_model_dir = os.path.join(OCR_BASE_DIR, "whl", "table", "en_ppstructure_mobile_v2.0_SLANet_infer")
        # params.layout_model_dir = os.path.join(BASE_DIR, "whl", "layout")

        params.rec_char_dict_path = os.path.join(OCR_BASE_DIR, "dict", "en_dict.txt")
        params.table_char_dict_path = os.path.join(OCR_BASE_DIR, "dict", "table_structure_dict.txt")
        # params.layout_dict_path = os.path.join(BASE_DIR, "dict", "layout_publaynet_dict.txt")

        super().__init__(params)

    def __call__(self, img, path):
        # result = dict()
        structure_res, elapse = self._structure(img)
        # result["cell_bbox"] = structure_res[1].tolist()
        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(img)
        # result["boxes"] = [x.tolist() for x in dt_boxes]
        boxes = [x.tolist() for x in dt_boxes]
        # result["rec_res"] = rec_res
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        # result["html"] = pred_html
        img = self.draw_bbox(img, boxes)
        cv2.imwrite(path, img)
        return pred_html

    def draw_bbox(self, img, boxes):
        # img = copy.deepcopy(img)
        boxes = np.array(boxes).astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return img


# %%
def clean_out(o, s):
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
    # logger.info(sub_item)
    return sub_item


# %%
class Worker:
    def __init__(self):
        self.batch_size = 8
        self.ocr_data = multiprocessing.Queue()
        # self.result = multiprocessing.Queue()

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

        # post = multiprocessing.Process(target=self.post_process)
        # post.start()

        self.process()
        runtime.shutdown()
        # post.join()

    def ocr(self):
        engine = OCR(layout=False, show_log=False, lang="en")
        outputs = []
        inputs = []
        with open('data.json', 'r') as f:
            data = json.load(f)
        for item in data:
            path = item["path"]
            img = cv2.imread(path)
            html = engine(img, path)
            rows, cols = -1, -1
            q1 = f'This is a table image. The caption of the table is "{item["caption"]}". The result of OCR in html format is as follows: {html}.'
            outputs.append((item["image_path"], rows, cols))
            inputs.append({"path": item["path"], "q1": q1, "q3": item["q3"]})
            if len(outputs) == self.batch_size:
                self.ocr_data.put((outputs, inputs))
                outputs, inputs = [], []
        if outputs:
            self.ocr_data.put((outputs, inputs))
        self.ocr_data.put(None)

    def process(self):
        flag = True
        submission = []
        while flag:
            item = self.ocr_data.get()
            if item is None:
                break
            outputs, inputs = item
            states = one_image.run_batch(inputs)
            for o, s in zip(outputs, states):
                sub_item = clean_out(o, s)
                submission.append(sub_item)
        if len(submission) != 5360:
            raise Exception(f"Submission length is {len(submission)}")
        with open('submission.json', 'w') as f:
            json.dump(submission, f)
        #     self.result.put((outputs, results))
        # self.result.put(None)

    # def post_process(self):
    #     submission = []
    #     while True:
    #         item = self.result.get()
    #         if item is None:
    #             break
    #         outputs, states = item
    #         for o, s in zip(outputs, states):
    #             sub_item = clean_out(o, s)
    #             submission.append(sub_item)
    #     if len(submission) != 5360:
    #         raise Exception(f"Submission length is {len(submission)}")
    #     with open('submission.json', 'w') as f:
    #         json.dump(submission, f)


# %%
p.join()
worker = Worker()
worker.run()
