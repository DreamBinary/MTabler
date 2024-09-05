def rewrite():
    import os
    import json
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data = json.load(f)
        data = list(data)[:1000]
    with open("./data.json", 'w') as f:
        # write path to json
        new_data = []
        for d in data:
            d["path"] = os.path.join(base_dir, "test_images", d["image_path"])
            new_data.append(d)
        json.dump(new_data, f)


import multiprocessing

p = multiprocessing.Process(target=rewrite)
p.start()

pkgs_path = "/bohr/pkgs-7x29/v18/pkgs"
# llava_lib_path = "/bohr/libb-bg5b/v3/llava"
# tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"
unitable_model = "/bohr/unii-7sxm/v1/unitable/weights"
unitable_vocab = "/bohr/unii-7sxm/v1/unitable/vocab"
unitable_src = "/bohr/unii-7sxm/v1/unitable/src"
new_json = "./data.json"
# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"


import os

# os.system(f"pip install {pkgs_path}/* --ignore-installed")
os.system(f"cp -r {unitable_src} .")
# os.system(f"cp -r {llava_lib_path} .")
# # 提交时可能不能联网，设置成离线模式防止联网失败报错
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
device = "cuda"

import json
import torch.multiprocessing as multiprocessing

from sglang.lang.chat_template import get_chat_template
import sglang as sgl
from sglang import Runtime

import re
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple, Sequence, Optional, Union

import tokenizers as tk
import torch
from PIL import Image
from torch import nn, Tensor
from torchvision import transforms

from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
from src.utils import subsequent_mask, pred_token_within_range, greedy_sampling, html_str_to_token_list
from src.vocab import (
    HTML_TOKENS,
    TASK_TOKENS,
    RESERVED_TOKENS,
    BBOX_TOKENS,
)

warnings.filterwarnings('ignore')
import logging

multiprocessing.log_to_stderr(logging.INFO)
logger = multiprocessing.get_logger()

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()

MODEL_DIR = Path(unitable_model)
MODEL_FILE_NAME = ["unitable_large_structure.pt", "unitable_large_bbox.pt", "unitable_large_content.pt"]
VOCAB_DIR = Path(unitable_vocab)
VOCAB_HTML = VOCAB_DIR / "vocab_html.json"
VOCAB_BBOX = VOCAB_DIR / "vocab_bbox.json"
VOCAB_CELL = VOCAB_DIR / "vocab_cell_6k.json"

VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
INVALID_CELL_TOKEN = (["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS)
VALID_BBOX_TOKEN = ["<eos>"] + BBOX_TOKENS  # image size will be addressed after instantiation

# UniTable large model
d_model = 768
patch_size = 16
nhead = 12
dropout = 0.2


def autoregressive_decode(
        model: EncoderDecoder,
        image: Tensor,
        prefix: Sequence[int],
        max_decode_len: int,
        eos_id: int,
        token_whitelist: Optional[Sequence[int]] = None,
        token_blacklist: Optional[Sequence[int]] = None,
) -> Tensor:
    with torch.inference_mode():
        memory = model.encode(image)
        context = torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)

        for _ in range(max_decode_len):
            eos_flag = [eos_id in k for k in context]
            if all(eos_flag):
                break

            causal_mask = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(
                memory, context, tgt_mask=causal_mask, tgt_padding_mask=None
            )
            logits = model.generator(logits)[:, -1, :]

            logits = pred_token_within_range(
                logits.detach(),
                white_list=token_whitelist,
                black_list=token_blacklist,
            )

            next_probs, next_tokens = greedy_sampling(logits)
            context = torch.cat([context, next_tokens], dim=1)
    return context


def load_vocab_and_model(
        vocab_path: Union[str, Path],
        max_seq_len: int,
        model_weights: Union[str, Path],
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
) -> Tuple[tk.Tokenizer, EncoderDecoder]:
    vocab_path = str(vocab_path)
    vocab = tk.Tokenizer.from_file(vocab_path)
    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=d_model,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=max_seq_len,
        dropout=dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    model.load_state_dict(torch.load(model_weights, map_location="cpu"))
    model = torch.compile(model)
    model = model.to(device)
    return vocab, model


def image_to_tensor(image: Image, size: Tuple[int, int]) -> Tensor:
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.86597056, 0.88463002, 0.87491087],
            std=[0.20686628, 0.18201602, 0.18485524])
    ])
    image_tensor = T(image)
    image_tensor = image_tensor.to(device).unsqueeze(0)

    return image_tensor


def rescale_bbox(
        bbox: Sequence[Sequence[float]],
        src: Tuple[int, int],
        tgt: Tuple[int, int]
) -> Sequence[Sequence[float]]:
    assert len(src) == len(tgt) == 2
    ratio = [tgt[0] / src[0], tgt[1] / src[1]] * 2
    bbox = [[int(round(i * j)) for i, j in zip(entry, ratio)] for entry in bbox]
    return bbox


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


# class Unitable:
#     def __init__(self):
#         # manager = multiprocessing.Manager()
#         # self.html = manager.list()
#         # self.bbox = manager.list()
#         # self.cell_input = multiprocessing.Queue()
#         # self.shape = manager.list()
#         self.result = multiprocessing.Queue()
#
#     # def run(self):
#     #     ps = [
#     #         multiprocessing.Process(target=self.img_tsr),
#     #         # multiprocessing.Process(target=self.img_bbox),
#     #         # multiprocessing.Process(target=self.img_tcr),
#     #     ]
#     #     for p in ps:
#     #         p.start()
#     #     for p in ps:
#     #         p.join()
#
#     def get(self):
#         return self.result.get()
#
#     def size(self):
#         return self.result.qsize()
#
#         # def __call__(self, image):
#
#     #     html = self.img_tsr(image)
#     #     bbox = self.img_bbox(image)
#     #     cell = self.img_tcr(image, bbox)
#     #     code = self.img2html(html, cell)
#     #     image = self.draw_bbox(image, bbox)
#     #     rows, cols = count_rows_and_columns(html)
#     #     print("HTML:", html, flush=True)
#     #     print("BBOX:", bbox, flush=True)
#     #     print("CODE:", code, flush=True)
#     #     print("ROWS, COLS:", rows, cols, flush=True)
#     #     return image, code, rows, cols
#     # return self.img_tsr(image)
#
#     def img_tsr(self):
#         backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
#         encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
#                           norm_first=True, nlayer=12, ff_ratio=4)
#         decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
#                           norm_first=True, nlayer=4, ff_ratio=4)
#         vocab_html, model_html = load_vocab_and_model(
#             vocab_path=VOCAB_HTML,
#             max_seq_len=784,
#             model_weights=MODEL_DIR / MODEL_FILE_NAME[0],
#             backbone=backbone,
#             encoder=encoder,
#             decoder=decoder
#         )
#         with open(new_json, 'r') as f:
#             data = json.load(f)
#
#         # for item in data:
#         for item in data:
#             image = Image.open(item["path"]).convert("RGB")
#             # Image transformation
#             image_tensor = image_to_tensor(image, size=(448, 448))
#
#             # Inference
#             pred_html = autoregressive_decode(
#                 model=model_html,
#                 image=image_tensor,
#                 prefix=[vocab_html.token_to_id("[html]")],
#                 max_decode_len=512,
#                 eos_id=vocab_html.token_to_id("<eos>"),
#                 token_whitelist=[vocab_html.token_to_id(i) for i in VALID_HTML_TOKEN],
#                 token_blacklist=None
#             )
#
#             # Convert token id to token text
#             pred_html = pred_html.detach().cpu().numpy()[0]
#             pred_html = vocab_html.decode(pred_html, skip_special_tokens=False)
#             pred_html = html_str_to_token_list(pred_html)
#
#             rows, cols = count_rows_and_columns(pred_html)
#             # self.shape.append((rows, cols))
#             self.result.put((item, rows, cols, pred_html))
#             # logger.info(f"Image {idx} processed")
#         self.result.put(None)
#         #     if self.bbox:
#         #         self.cell_input.put((pred_html, self.bbox.pop(0)))
#         #     else:
#         #         self.html.append(pred_html)
#         # if not self.html and not self.bbox:
#         #     self.cell_input.put(None)
#
#     # def img_bbox(self):
#     #     backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
#     #     encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
#     #                       norm_first=True, nlayer=12, ff_ratio=4)
#     #     decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
#     #                       norm_first=True, nlayer=4, ff_ratio=4)
#     #     vocab_bbox, model_bbox = load_vocab_and_model(
#     #         vocab_path=VOCAB_BBOX,
#     #         max_seq_len=1024,
#     #         model_weights=MODEL_DIR / MODEL_FILE_NAME[1],
#     #         backbone=backbone,
#     #         encoder=encoder,
#     #         decoder=decoder
#     #     )
#     #     for item in data:
#     #         image = Image.open(os.path.join(base_dir, "test_images", item["image_path"])).convert("RGB")
#     #
#     #         # Image transformation
#     #         image_tensor = image_to_tensor(image, size=(448, 448))
#     #         image_size = image.size
#     #         # Inference
#     #         pred_bbox = autoregressive_decode(
#     #             model=model_bbox,
#     #             image=image_tensor,
#     #             prefix=[vocab_bbox.token_to_id("[bbox]")],
#     #             max_decode_len=1024,
#     #             eos_id=vocab_bbox.token_to_id("<eos>"),
#     #             token_whitelist=[vocab_bbox.token_to_id(i) for i in VALID_BBOX_TOKEN[: 449]],
#     #             token_blacklist=None
#     #         )
#     #
#     #         # Convert token id to token text
#     #         pred_bbox = pred_bbox.detach().cpu().numpy()[0]
#     #         pred_bbox = vocab_bbox.decode(pred_bbox, skip_special_tokens=False)
#     #         pred_bbox = bbox_str_to_token_list(pred_bbox)
#     #         pred_bbox = rescale_bbox(pred_bbox, src=(448, 448), tgt=image_size)
#     #         if self.html:
#     #             self.cell_input.put((self.html.pop(0), pred_bbox))
#     #         else:
#     #             self.bbox.append(pred_bbox)
#     #     if not self.html and not self.bbox:
#     #         self.cell_input.put(None)
#
#     # def img_tcr(self):
#     #     backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
#     #     encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
#     #                       norm_first=True, nlayer=12, ff_ratio=4)
#     #     decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
#     #                       norm_first=True, nlayer=4, ff_ratio=4)
#     #     vocab_cell, model_cell = load_vocab_and_model(
#     #         vocab_path=VOCAB_CELL,
#     #         max_seq_len=200,
#     #         model_weights=MODEL_DIR / MODEL_FILE_NAME[2],
#     #         backbone=backbone,
#     #         encoder=encoder,
#     #         decoder=decoder
#     #     )
#     #     idx = 0
#     #     while True:
#     #         item = self.cell_input.get()
#     #         if item is None:
#     #             break
#     #         html, bbox = item
#     #         image = Image.open(os.path.join(base_dir, "test_images", data[idx]["image_path"])).convert("RGB")
#     #
#     #         # Cell image cropping and transformation
#     #         image_tensor = [image_to_tensor(image.crop(b), size=(112, 448)) for b in bbox]
#     #         image_tensor = torch.cat(image_tensor, dim=0)
#     #
#     #         # Inference
#     #         pred_cell = autoregressive_decode(
#     #             model=model_cell,
#     #             image=image_tensor,
#     #             prefix=[vocab_cell.token_to_id("[cell]")],
#     #             max_decode_len=200,
#     #             eos_id=vocab_cell.token_to_id("<eos>"),
#     #             token_whitelist=None,
#     #             token_blacklist=[vocab_cell.token_to_id(i) for i in INVALID_CELL_TOKEN]
#     #         )
#     #
#     #         # Convert token id to token text
#     #         pred_cell = pred_cell.detach().cpu().numpy()
#     #         pred_cell = vocab_cell.decode_batch(pred_cell, skip_special_tokens=False)
#     #         pred_cell = [cell_str_to_token_list(i) for i in pred_cell]
#     #         pred_cell = [re.sub(r'(\d).\s+(\d)', r'\1.\2', i) for i in pred_cell]
#     #
#     #         code = build_table_from_html_and_cell(html, pred_cell)
#     #         code = "".join(code)
#     #
#     #         self.result.put((idx, self.shape.pop(0), code))
#     #
#     #         # logger.info(f"Image {idx} processed")
#     #         # logger.info(f"HTML: {html}")
#     #         # logger.info(f"BBOX: {bbox}")
#     #         # logger.info(f"CODE: {code}")
#     #         idx += 1
#     #     self.result.put(None)
#
#     # def draw_bbox(self, image, bbox):
#     #     draw = ImageDraw.Draw(image)
#     #     for b in bbox:
#     #         draw.rectangle(b, outline="red", width=1)
#     #     return image


@sgl.function
def one_image(s, path, q1, q3):
    q2 = """Based on the table, caption and html structure, which subject is most relevant to the table and caption?
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
        sgl.gen("subject",
                # choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                max_tokens=2, temperature=0.0, top_p=1
                ))
    s += sgl.user(q3)
    s += sgl.assistant(
        sgl.gen("option",
                # choices=["A", "B", "C", "D"],
                max_tokens=2, temperature=0.0, top_p=1
                ))





class Worker:
    def __init__(self):
        self.batch_size = 128
        self.ocr_result = multiprocessing.Queue()
        self.input_data = multiprocessing.Queue()
        self.result = multiprocessing.Queue()

    def run(self):
        uni = multiprocessing.Process(target=self.unitable)
        pre = multiprocessing.Process(target=self.preprocess)
        uni.start()
        pre.start()

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
            disable_regex_jump_forward=True,
            enable_torch_compile=True,
        )
        runtime.endpoint.chat_template = get_chat_template("qwen")
        sgl.set_default_backend(runtime)

        post = multiprocessing.Process(target=self.post_process)
        post.start()

        self.process()
        runtime.shutdown()
        post.join()
        # ocr.join()

    def unitable(self):
        backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
        encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
                          norm_first=True, nlayer=12, ff_ratio=4)
        decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
                          norm_first=True, nlayer=4, ff_ratio=4)
        vocab_html, model_html = load_vocab_and_model(
            vocab_path=VOCAB_HTML,
            max_seq_len=784,
            model_weights=MODEL_DIR / MODEL_FILE_NAME[0],
            backbone=backbone,
            encoder=encoder,
            decoder=decoder
        )
        with open(new_json, 'r') as f:
            data = json.load(f)

        # for item in data:
        for item in data:
            image = Image.open(item["path"]).convert("RGB")
            # Image transformation
            image_tensor = image_to_tensor(image, size=(448, 448))

            # Inference
            pred_html = autoregressive_decode(
                model=model_html,
                image=image_tensor,
                prefix=[vocab_html.token_to_id("[html]")],
                max_decode_len=512,
                eos_id=vocab_html.token_to_id("<eos>"),
                token_whitelist=[vocab_html.token_to_id(i) for i in VALID_HTML_TOKEN],
                token_blacklist=None
            )

            # Convert token id to token text
            pred_html = pred_html.detach().cpu().numpy()[0]
            pred_html = vocab_html.decode(pred_html, skip_special_tokens=False)
            pred_html = html_str_to_token_list(pred_html)

            rows, cols = count_rows_and_columns(pred_html)
            # self.shape.append((rows, cols))
            self.ocr_result.put((item, rows, cols, pred_html))
            logger.info(f"Image {item['image_path']} processed {rows} {cols}")
            # logger.info(f"Image {idx} processed")
        self.ocr_result.put(None)

    def preprocess(self):
        flag = True
        while flag:
            ocr_result = self.ocr_result.get()
            if ocr_result is None:
                break
            # size = min(self.ocr_result.size(), self.batch_size)
            size = self.ocr_result.qsize()
            batch_result = [ocr_result]
            for _ in range(size):
                batch_result.append(self.ocr_result.get())
            if batch_result[-1] is None:
                batch_result.pop()
                flag = False
            batch_output = []
            batch_data = []
            for item, rows, cols, tsr in batch_result:
                question = item["question"]
                question = question[0].lower() + question[1:]
                q3 = f"""Based on the table, caption and html structure, {question}
A) {item["options"][0]}
B) {item["options"][1]}
C) {item["options"][2]}
D) {item["options"][3]}
"""
                q1 = f'This is a table image. The caption of the table is "{item["caption"]}". The structure of the table in html format is as follows: {tsr}.'
                batch_output.append((item["image_path"], rows, cols))
                batch_data.append({
                    "path": item["path"],
                    "q1": q1,
                    "q3": q3,
                })
            self.input_data.put((batch_output, batch_data))
        self.input_data.put(None)

    def process(self):
        while True:
            item = self.input_data.get()
            if item is None:
                break
            outputs, datas = item
            states = one_image.run_batch(datas)
            self.result.put((outputs, states))
        self.result.put(None)
        #     for o, s in zip(outputs, states):
        #         self.clean_out(o, s)
        # with open('submission.json', 'w') as f:
        #     json.dump(self.submission, f)

    def post_process(self):
        submission = []
        while True:
            item = self.result.get()
            if item is None:
                break
            outputs, states = item
            for o, s in zip(outputs, states):
                sub_item = clean_out(o, s)
                submission.append(sub_item)
                logger.info(sub_item)
        with open('submission.json', 'w') as f:
            json.dump(submission, f)


p.join()
worker = Worker()
worker.run()