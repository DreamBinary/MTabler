import json

import multiprocessing
import os

from PIL import Image
from tqdm import tqdm
import logging
multiprocessing.log_to_stderr(logging.INFO)
logger = multiprocessing.get_logger()
logging.basicConfig(filename='sgl_unitable4.log', level=logging.INFO)

q_prefix = "Based on the table, caption and html structure, "

manager = multiprocessing.Manager()
data = manager.list()


def rewrite():
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data_t = json.load(f)
        data_t = list(data_t)[:50]
        # write path to json
    # new_data = []
    for d in tqdm(data_t):
        path = os.path.join(base_dir, "test_images", d["image_path"])
        image = Image.open(path).convert("RGB")
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
            "image": image,
            "q3": q3,
        })
    # save data to torch
    logger.info("Data prepared")
    # data_send.send(new_data)
    # data_send.close()
    # logger.info("Data sent")


p = multiprocessing.Process(target=rewrite)
p.start()

pkgs_path = "/bohr/pkgs-7x29/v18/pkgs"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"
unitable_model = "/bohr/unii-7sxm/v1/unitable/weights"
unitable_vocab = "/bohr/unii-7sxm/v1/unitable/vocab"
unitable_src = "/bohr/unii-7sxm/v1/unitable/src"

# os.system(f"pip install {pkgs_path}/* --ignore-installed")
os.system(f"cp -r {unitable_src} .")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
device = "cuda"

import torch.multiprocessing as multiprocessing
import re
from sglang.lang.chat_template import get_chat_template
import sglang as sgl
from sglang import Runtime
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple, Sequence, Optional, Union
import tokenizers as tk
import torch

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
device = torch.device("cuda:0")

# UniTable large model
d_model = 768
patch_size = 16
nhead = 12
dropout = 0.2


# %%
def autoregressive_decode(
        model: EncoderDecoder,
        image: Tensor,
        prefix: Sequence[int],
        max_decode_len: int,
        eos_id: int,
        token_whitelist: Optional[Sequence[int]] = None,
        token_blacklist: Optional[Sequence[int]] = None,
) -> Tensor:
    # model.eval()
    # with torch.no_grad():
    #     memory = model.encode(image)
    #     context = torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)

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
):
    backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
    encoder = Encoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
                      norm_first=True, nlayer=12, ff_ratio=4)
    decoder = Decoder(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu",
                      norm_first=True, nlayer=4, ff_ratio=4)
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
    model = model.to(device)
    model = torch.compile(model)
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
    try:
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
    except Exception:
        columns = 0
    return rows, columns


def unitable():
    vocab_html, model_html = load_vocab_and_model(
        vocab_path=VOCAB_HTML,
        max_seq_len=784,
        model_weights=MODEL_DIR / MODEL_FILE_NAME[0],

    )
    for idx, item in enumerate(tqdm(data)):
        image = item["image"]
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
        data[idx]["rows"] = rows
        data[idx]["cols"] = cols
        data[idx]["q1"] = \
            f'This is a table image. The caption of the table is "{item["caption"]}". The structure of the table in html format is as follows: {pred_html}.'


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


def clean_out(o, s):
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
        "image_path": o["image_path"],
        "category": category,
        "cols": o["cols"],
        "rows": o["rows"],
        "answer": answer,
    }
    # logger.info(sub_item)
    return sub_item


class Worker:
    def __init__(self):
        self.batch_size = 128
        self.result = multiprocessing.Queue()

    def run(self):
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
            disable_regex_jump_forward=True
        )
        runtime.endpoint.chat_template = get_chat_template("qwen")
        sgl.set_default_backend(runtime)

        post = multiprocessing.Process(target=self.post_process)
        post.start()
        self.process()
        runtime.shutdown()
        post.join()

    def process(self):
        l = len(data)
        for i in range(0, l, self.batch_size):
            logger.info(f"-->> Processing {i} to {i + self.batch_size}")
            batch = data[i:i + self.batch_size]
            inputs = [
                {"path": item["path"], "q1": item["q1"], "q3": item["q3"]} for item in batch
            ]
            states = one_image.run_batch(inputs)
            self.result.put((batch, states))
        self.result.put(None)

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
unitable()
torch.cuda.empty_cache()
worker = Worker()
worker.run()
