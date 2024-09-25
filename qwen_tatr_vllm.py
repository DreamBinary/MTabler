import json
import math
import multiprocessing
import os
import re
import warnings
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

opkgs_path = "/bohr/opkgs-k2wz/v2/opkgs"
tatr_path = "/bohr/tatr-xdh6/v2/tatr"
str_model_path = '/bohr/TATR-xmup/v1/TATR/TATR-v1.1-All-msft.pth'
str_config_path = '/bohr/TATR-xmup/v1/TATR/structure_config.json'
model_path = "/bohr/cach-rxl3/v17/cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f"
torch_hub_path = "/bohr/thub-w4uy/v1"
cache_path = "/bohr/cach-rxl3/v17/cache"

os.system(f"pip3 install {opkgs_path}/*")
os.system(f"cp -r {tatr_path} .")
# os.system(f"cp -r {raw_cache_path} .")
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["TORCH_HOME"] = torch_hub_path
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


batch_size = 8
submission = []
data = multiprocessing.Queue()


def ocr():
    from tatr import TableEngine
    engine = TableEngine(
        str_device=device,
        str_model_path=str_model_path,
        str_config_path=str_config_path
    )
    processor = AutoProcessor.from_pretrained(model_path)

    def create_msg(sys, path, q):
        return processor.apply_chat_template([
            {"role": "system",
             "content": sys},
            {"role": "user", "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": q}
            ]}
        ], tokenize=False, add_generation_prompt=True)


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
            data_t = list(json.load(f))[:100]
    q_prefix = "Based on the latex table, caption and html structure, "
    sys1 = "You are a helpful assistant."
    sys2 = "You are a helpful assistant. Provide only a label [A-H] of the correct answer for multiple-choice questions."
    sys3 = "You are a helpful assistant. Provide only a label [A-D] of the correct answer for multiple-choice questions."
    shapes = []
    batch_inputs = []
    for d in data_t:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        # w_path = os.path.join(NEW_IMG_DIR, d["image_path"])
        img = Image.open(r_path).convert("RGB")
        html, rows, cols = engine(img)
        q0 = f'This is a table image. The caption of the table is "{d["caption"]}". The structure of the table in html format is as follows: {html}.'
        q1 = f"{q0}{q_prefix}how many rows and columns are in the table? Provide only two positive integers for rows and columns, separated by a comma. It might be '{rows},{cols}', you can use it as a reference."
        q2 = f"""{q0}{q_prefix}which subject is most relevant to the table or caption?
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
        q3 = f"""{q0}{q_prefix}{question}
A) {d["options"][0]}
B) {d["options"][1]}
C) {d["options"][2]}
D) {d["options"][3]}
"""
        shapes.append((d["image_path"], rows, cols))
        img = fetch_image(img)
        batch_inputs.append({
            "prompt": create_msg(sys1, r_path, q1),
            "multi_modal_data": {
                "image": img
            }
        })
        batch_inputs.append({
            "prompt": create_msg(sys2, r_path, q2),
            "multi_modal_data": {
                "image": img
            }
        })
        batch_inputs.append({
            "prompt": create_msg(sys3, r_path, q3),
            "multi_modal_data": {
                "image": img
            }
        })
        if len(shapes) == batch_size:
            data.put((shapes, batch_inputs))
            shapes, batch_inputs = [], []
    if len(shapes) > 0:
        data.put((shapes, batch_inputs))
    data.put(None)


def process():
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=16,
        stop_token_ids=[],
    )
    while True:
        try:
            item = data.get(timeout=300)
            if item is None:
                break
        except:
            break
        shapes, batch_inputs = item
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
        category = ""
        answer = -1
        shape = ans[3 * i]
        subject = ans[3 * i + 1]
        option = ans[3 * i + 2]
        try:
            pattern = r'.*?(\d+).*?,.*?(\d+).*?'
            match = re.match(pattern, shape)
            if match:
                rows, cols = match.groups()
                rows = int(rows)
                cols = int(cols)
        except:
            pass
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
        submission.append(sub_item)


def fetch_image_path(img_path, size_factor: int = IMAGE_FACTOR) -> Image.Image:
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


def fetch_image(img, size_factor: int = IMAGE_FACTOR) -> Image.Image:
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