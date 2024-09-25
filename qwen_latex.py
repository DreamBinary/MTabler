import json
import math
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
model_path = "/bohr/cach-rxl3/v17/cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f"
cache_path = "/bohr/cach-rxl3/v17/cache"

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


def count_rows_cols(latex_code):
    try:
        # 查找列数：根据表格行的定义找到表格列标识符，如 |l|c|c|c|c|
        columns = re.search(r'\\begin\{tabular\}\{([^\}]+)\}', latex_code)
        if columns:
            num_cols = len([c for c in columns.group(1) if c.isalpha()])
        else:
            num_cols = 0

        # 查找行数：根据 \hline 分隔符统计表格的行数
        rows = latex_code.split(r'\hline')
        num_rows = sum(1 for row in rows if '&' in row or '\\rule' in row)

        return num_rows, num_cols
    except:
        return -1, -1


llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 1},
)
processor = AutoProcessor.from_pretrained(model_path)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1,
    repetition_penalty=1.05,
    max_tokens=4096,
    stop_token_ids=[],
)
batch_size = 16
latex_data = []
if os.environ.get('DATA_PATH_B'):
    base_dir = os.environ.get('DATA_PATH_B')
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data = json.load(f)
        # data_t = list(json.load(f))[:100]
else:
    base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data = list(json.load(f))[:10]


def generate_latex(inputs, cnt):
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    latex = [output.outputs[0].text for output in outputs]
    for ii, l in enumerate(latex):
        # print(l)
        rows, cols = count_rows_cols(l)
        idx = ii + cnt * batch_size
        data[idx]["latex"] = l
        data[idx]["rows"] = rows
        data[idx]["cols"] = cols


def preprocess():
    prompt = processor.apply_chat_template([
        {"role": "system",
         "content": "You are a helpful assistant. Provide only latex code for the table in the image."},
        {"role": "user", "content": [
            {"type": "image", "image": ""},
            {"type": "text", "text": "Convert the table in the image to latex code."}
        ]}
    ], tokenize=False, add_generation_prompt=True)
    inputs = []
    cnt = 0
    for d in data:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        img = fetch_image_path(r_path)
        input = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": img
            }
        }
        inputs.append(input)
        if len(inputs) == batch_size:
            generate_latex(inputs, cnt)
            cnt += 1
            inputs = []
    if len(inputs) > 0:
        generate_latex(inputs, cnt)


def process():
    def create_msg(sys, q):
        return processor.apply_chat_template([
            {"role": "system",
             "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": q}
            ]}
        ], tokenize=False, add_generation_prompt=True)

    q_prefix = "Based on the latex table, "
    sys2 = "You are a helpful assistant. Provide only a label [A-H] of the correct answer for multiple-choice questions."
    sys3 = "You are a helpful assistant. Provide only a label [A-D] of the correct answer for multiple-choice questions."
    inputs = []
    cnt = 0
    for d in data:
        q0 = f'"{d["latex"]}" This is latex code for a table. The caption of the table is "{d["caption"]}".'
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
        inputs.append({
            "prompt": create_msg(sys2, q2),
        })
        inputs.append({
            "prompt": create_msg(sys3, q3),
        })
        # print(inputs)
        if len(inputs) == 2 * batch_size:
            generate_ans(inputs, cnt)
            cnt += 1
            inputs = []
    if len(inputs) > 0:
        generate_ans(inputs, cnt)


def generate_ans(inputs, cnt):
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    l = len(outputs) // 2
    for i in range(l):
        idx = cnt * batch_size + i
        data[idx]["subject"] = outputs[2 * i]
        data[idx]["option"] = outputs[2 * i + 1]


def postprocess():
    submission = []
    for d in data:
        subject = d["subject"]
        option = d["option"]
        category = ""
        answer = -1
        try:
            match = re.search(r'[A-Za-z]', subject)
            if match:
                category = match.group(0).upper()
                category = sub_list[l2i[category]]
        except:
            pass
        try:
            match = re.search(r'[A-Za-z]', option)
            if match:
                answer = match.group(0).upper()
                answer = l2i[answer]
        except:
            pass

        sub_item = {
            "image_path": d["image_path"],
            "category": category,
            "cols": d["cols"],
            "rows": d["rows"],
            "answer": answer,
        }
        # print(sub_item)
        submission.append(sub_item)
    if len(submission) != 5360:
        with open('error.json', 'w') as f:
            json.dump(submission, f)
        raise Exception(f"Submission length is {len(submission)}")
    with open('submission.json', 'w') as f:
        json.dump(submission, f)


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


preprocess()
process()
postprocess()
