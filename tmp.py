import itertools
import json
import math
import multiprocessing
import os
import random
import re
import warnings
from collections import defaultdict

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.guided_decoding import GuidedDecodingRequest

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
label = "ABCDEFGH"
for i, letter in enumerate(label):
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


option2 = [
    "Physics",
    "Mathematics",
    "Computer Science",
    "Quantitative Biology",
    "Quantitative Finance",
    "Statistics",
    "Electrical Engineering and Systems Science",
    "Economics",
]

if os.environ.get('DATA_PATH_B'):
    base_dir = os.environ.get('DATA_PATH_B')
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        raw_data = json.load(f)
        # data_t = list(json.load(f))[:100]
else:
    base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        raw_data = list(json.load(f))[:10]
length = len(raw_data)
ocr_data = multiprocessing.Manager().list()
batch_size = 16
orders4 = list(itertools.permutations([0, 1, 2, 3]))
orders8 = list(itertools.permutations([0, 1, 2, 3, 4, 5, 6, 7]))
tmp_ans2 = defaultdict(lambda: defaultdict(int))
tmp_ans3 = defaultdict(lambda: defaultdict(int))
ordered2 = defaultdict(list)
ordered3 = defaultdict(list)
final_ans2 = [-1] * length
final_ans3 = [-1] * length
placeholder = "%%pl_ac_eh_+=ol_&der%%"


def shuffle3(idx, arr):
    while True:
        ri = random.randint(0, 23)
        if ri not in ordered3[idx]:
            ordered3[idx].append(ri)
            order = orders4[ri]
            shuffled = [f"{label[i]}) {arr[j]}" for i, j in enumerate(order)]
            shuffled = "\n".join(shuffled)
            break
    return shuffled, order


def shuffle2(idx):
    while True:
        ri = random.randint(0, 40319)
        if ri not in ordered2[idx]:
            ordered2[idx].append(ri)
            order = orders8[ri]
            shuffled = [f"{label[i]}) {option2[j]}" for i, j in enumerate(order)]
            shuffled = "\n".join(shuffled)
            break
    return shuffled, order


def gen_inputs2(idxs):
    inputs = []
    orders = []
    for i in idxs:
        option, order = shuffle2(i)
        q2 = ocr_data[i]["q2"].replace(placeholder, option)
        inputs.append({
            "prompt": q2,
            "multi_modal_data": {
                "image": ocr_data[i]["img"]
            }
        })
        orders.append(order)
    orders.append(-1)
    return inputs, orders


def gen_inputs3(idxs):
    inputs = []
    orders = []
    for i in idxs:
        option, order = shuffle3(i, raw_data[i]["options"])
        q3 = ocr_data[i]["q3"].replace(placeholder, option)
        inputs.append({
            "prompt": q3,
            "multi_modal_data": {
                "image": ocr_data[i]["img"]
            }
        })
        orders.append(order)
    orders.append(-1)
    return inputs, orders


def process():
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=4,
        stop_token_ids=[],
    )
    guided_options_request2 = GuidedDecodingRequest(
        guided_regex=r"[A-Ha-h]",
    )
    guided_options_request3 = GuidedDecodingRequest(
        guided_regex=r"[A-Da-d]",
    )

    idx2, idx3 = 0, 0
    batch_idx2, batch_idx3 = [], []
    while True:
        if len(batch_idx2) == 0 and len(batch_idx3) == 0 and idx2 >= length and idx3 >= length:
            break
        while len(batch_idx2) < batch_size and idx2 < length:
            batch_idx2.append(idx2)
            idx2 += 1
        while len(batch_idx3) < batch_size and idx3 < length:
            batch_idx3.append(idx3)
            idx3 += 1
        inputs2, orders2 = gen_inputs2(batch_idx2)
        inputs3, orders3 = gen_inputs3(batch_idx3)

        outputs2 = llm.generate(
            inputs2,
            sampling_params=sampling_params,
            use_tqdm=False,
            guided_options_request=guided_options_request2
        )
        outputs3 = llm.generate(
            inputs3,
            sampling_params=sampling_params,
            use_tqdm=False,
            guided_options_request=guided_options_request3
        )
        ans2 = [clean_ans(output.outputs[0].text) for output in outputs2]
        ans3 = [clean_ans(output.outputs[0].text) for output in outputs3]

        t2, t3 = [], []
        for i, a, o in zip(batch_idx2, ans2, orders2):
            real_ans = o[a]
            if tmp_ans2[i][real_ans] == 3:
                final_ans2[i] = real_ans
                t2.append(i)
            else:
                tmp_ans2[i][real_ans] += 1
        for i, a, o in zip(batch_idx3, ans3, orders3):
            real_ans = o[a]
            if tmp_ans3[i][real_ans] == 3:
                final_ans3[i] = real_ans
                t3.append(i)
            else:
                tmp_ans3[i][real_ans] += 1
        for i in t2:
            batch_idx2.remove(i)
        for i in t3:
            batch_idx3.remove(i)


def clean_ans(ans):
    try:
        match = re.search(r'[A-Ha-h]', ans)
        if match:
            return l2i[match.group(0).upper()]
    except:
        return -1
    return -1


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


def ocr():
    from tatr import TableEngine
    structure_class_thresholds = {
        "table": 0.5,
        "table column": 0.5,
        "table row": 0.5,
        "table column header": 0.5,
        "table projected row header": 0.5,
        "table spanning cell": 0.5,
        "no object": 10
    }
    engine = TableEngine(
        str_device=device,
        str_model_path=str_model_path,
        str_config_path=str_config_path,
        str_class_thresholds=structure_class_thresholds
    )

    template = """<|im_start|>system
{sys}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>
<|im_start|>assistant"""

    q_prefix = "Based on the table, caption and html structure, "
    sys2 = "You are a helpful assistant. Provide only a label [A-H] of the correct answer for multiple-choice questions."
    sys3 = "You are a helpful assistant. Provide only a label [A-D] of the correct answer for multiple-choice questions."
    for d in raw_data:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        img = Image.open(r_path).convert("RGB")
        html, rows, cols = engine(img)
        q1 = f'This is a table image. The caption of the table is "{d["caption"]}". The structure of the table in html format is as follows: {html}.'
        q2 = f"""{q1}{q_prefix}which subject is most relevant to the table or caption?\n{placeholder}"""
        question = d["question"]
        question = question[0].lower() + question[1:]
        q3 = f"""{q1}{q_prefix}{question}\n{placeholder}"""
        q2 = template.format(sys=sys2, q=q2)
        q3 = template.format(sys=sys3, q=q3)
        ocr_data.append({
            "rows": rows,
            "cols": cols,
            "img": fetch_image(img),
            "q2": q2,
            "q3": q3,
        })


def postprocess():
    submission = []
    for i in range(length):
        image_path = raw_data[i]["image_path"]
        rows = ocr_data[i]["rows"]
        cols = ocr_data[i]["cols"]
        category = sub_list[final_ans2[i]]
        answer = final_ans3[i]
        submission.append({
            "image_path": image_path,
            "category": category,
            "cols": cols,
            "rows": rows,
            "answer": answer,
        })
    if len(submission) != 5360:
        with open('error.json', 'w') as f:
            json.dump(submission, f)
        raise Exception(f"Submission length is {len(submission)}")
    with open('submission.json', 'w') as f:
        json.dump(submission, f)


ocr_process = multiprocessing.Process(target=ocr)
ocr_process.start()
process()
postprocess()
