import json
import math
import os
import re
import warnings
from collections import defaultdict

import torch
from PIL import Image
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
qwen_path = "/bohr/cach-rxl3/v17/cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f"
llama_path = "/bohr/cach-rxl3/v15/cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f"
# os.system(f"cp -r {raw_cache_path} .")
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
        rows = latex_code.split(r'\\')
        num_rows = sum(1 for row in rows if '&' in row or '\\rule' in row)

        return num_rows, num_cols
    except:
        return -1, -1


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


def generate_latex(llm, sampling_params, inputs, cnt):
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
    llm = LLM(
        model=qwen_path,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=4096,
        stop_token_ids=[],
    )
    # processor = AutoProcessor.from_pretrained(qwen_path)
    # prompt = processor.apply_chat_template([
    #     {"role": "system",
    #      "content": "You are a helpful assistant. Provide only latex code for the table in the image."},
    #     {"role": "user", "content": [
    #         {"type": "image", "image": ""},
    #         {"type": "text", "text": "Convert the table in the image to latex code."}
    #     ]}
    # ], tokenize=False, add_generation_prompt=True)
    # # print("---------------------->> prompt")
    # # print(prompt)
    prompt = """<|im_start|>system
You are a helpful assistant. Provide only latex code for the table in the image.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Convert the table in the image to latex code.<|im_end|>
<|im_start|>assistant"""
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
            generate_latex(llm, sampling_params, inputs, cnt)
            cnt += 1
            inputs = []
    if len(inputs) > 0:
        generate_latex(llm, sampling_params, inputs, cnt)


def process():
    llm = LLM(model=llama_path)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=2,
        stop_token_ids=[],
    )

    #     processor = AutoProcessor.from_pretrained(llama_path)
    #
    #     """
    #     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # Cutting Knowledge Date: December 2023
    # Today Date: 26 Jul 2024
    # sys2<|eot_id|><|start_header_id|>user<|end_header_id|>
    # q2<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #     """

    q_prefix = "Based on the latex table, "
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
Provide only the label [A-H] of the best answer for multiple-choice question without extra explanation (just a letter).
"""
        question = d["question"]
        question = question[0].lower() + question[1:]
        q3 = f"""{q0}{q_prefix}{question}
A) {d["options"][0]}
B) {d["options"][1]}
C) {d["options"][2]}
D) {d["options"][3]}
Provide only the label [A-D] of the correct best for multiple-choice question without extra explanation (just a letter).
"""
        inputs.append(q2)
        inputs.append(q3)
        if len(inputs) == 2 * batch_size:
            generate_ans(llm, sampling_params, inputs, cnt)
            cnt += 1
            inputs = []
    if len(inputs) > 0:
        generate_ans(llm, sampling_params, inputs, cnt)


def generate_ans(llm, sampling_params, inputs, cnt):
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    outputs = [output.outputs[0].text for output in outputs]
    l = len(outputs) // 2
    # print("-->> ans")
    # print(outputs)
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
