q_prefix = "Based on the table, caption and html structure, "


def rewrite():
    import os
    import json
    NEW_IMG_DIR = "new_images"
    os.makedirs(NEW_IMG_DIR, exist_ok=True)
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            data_t = json.load(f)
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            data_t = list(json.load(f))[:10]
    data = []
    for d in data_t:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        w_path = os.path.join(NEW_IMG_DIR, d["image_path"])
        question = d["question"]
        question = question[0].lower() + question[1:]
        q3 = f"{q_prefix}{question}"
        data.append({
            "r_path": r_path,
            "w_path": w_path,
            "image_path": d["image_path"],
            "caption": d["caption"],
            "q3": q3,
            "options": [
                f'A) {d["options"][0]}',
                f'B) {d["options"][1]}',
                f'C) {d["options"][2]}',
                f'D) {d["options"][3]}'
            ]
        })

    with open('data.json', 'w') as f:
        json.dump(data, f)


import multiprocessing
import os

# multiprocessing.log_to_stderr(logging.INFO)
# logger = multiprocessing.get_logger()
# logging.basicConfig(filename='sgl_unitable4.log', level=logging.INFO)

p = multiprocessing.Process(target=rewrite)
p.start()

pkgs_path = "/bohr/pkgs-7x29/v18/pkgs"
opkgs_path = "/bohr/opkgs-k2wz/v1/opkgs"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"
torch_hub_path = "/bohr/thub-w4uy/v1"
tatr_path = "/bohr/tatr-xdh6/v1/tatr"
str_model_path = '/bohr/TATR-xmup/v1/TATR/TATR-v1.1-All-msft.pth'
str_config_path = '/bohr/TATR-xmup/v1/TATR/structure_config.json'
os.system("pip uninstall psutil -y")
os.system(f"pip3 install {pkgs_path}/* --ignore-installed")
os.system(f"pip3 install {opkgs_path}/*")
os.system(f"cp -r {tatr_path} .")
# # 提交时可能不能联网，设置成离线模式防止联网失败报错
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
os.environ["TORCH_HOME"] = torch_hub_path
device = "cuda"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from collections import defaultdict
from sglang.lang.chat_template import get_chat_template
from PIL import Image
import json
import sys
from sglang import Runtime
import warnings
import sglang as sgl
import torch
import multiprocessing
import re
from tatr import TableEngine

warnings.filterwarnings("ignore")

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()

q2 = f"""{q_prefix}which subject is most relevant to the table or caption?"""


@sgl.function
def one_image(s, path, q1, q3, options):
    s += sgl.system(
        "You are a helpful assistant. Provide only an label ([A-H] or [A-D]) of the correct answer for multiple-choice questions.")
    s += sgl.user(sgl.image(path) + q1)
    s += sgl.assistant("I have a general understanding of the information in this table.")
    s += sgl.user(q2)
    s += sgl.assistant(
        sgl.select(
            "subject",
            choices=[
                "A) Physics",
                "B) Mathematics",
                "C) Computer Science",
                "D) Quantitative Biology",
                "E) Quantitative Finance",
                "F) Statistics",
                "G) Electrical Engineering and Systems Science",
                "H) Economics",
            ],
            temperature=0.0
        )
    )
    s += sgl.user(q3)
    s += sgl.assistant(
        sgl.select(
            "option",
            choices=options,
            temperature=0.0
        )
    )


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
    return sub_item


class Worker:
    def __init__(self):
        self.batch_size = 8
        self.ocr_data = multiprocessing.Queue()

    def run(self):
        ocr_process = multiprocessing.Process(target=self.ocr)
        ocr_process.start()

        model_overide_args = {
            "attn_implementation": "eager",
            "multimodal": True,
            "overwrite_config": {
                "image_aspect_ratio": "anyres_max_9"
            },
        }
        runtime = Runtime(
            model_path=model_path,
            model_overide_args=model_overide_args,
            # disable_regex_jump_forward=True,
            # # enable_mixed_chunk=True,
            # triton_attention_reduce_in_fp32=True,
        )
        runtime.endpoint.chat_template = get_chat_template("qwen")
        sgl.set_default_backend(runtime)

        # post = multiprocessing.Process(target=self.post_process)
        # post.start()

        self.process()
        runtime.shutdown()
        # post.join()

    def ocr(self):
        engine = TableEngine(
            str_device=device,
            str_model_path=str_model_path,
            str_config_path=str_config_path
        )
        outputs = []
        inputs = []
        with open('data.json', 'r') as f:
            data = json.load(f)
        for item in data:
            img = Image.open(item["r_path"])
            html, rows, cols = engine(img, item["w_path"], tokens=[])
            q1 = f'This is a table image. The caption of the table is "{item["caption"]}". The structure of the table in html format is as follows: {html}.'
            outputs.append((item["image_path"], rows, cols))
            inputs.append({"path": item["r_path"], "q1": q1, "q3": item["q3"], "options": item["options"]})
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
            try:
                item = self.ocr_data.get(timeout=300)
                if item is None:
                    break
            except:
                break

            outputs, inputs = item
            states = one_image.run_batch(inputs)
            for o, s in zip(outputs, states):
                sub_item = clean_out(o, s)
                # logger.info(sub_item)
                submission.append(sub_item)
        if len(submission) != 5360:
            sys.exit(f"Submission length is {len(submission)}")
        with open('submission.json', 'w') as f:
            json.dump(submission, f)
        sys.exit(0)


p.join()
worker = Worker()
worker.run()
