import os

model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
gotp_path = "/bohr/gotp-adm2/v1/GOT"
gotw_path = "/bohr/gott-117w/v1/GOT_weights"
cache_path = "/bohr/cach-rxl3/v15/cache"

os.system(f"cp -r {gotp_path} .")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
device = "cuda"

import json
import re
import warnings
from collections import defaultdict
import sglang as sgl
import torch
from PIL import Image
import multiprocessing
from sglang import Runtime
from transformers import AutoTokenizer
from GOT.model import *
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import SeparatorStyle, Conversation
from GOT.utils.utils import KeywordsStoppingCriteria

warnings.filterwarnings("ignore")

# import logging
#
# multiprocessing.log_to_stderr(logging.INFO)
# logger = multiprocessing.get_logger()
# logging.basicConfig(filename='log.log', level=logging.INFO)

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


class GOT:
    def __init__(self, model_name):
        model_name = os.path.expanduser(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = GOTQwenForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            device_map='cuda',
            use_safetensors=True,
            pad_token_id=151643
        ).eval()
        self.model.to(device='cuda', dtype=torch.bfloat16)
        self.image_processor = BlipImageEvalProcessor(image_size=1024)
        self.image_processor_high = BlipImageEvalProcessor(image_size=1024)
        image_token_len = 256
        qs = 'OCR with format: '
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
        conv = Conversation(
            system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
            roles=["<|im_start|>user\n", "<|im_start|>assistant\n"],
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [self.stop_str]

        inputs = self.tokenizer([prompt])
        self.input_ids = torch.as_tensor(inputs.input_ids).cuda()
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

    def __call__(self, image):
        image_1 = image.copy()
        image_tensor = self.image_processor(image)
        image_tensor_1 = self.image_processor_high(image_1)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                self.input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                # streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[self.stopping_criteria]
            )

            outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()

            if outputs.endswith(self.stop_str):
                outputs = outputs[:-len(self.stop_str)]
            outputs = outputs.strip()
        rows, cols = self.count_rows_cols(outputs)
        return outputs, rows, cols

    def count_rows_cols(self, latex_code):
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


data = multiprocessing.Queue()


def rewrite():
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
    q_prefix = "Based on the latex table and caption, "
    engine = GOT(gotw_path)
    for d in data_t:
        r_path = os.path.join(base_dir, "test_images", d["image_path"])
        # w_path = os.path.join(NEW_IMG_DIR, d["image_path"])

        img = Image.open(r_path).convert('RGB')
        latex, rows, cols = engine(img)
        q2 = f""""{latex}" This is the latex code of the table with caption "{d["caption"]}". {q_prefix}which subject is most relevant to the table or caption?
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
        q3 = f"""{q_prefix}{question}
A) {d["options"][0]}
B) {d["options"][1]}
C) {d["options"][2]}
D) {d["options"][3]}
"""
        data.put({
            "image_path": d["image_path"],
            "q2": q2,
            "q3": q3,
            "rows": rows,
            "cols": cols
        })
    data.put(None)


@sgl.function
def one_image(s, q2, q3):
    s += sgl.system(
        "You are a helpful assistant. Provide only an label ([A-H] or [A-D]) of the correct answer for multiple-choice questions."
    )
    s += sgl.user(q2)
    s += sgl.assistant(
        "subject label: " + sgl.gen_string(
            "subject",
            # choices=["A", "B", "C", "D", "E", "F", "G", "H"],
            max_tokens=2, temperature=0.0, top_p=1
        )
    )
    s += sgl.user(q3)
    s += sgl.assistant(
        sgl.gen_string(
            "option",
            # choices=["A", "B", "C", "D"],
            max_tokens=2, temperature=0.0, top_p=1
        )
    )


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
    return sub_item


def run():
    runtime = Runtime(
        model_path=model_path,
        mem_fraction_static=0.85
        # model_overide_args=model_overide_args,
        # disable_regex_jump_forward=True,
        # # enable_mixed_chunk=True,
        # triton_attention_reduce_in_fp32=True,
    )
    sgl.set_default_backend(runtime)
    submission = []
    while True:
        try:
            item = data.get(timeout=300)
            if item is None:
                break
        except:
            break
        size = min(data.qsize(), 8)
        items = [item] + [data.get() for _ in range(size)]
        inputs = [{"q2": item["q2"], "q3": item["q3"]} for item in items]
        states = one_image.run_batch(inputs)
        for o, s in zip(items, states):
            sub_item = clean_out(o, s)
            # logger.info(sub_item)
            submission.append(sub_item)
    if len(submission) != 5360:
        import sys
        sys.exit(f"Submission length is {len(submission)}")
    with open('submission.json', 'w') as f:
        json.dump(submission, f)
    runtime.shutdown()


p = multiprocessing.Process(target=rewrite)
p.start()
run()
