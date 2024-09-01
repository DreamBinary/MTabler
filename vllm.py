# config env
pkgs_path = "/bohr/pkgs-7x29/v5/pkgs"
llava_lib_path = "/bohr/libb-bg5b/v3/llava"
tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"

help_model_path = "OpenGVLab/InternVL2-2B"
main_model_path = "Qwen/Qwen2-VL-7B-Instruct"
cache_path = "/bohr/cach-rxl3/v7/cache"

# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"

import os

# # 提交时可能不能联网，设置成离线模式防止联网失败报错
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
device = "cuda"

import warnings

warnings.filterwarnings("ignore")

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import threading
import queue
from qwen_vl_utils import process_vision_info
from llava.conversation import Conversation, SeparatorStyle
from llava.utils import disable_torch_init
import json
from llava.constants import DEFAULT_IMAGE_TOKEN
import torch

from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from collections import defaultdict
import re

from vllm import LLM, SamplingParams, TextPrompt
from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions

args = type('Args', (), {
    "conv_mode": None,
    "sep": ",",
    "temperature": 0,
    "top_p": 1,
    "num_beams": 1,
    "max_new_tokens": 8
})()

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')

torch.cuda.empty_cache()
disable_torch_init()

if os.environ.get('DATA_PATH_B'):  # 提交时会选择隐藏的测试数据集路径（A+B榜），数据集的格式与A榜数据相同，但数目不同（5360张）
    base_dir = os.environ.get('DATA_PATH_B')
else:
    base_dir = '/bohr/form-recognition-train-b6y2/v4'  # 示例，把A榜测试数据集路径作为测试集路径，仅开发时挂载A榜数据用于debug   # 示例，把A榜测试数据集路径作为测试集路径，仅开发时挂载A榜数据用于debug


def clean_out(image_path, out_list):
    matches = re.findall(r"\d+", out_list[0])
    if len(matches) >= 2:
        rows, cols = int(matches[0]), int(matches[1])
    elif len(matches) == 1:
        rows = cols = int(matches[0])
    else:
        rows = cols = -1

    sub_item = {
        "image_path": image_path,
        "category": sub_list[l2i[out_list[1][0]]],
        "cols": cols,
        "rows": rows,
        "answer": l2i[out_list[2][0]],
    }
    return sub_item


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class Worker:
    def __init__(self):
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            self.data = json.load(f)
        self.tsr_result = []
        self.help_result = []
        self.main_input = queue.Queue()

        # self.help_model = AutoModel.from_pretrained(
        #     help_model_path,
        #     torch_dtype=torch.bfloat16,
        #     load_in_8bit=True,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True).eval()
        #
        # self.help_tokenizer = AutoTokenizer.from_pretrained(help_model_path, trust_remote_code=True, use_fast=False)

        # self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
        #     main_model_path, None, "llava_qwen", device_map="auto",
        #     attn_implementation='sdpa',
        #     # load_8bit=True,
        #     # load_4bit=False,
        #     **{
        #         "multimodal": True,
        #         "overwrite_config": {
        #             "image_aspect_ratio": "anyres_max_9"
        #         }
        #     }
        # )

        self.tsr_img_processor = AutoImageProcessor.from_pretrained(tsr_model_path)
        self.tsr_img_processor.size = {'height': 384, 'width': 384}
        self.tsr_model = TableTransformerForObjectDetection.from_pretrained(tsr_model_path)
        label2id = self.tsr_model.config.label2id
        self.label_row = label2id['table row']
        self.label_col = label2id['table column']

        self.llm = LLM(model=main_model_path)

    def run(self):
        tasks = [
            self.tsr_process,
            self.help_process
        ]
        threads = [threading.Thread(target=task) for task in tasks]
        for thread in threads:
            thread.start()
        self.main_process()
        for thread in threads:
            thread.join()

    def tsr_process(self):
        for item in self.data:
            path = os.path.join(base_dir, 'test_images', item["image_path"])
            image = Image.open(path).convert("RGB")
            inputs = self.tsr_img_processor(images=image, return_tensors="pt")
            outputs = self.tsr_model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])  # (height, width) of each image in the batch
            results = \
                self.tsr_img_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[
                    0]
            draw = ImageDraw.Draw(image)
            rows = 0
            cols = 0
            for label, box in zip(results["labels"], results["boxes"]):
                label, box = label.item(), box.tolist()
                draw.rectangle(box, outline="red", width=1)
                if label == self.label_row:
                    rows += 1
                elif label == self.label_col:
                    cols += 1
            if self.help_result:
                self.main_input.put((self.help_result.pop(0), (image, rows, cols)))
            else:
                self.tsr_result.append((image, rows, cols))
            # print("TSR", rows, cols)
            # print("-->> len_help", len(self.help_result))
        if not self.help_result:
            self.main_input.put(None)

    def help_process(self):
        generation_config = dict(max_new_tokens=64, do_sample=False)
        for item in self.data:
            path = os.path.join(base_dir, 'test_images', item["image_path"])
            caption = item["caption"]
            pixel_values = load_image(path, max_num=12).to(torch.bfloat16).cuda()
            qs_list = [
                f'Based on the provided table, what is its shape? Answer with two positive integers for rows and columns, separated by a comma:',
                f"""Based on the provided table and caption, select the most relevant subject from (A. Physics, B. Mathematics, C. ComputerScience, D. QuantitativeBiology, E. QuantitativeFinance, F. Statistics, G. ElectricalEngineeringandSystemsScience, H. Economics). Answer with the option's letter from the given choices directly.""",
                f"""Based on the provided table and caption, for the question: "{item["question"]}", select the most correct option from (A. {item["options"][0]}, B. {item["options"][1]}, C. {item["options"][2]}, D. {item["options"][3]}). Answer with the option's letter from the given choices directly."""
            ]
            history = [
                (f'<image>\n This is a table image. The caption of the table is "{caption}".',
                 'I have a general understanding of the information in this table.')
            ]
            out_list = []
            for question in qs_list:
                response, history = self.help_model.chat(self.help_tokenizer, pixel_values, question, generation_config,
                                                         history=history, return_history=True)
                out_list.append(response)

            # print("HELP:", out_list)
            # print("-->> len_tsr:", len(self.tsr_result))
            if self.tsr_result:
                self.main_input.put(((item["image_path"], caption, qs_list, out_list), self.tsr_result.pop(0)))
            else:
                self.help_result.append((item["image_path"], caption, qs_list, out_list))
        if not self.tsr_result:
            self.main_input.put(None)

    def main_process(self):
        submission = []
        while True:
            item = self.main_input.get()
            if item is None:
                break
            size = self.main_input.qsize()
            items = [item] + [self.main_input.get() for _ in range(size)]
            img_paths, captions, qs_lists, out_lists, images, rows, cols = zip(*[
                (image_path, caption, qs_list, out_list, image, row, col)
                for (image_path, caption, qs_list, out_list), (image, row, col) in items
            ])
            size += 1

            out_lists = self.vllm_images(images, convs, qs_lists)
            sub_items = [clean_out(img_path, out_list) for img_path, out_list in zip(img_paths, out_lists)]
            # print("MAIN:", out_list)
            submission.extend(sub_items)
        with open('submission.json', 'w') as f:
            json.dump(submission, f)

    def vllm_images(self, image_list, conv_list, qs_list):
        num = len(image_list)
        ans_list = [[], [], []]
        sampling_params = SamplingParams(temperature=0.2, max_tokens=64, stop_token_ids=None),
        guided_request = [
            LLMGuidedOptions(guided_regex="^\d,\s*\d$"),
            LLMGuidedOptions(guided_choice=["A.Physics", "B.Mathematics", "C.ComputerScience", "D.QuantitativeBiology",
                                            "E.QuantitativeFinance", "F.Statistics",
                                            "G.ElectricalEngineeringandSystemsScience", "H.Economics"]),
            LLMGuidedOptions(guided_regex="", guided_choice=["A", "B", "C", "D"]),
        ]
        for q_idx in range(3):
            prompts = []
            for i in range(num):
                q = qs_list[i][q_idx]
                conv_list[i].append_message(conv_list[i].roles[0], q)
                conv_list[i].append_message(conv_list[i].roles[1], None)
                prompt = conv_list[i].get_prompt()
                prompts.append(TextPrompt(prompt=prompt, multi_modal_data={"image": image_list[i]}))

            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True,
                                        guided_options_request=guided_request[q_idx])

            for i in range(num):
                o = outputs[i]
                text = o.outputs[0].text
                ans_list[q_idx].append(text)
                conv_list[i].messages[-1][-1] = text

        return list(zip(*ans_list))


worker = Worker()
worker.run()
