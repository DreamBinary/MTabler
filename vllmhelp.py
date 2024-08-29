
# config env
pkgs_path = "/bohr/pkgs-7x29/v13/pkgs"
llava_lib_path = "/bohr/libb-bg5b/v3/llava"
tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"

help_model_path = "OpenGVLab/InternVL2-2B"
main_model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v7/cache"

# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"

import os

os.system(f"pip install --no-index --find-links={pkgs_path}  --ignore-installed {pkgs_path}/*")
os.system(f"cp {llava_lib_path} . -r")

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
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

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
import multiprocessing
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

# class Generator(SequenceGeneratorAdapter):
#     def __init__(self, model):
#         super().__init__(model, None, multinomial())
# 
#     def generate(
#             self,
#             prompts: Union[str, List[str]],
#             logits_processor: FSMLogitsProcessor,
#             max_tokens: Optional[int] = None,
#             stop_at: Optional[Union[str, List[str]]] = None,
#             seed: Optional[int] = None,
#             **model_specific_params
#     ):
#         generation_params = self.prepare_generation_parameters(
#             max_tokens, stop_at, seed
#         )
# 
#         completions = self.model.generate(
#             prompts,
#             generation_params,
#             logits_processor,
#             self.sampling_params,
#             **model_specific_params,
#         )
# 
#         return self._format(completions)

class Worker:
    def __init__(self):
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            self.data = json.load(f)
            # self.data = list(json.load(f))[:2]
        # manager = multiprocessing.Manager()
        self.tsr_result = multiprocessing.Queue()
        # self.help_result = manager.list()
        self.main_input = multiprocessing.Queue()

    def run(self):
        # multiprocessing.set_start_method('spawn')
        help_process = multiprocessing.Process(target=self.help_process, daemon=True)
        tsr_process = multiprocessing.Process(target=self.tsr_process, daemon=True)
        help_process.start()
        tsr_process.start()
        self.main_process()
        # tsr_process.join()
        # help_process.join()
        # print("RUN END")

    def tsr_process(self):
        tsr_img_processor = AutoImageProcessor.from_pretrained(tsr_model_path)
        tsr_img_processor.size = {'height': 384, 'width': 384}
        tsr_model = TableTransformerForObjectDetection.from_pretrained(tsr_model_path)
        label2id = tsr_model.config.label2id
        label_row = label2id['table row']
        label_col = label2id['table column']
        for item in self.data:
            path = os.path.join(base_dir, 'test_images', item["image_path"])
            image = Image.open(path).convert("RGB")
            inputs = tsr_img_processor(images=image, return_tensors="pt")
            outputs = tsr_model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])  # (height, width) of each image in the batch
            results = \
                tsr_img_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
            draw = ImageDraw.Draw(image)
            rows = 0
            cols = 0
            for label, box in zip(results["labels"], results["boxes"]):
                label, box = label.item(), box.tolist()
                draw.rectangle(box, outline="red", width=1)
                if label == label_row:
                    rows += 1
                elif label == label_col:
                    cols += 1
            self.tsr_result.put((image, rows, cols, item))
            # if self.help_result:
            #     self.main_input.put((self.help_result.pop(0), (image, rows, cols)))
            # else:
            #     self.tsr_result.append((image, rows, cols))
            print("TSR", rows, cols)
            # print("-->> len_help", len(self.help_result))
        # if not self.help_result and not self.tsr_result:
        #     self.main_input.put(None)
        self.tsr_result.put(None)

    def help_process(self):
        try:
            model = LLM(
                model=help_model_path,
                trust_remote_code=True,
                dtype="float16",
                max_model_len=2048,
                enforce_eager=True,
                gpu_memory_utilization=0.5
            )
            tokenizer = model.get_tokenizer()
            stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
            stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
            flag = True
            while flag:
                tsr_item = self.tsr_result.get()
                if tsr_item is None:
                    break
                size = self.tsr_result.qsize()
                size = min(size, 4)
                tsr_items = [tsr_item] + [self.tsr_result.get() for _ in range(size)]
                if tsr_items[-1] is None:
                    tsr_items = tsr_items[:-1]
                    flag = False
                images, rows, cols, items = map(list, zip(*tsr_items))
                size = len(tsr_items)
                qs_list = [
                    [
                        f'Based on the provided table, what is its shape? Answer with two positive integers for rows and columns.',
                        f"""Based on the provided table and caption, select the most relevant subject from (A. Physics, B. Mathematics, C. ComputerScience, D. QuantitativeBiology, E. QuantitativeFinance, F. Statistics, G. ElectricalEngineeringandSystemsScience, H. Economics).""",
                        f"""Based on the provided table and caption, for the question: "{item["question"]}", select the most correct option from (A. {item["options"][0]}, B. {item["options"][1]}, C. {item["options"][2]}, D. {item["options"][3]}). Answer with the option's letter from the given choices directly."""
                    ] for item in items
                ]
                prefix = [
                    [
                        f'<image>\n This is a table image with red borders. The table shape might be ({rows[i]}, {cols[i]}) but could vary.',
                        f'<image>\n This is a table image with red borders. The caption of the table is "{items[i]["caption"]}"',
                        f'<image>\n This is a table image with red borders. The caption of the table is "{items[i]["caption"]}"',
                    ] for i in range(size)
                ]
                # messages = [
                #     [
                #         {'role': 'system',
                #          'content': "You are a helpful assistant. Provide only an option's letter or an integer for each question, without any additional explanation."},
                #         {'role': 'user',
                #          'content': f'<image>\n This is a table image with red borders. The table shape might be ({rows[i]}, {cols[i]}) but could vary. The caption of the table is "{items[i]["caption"]}".'},
                #         {'role': 'assistant',
                #          'content': "I have a general understanding of the information in this table."}
                #     ] for i in range(size)
                # ]
                guided_request = [
                    LLMGuidedOptions(guided_regex=r"\d,\s*\d"),
                    LLMGuidedOptions(
                        guided_choice=["A.Physics", "B.Mathematics", "C.ComputerScience", "D.QuantitativeBiology",
                                       "E.QuantitativeFinance", "F.Statistics",
                                       "G.ElectricalEngineeringandSystemsScience", "H.Economics"]),
                    LLMGuidedOptions(guided_choice=["A", "B", "C", "D"]),
                ]
                out_list = [[], [], []]
                for q_idx in range(3):
                    prompts = []
                    for i in range(size):
                        message = [{'role': 'user', 'content': f'{prefix[i][q_idx]} {qs_list[i][q_idx]}'}]
                        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                        prompt = TextPrompt(prompt=prompt, multi_modal_data={"image": images[i]})
                        prompts.append(prompt)
                        # print("HELP PROMPT:", prompt)
                    # noinspection PyTypeChecker
                    outputs = model.generate(prompts=prompts,
                                             sampling_params=SamplingParams(
                                                 temperature=0, max_tokens=64, stop_token_ids=stop_token_ids
                                             ),
                                             guided_options_request=guided_request[q_idx],
                                             use_tqdm=False)
                    for i, output in enumerate(outputs):
                        text = output.outputs[0].text
                        print("HELP OUT:", text)
                        out_list[q_idx].append(text)
                out_list = list(zip(*out_list))
                for i in range(size):
                    self.main_input.put(
                        (items[i], qs_list[i], out_list[i], images[i], rows[i], cols[i])
                    )
        except Exception as e:
            print(e)
            self.main_input.put(None)
        self.main_input.put(None)

    def main_process(self):
        tokenizer, model, image_processor, _ = load_pretrained_model(
            main_model_path, None, "llava_qwen", device_map="auto",
            attn_implementation='sdpa',
            # load_8bit=True,
            # load_4bit=False,
            **{
                "multimodal": True,
                "overwrite_config": {
                    "image_aspect_ratio": "anyres_max_9"
                }
            }
        )

        submission = []
        while True:
            item = self.main_input.get()
            # print("MAIN ITEM", item)
            if item is None:
                # if len(submission) < 4000:
                #     sys.exit()
                break

            (item, qs_list, out_list, image, rows, cols) = item
            image_sizes = [image.size]
            images = [image]
            image_tensors = [
                process_images(images, image_processor, model.config)[0].to(dtype=torch.float16, device=device)]
            conv = Conversation(
                system="""<|im_start|>system
                        You are a helpful assistant. Provide only an option's letter or an integer for each question, without any additional explanation.""",
                roles=["<|im_start|>user", "<|im_start|>assistant"],
                version="qwen",
                messages=[
                    ["<|im_start|>user",
                     f'{DEFAULT_IMAGE_TOKEN}\n This is a table image with red borders. The table shape might be ({rows}, {cols}) but could vary. The caption of the table is "{item["caption"]}". Besides that, for the following three questions, the answer from the other model is {out_list}, which you can use as a reference.'],
                    ["<|im_start|>assistant", "I have a general understanding of the information in this table."]
                ],
                offset=0,
                sep_style=SeparatorStyle.CHATML,
                sep="<|im_end|>",
            )
            out_list = self.one_image(model, tokenizer, image_tensors, image_sizes, conv, qs_list)
            sub_item = clean_out(item["image_path"], out_list)
            print("MAIN:", out_list)
            submission.append(sub_item)
        with open('submission.json', 'w') as f:
            json.dump(submission, f)

    def one_image(self, model, tokenizer, image_tensors, image_sizes, conv, qs_list):
        out_list = []
        with torch.inference_mode():
            for qs in qs_list:
                # for qs, processor in zip(qs_list, logits_processor):
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

                output_ids = model.generate(
                    prompts=input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    top_p=1,
                    num_beams=1,
                    max_new_tokens=8,
                    use_cache=True,
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                conv.messages[-1][-1] = outputs
                out_list.append(outputs)
        return out_list

    # def main_process(self):
    #     tokenizer, model, image_processor, _ = load_pretrained_model(
    #         main_model_path, None, "llava_qwen", device_map="auto",
    #         attn_implementation='sdpa',
    #         # load_8bit=True,
    #         # load_4bit=False,
    #         **{
    #             "multimodal": True,
    #             "overwrite_config": {
    #                 "image_aspect_ratio": "anyres_max_9"
    #             }
    #         }
    #     )
    # 
    #     generator = Generator(model)
    #     processor_list = [
    #         RegexLogitsProcessor(r"\d,\s*\d", tokenizer=tokenizer),
    #         RegexLogitsProcessor(r"A|B|C|D|E|F|G|H", tokenizer=tokenizer),
    #         RegexLogitsProcessor(r"A|B|C|D", tokenizer=tokenizer),
    #     ]
    #     submission = []
    #     while True:
    #         item = self.main_input.get()
    #         # print("MAIN ITEM", item)
    #         if item is None:
    #             if len(submission) < 4000:
    #                 sys.exit()
    #             break
    # 
    #         (item, qs_list, out_list, image, rows, cols) = item
    #         image_sizes = [image.size]
    #         images = [image]
    #         image_tensors = [
    #             process_images(images, image_processor, model.config)[0].to(dtype=torch.float16, device=device)]
    #         conv = Conversation(
    #             system="""<|im_start|>system
    #                     You are a helpful assistant. Provide only an option's letter or an integer for each question, without any additional explanation.""",
    #             roles=["<|im_start|>user", "<|im_start|>assistant"],
    #             version="qwen",
    #             messages=[
    #                 ["<|im_start|>user",
    #                  f'{DEFAULT_IMAGE_TOKEN}\n This is a table image with red borders. The table shape might be ({rows}, {cols}) but could vary. The caption of the table is "{item["caption"]}". Besides that, for the following three questions, the answer from the other model is {out_list}, which you can use as a reference.'],
    #                 ["<|im_start|>assistant", "I have a general understanding of the information in this table."]
    #             ],
    #             offset=0,
    #             sep_style=SeparatorStyle.CHATML,
    #             sep="<|im_end|>",
    #         )
    #         out_list = self.one_image(generator, tokenizer, processor_list, image_tensors, image_sizes, conv, qs_list)
    #         sub_item = clean_out(item["image_path"], out_list)
    #         # print("MAIN:", out_list)
    #         submission.append(sub_item)
    #     with open('submission.json', 'w') as f:
    #         json.dump(submission, f)

    # def one_image(self, generator: Generator, tokenizer, logits_processor, image_tensors, image_sizes, conv, qs_list):
    #     out_list = []
    #     with torch.inference_mode():
    #         # for qs in qs_list:
    #         for qs, processor in zip(qs_list, logits_processor):
    #             conv.append_message(conv.roles[0], qs)
    #             conv.append_message(conv.roles[1], None)
    #             prompt = conv.get_prompt()
    #             input_ids = tokenizer_image_token(
    #                 prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    # 
    #             output_ids = generator.generate(
    #                 prompts=input_ids,
    #                 logits_processor=processor,
    #                 images=image_tensors,
    #                 image_sizes=image_sizes,
    #                 do_sample=False,
    #                 temperature=0,
    #                 top_p=1,
    #                 num_beams=1,
    #                 max_new_tokens=8,
    #                 use_cache=True,
    #             )
    #             outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #             conv.messages[-1][-1] = outputs
    #             out_list.append(outputs)
    #     return out_list

worker = Worker()
worker.run()