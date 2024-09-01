# hf_hub_download(repo_id="SpursgoZmy/table-llava-v1.5-7b",
#                 filename="config.json",
#                 resume_download=True,
#                 local_dir="./model")

# snapshot_download(repo_id="liuhaotian/llava-v1.6-vicuna-7b",
#                   resume_download=True,
#                   local_dir="model/llava-v1.6-vicuna-7b")


# snapshot_download(repo_id="MMInstruction/ArxivQA",
#                   repo_type="dataset",
#                   resume_download=True,
#                   local_dir="data/ArxivQA")


# snapshot_download(repo_id="ByteDance/ComTQA",
#                   repo_type="dataset",
#                   resume_download=True,
#                   local_dir="data/ComTQA")


# snapshot_download(repo_id="SpursgoZmy/MMTab",
#                   repo_type="dataset",
#                   resume_download=True,
#                   local_dir="data/MMTab")
#
# snapshot_download(repo_id="microsoft/table-structure-recognition-v1.1-all",
#                   resume_download=True,
#                   local_dir="model/table-structure-recognition-v1.1-all")


# snapshot_download(repo_id="lmms-lab/llava-onevision-qwen2-7b-ov",
#                   resume_download=True,
#                   local_dir="lmms-lab/llava-onevision-qwen2-7b-ov")

# snapshot_download(repo_id="google/siglip-so400m-patch14-384",
#                   resume_download=True,
#                   local_dir="model/siglip-so400m-patch14-384")

# snapshot_download(repo_id="terryoo/TableVQA-Bench",
#                   repo_type="dataset",
#                   resume_download=True,
#                   local_dir="data/TableVQA-Bench")

# !export HF_ENDPOINT=https://hf-mirror.com;huggingface-cli download --resume-download OpenGVLab/InternVL2-1B --local-dir InternVL2-1B --exclude *.zip
# !export HF_ENDPOINT=https://hf-mirror.com;huggingface-cli download --resume-download SpursgoZmy/table-llava-v1.5-7b --local-dir table_llava
# HF_ENDPOINT=https://hf-mirror.com HUGGINGFACE_HUB_CACHE="./cache" HF_HOME="./cache" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python download.py

# path = "OpenGVLab/InternVL2-8B"
# path = "google/siglip-so400m-patch14-384"
# # model = AutoModel.from_pretrained(
# #     path,
# #
# #     cache_dir="./cache",
# #     trust_remote_code=True)
# CLIPVisionModel.from_pretrained(
#     path, torch_dtype=torch.float16, cache_dir="./cache", trust_remote_code=True
# )


from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )
#
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer

#
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
#
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-si"
# model_name = get_model_name_from_path(model_path)
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path, None, model_name, device_map="auto",
# )
