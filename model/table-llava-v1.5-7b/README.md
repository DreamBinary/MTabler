---
datasets:
- SpursgoZmy/MMTab
- liuhaotian/LLaVA-Instruct-150K
- liuhaotian/LLaVA-Pretrain
language:
- en
metrics:
- accuracy
- bleu
- f1
pipeline_tag: image-text-to-text
---
# Table LLaVA Model Card

<!-- Provide a quick summary of what the model is/does. -->

Table LLaVA 7B is an open-source multimodal chatbot for understanding different table images and fulfilling diverse table-related requests, e.g., question answering, table cell description and structure understanding. 

See the ACL 2024 paper for more details: [Multimodal Table Understanding](https://arxiv.org/abs/2406.08100)

## Model Details

<!-- Provide a longer summary of what this model is. -->

**Model Type:** Table LLaVA 7B strictly follows the [LLaVA-v1.5](https://arxiv.org/abs/2310.03744) model architecture and training pipeline, 
with [CLIP-ViT-L-336px](https://huggingface.co/openai/clip-vit-large-patch14-336) as visual encoder (336*336 image resolution), 
[Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) as base LLM and a two-layer MLP as vision-language connector. 

It was trained with a two-stage pipeline as LLaVA:

1. Pre-training: train the vision-language connector with image-caption data and table recognition data.
2. Instruction tuning: train the vision-language connector and the base LLM with multimodal instruction following data of tabular and non-tabular tasks.

**Code Base:** We use the official code of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) for model training and inference, 
and the saved model checkpoint is uploaded to this repository. Thus, Table LLaVA can be used in the same way as the normal LLaVA v1.5 model with its original code. 

**Model Date:** Table-LLaVA 7B was trained in January 2024.

**Where to send questions or comments about the model:** https://github.com/SpursGoZmy/Table-LLaVA/issues

## Training dataset

The training data includes original LLaVA-1.5 data and specially constructed 
multimodal instruction-following data from the [MMTab dataset](https://huggingface.co/datasets/SpursgoZmy/MMTab), 
which is a large-scale dataset covering a wide range of table images and table-related tasks.

| Training Stage | Data Description | Data Size | Hugging Face Dataset |
| :---: | :---: | :---: | :---: | 
| Pre-training | 558K original LLaVA-1.5 pre-training data | 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) |
|              | 150K table recognition data | 150K | [MMTab-pre_pretrain_data_llava_format_150K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab) |
| Instruction Fine-tuning | 665K original LLaVA-1.5 fine-tuning data | 665K | [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
|              | 232K multimodal instruction tuning data of 14 tabular tasks | 232K | [MMTab-instruct_sft_data_llava_format_232K.json](https://huggingface.co/datasets/SpursgoZmy/MMTab) |

We also provide the merged pre-training and instruction fine-tuning data in the MMTab dataset, 
i.e., enhanced_llava_pretrain_data_708K.json and enhanced_llava_sft_data_898K.json, which was used to train Table LLaVA.

## Evaluation dataset

A collection of 17 held-in and 7 held-out tabular benchmarks, including 15 table-related tasks, e.g., table question answering and table2text generation. 
We also evaluate Table LLaVA on two non-tabular benchmarks: 
[TextVQA](https://textvqa.org/) and [llava-bench-in-the-wild](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild).

## License

Table LLaVA is based on LLaVA-1.5 and thus follows its license. Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## Intended use

**Primary intended uses:** The primary use of Table LLaVA is research on large multimodal models and chatbots, especially for multimodal table understanding.

**Primary intended users:** The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

## Limitations

Table LLaVA takes one table image as the model input. Digesting multiple table images would be valuable to support more application scenarios. Though the proposed Table-LLaVA demonstrates
great performance on a wide range of table-based
tasks, the resolution of input images (336*336) is relatively
low and may limit the upper bound of its capacity. Luckily, with the emergence of MLLMs which
possess higher input image resolution (e.g., Monkey (Li et al., 2023d), LLaVA-Next (Liu et al.,
2024)), researchers can use MMTab to develop more powerful tabular MLLM in the future research.
