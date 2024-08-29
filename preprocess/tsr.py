# -*- coding:utf-8 -*-
# @FileName : tsr.py
# @Time : 2024/8/12 19:39
# @Author : fiv


import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"
tsr_img_processor = AutoImageProcessor.from_pretrained(tsr_model_path)
tsr_img_processor.size = {'height': 800, 'width': 800}

print(tsr_img_processor.size)
tsr_model = TableTransformerForObjectDetection.from_pretrained(tsr_model_path)
label2id = tsr_model.config.label2id
label_row = label2id['table row']
label_col = label2id['table column']


def tsr_process(raw_image):
    image = raw_image.copy()
    inputs = tsr_img_processor(images=image, return_tensors="pt")
    outputs = tsr_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (height, width) of each image in the batch
    results = tsr_img_processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
    draw = ImageDraw.Draw(image)
    rows = 0
    cols = 0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()

        label = label.item()
        print(label)
        if label == label_row:
            draw.rectangle(box, outline="red", width=2)
            rows += 1
        elif label == label_col:
            draw.rectangle(box, outline="red", width=2)
            cols += 1
    return image, rows, cols


if __name__ == "__main__":
    file_path = "./tmp.png"
    raw_image = Image.open(file_path).convert("RGB")
    bordered_image, rows, cols = tsr_process(raw_image)
    bordered_image.save("tmp.png")
    print(rows, cols)
