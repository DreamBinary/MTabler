# -*- coding:utf-8 -*-
# @FileName : tmp.py
# @Time : 2024/9/10 13:35
# @Author : fiv


from tatr import TableEngine

engine = TableEngine(
    str_device='cuda',
    str_model_path='../model/TATR/TATR-v1.1-All-msft.pth',
    str_config_path='../model/TATR/structure_config.json',
)
from PIL import Image

img = Image.open('../img/table.jpg')
html, rows, cols = engine(img, './outputs/table.jpg', tokens=[])
print(html)
print(rows, cols)
