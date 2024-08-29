#!bin/bash

# 注意将pretrained_model的路径设置为本地路径。

python table/predict_structure.py --table_model_dir=../inference/slanet_lcnetv2_infer/ --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --image_dir=docs/table/table.jpg --output=../output/table_slanet_lcnetv2 --use_gpu=False --benchmark=True --enable_mkldnn=True --table_max_len=512
