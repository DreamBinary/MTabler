# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/table/SLANet_lcnetv2.yml -o Global.pretrained_model=path/best_accuracy Global.save_inference_dir=./inference/slanet_lcnetv2_infer
