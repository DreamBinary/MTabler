# -*- coding:utf-8 -*-
# @FileName : tsr.py
# @Time : 2024/8/28 21:57
# @Author : fiv
import cv2
import numpy as np
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer


class TSR(TableStructurer):
    def __init__(self, args):
        # python table/predict_structure.py - -table_model_dir =../ inference / slanet_lcnetv2_infer / --table_char_dict_path =../ ppocr / utils / dict / table_structure_dict.txt - -image_dir = docs / table / table.jpg - -output =../ output / table_slanet_lcnetv2 - -use_gpu = False - -benchmark = True - -enable_mkldnn = True - -table_max_len = 512

        super().__init__(args)

    def __call__(self, img):
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

        preds = {}
        preds["structure_probs"] = outputs[1]
        preds["loc_preds"] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result["structure_batch_list"][0]
        bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = structure_str_list[0]
        # structure_str_list = (
        #         ["<html>", "<body>", "<table>"]
        #         + structure_str_list
        #         + ["</table>", "</body>", "</html>"]
        # )
        return structure_str_list, bbox_list


def tsr(image_file_list):
    args = type("Args", (), {
        "table_model_dir": "../model/ch_ppstructure_openatom_SLANetv2_infer",
        "table_char_dict_path": "./table_structure_dict.txt",
        "use_gpu": False,
        # "gpu_id": 0,
        # "gpu_mem": 500,
        "use_npu": False,
        "use_mlu": False,
        "use_xpu": False,
        "precision": "fp32",
        "benchmark": False,
        "use_tensorrt": False,
        "use_onnx": False,
        "table_max_len": 512,
        "enable_mkldnn": True,
        "table_algorithm": "SLANet",
        "merge_no_span_structure": True,
        "cpu_threads": 16,
    })()

    table_structurer = TSR(args)
    result = []
    for image_file in image_file_list:
        # img = Image.open(image_file).convert("RGB")
        img = cv2.imread(image_file)
        structure_res = table_structurer(img)
        structure_str_list, bbox_list = structure_res
        boxes = np.array(bbox_list)
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        result.append((structure_str_list, img))
    return result


def count_rows_and_columns(html_tags):
    rows = 0
    max_columns = 0
    current_columns = 0
    rowspan_columns = {}

    for tag in html_tags:
        if tag.startswith('<tr>'):
            rows += 1
            current_columns = 0
            for col, span in rowspan_columns.items():
                if span > 1:
                    current_columns += 1
                    rowspan_columns[col] -= 1
            max_columns = max(max_columns, current_columns)
        elif tag.startswith('<td'):
            current_columns += 1
            if 'rowspan' in tag:
                rowspan_value = int(tag.split('rowspan="')[1].split('"')[0])
                rowspan_columns[current_columns] = rowspan_value
        elif tag == '</tr>':
            max_columns = max(max_columns, current_columns)

    return rows, max_columns

if __name__ == '__main__':
    image_file_list = ["./tmp.png"]

    result = tsr(image_file_list)
    for s, i in result:
        print(s)
        print(count_rows_and_columns(s))
        # save i
        cv2.imwrite("result.jpg", i)
