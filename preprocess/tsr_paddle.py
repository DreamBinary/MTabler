# -*- coding:utf-8 -*-
# @FileName : tsr.py
# @Time : 2024/8/28 21:57
# @Author : fiv
from collections import defaultdict

import cv2
import numpy as np
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer


class TSR(TableStructurer):
    def __init__(self, args):
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
    table_model_dir = "/root/MTabler/model/ch_ppstructure_openatom_SLANetv2_infer"
    table_char_dict_path = "./table_structure_dict.txt"
    args = type("Args", (), {
        "table_model_dir": table_model_dir,
        "table_char_dict_path": table_char_dict_path,
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
    index = 0
    columns_cnt = defaultdict(int)
    while index < len(html_tags):
        tag = html_tags[index]

        if tag == '<tr>':
            rows += 1
            current_columns = 0

            # Account for any ongoing rowspans from previous rows
            for col, span in rowspan_columns.items():
                if span > 1:
                    current_columns += 1
                    rowspan_columns[col] -= 1

        elif tag.startswith('<td'):
            colspan = 1
            rowspan = 1

            # Check if 'colspan' and 'rowspan' are in the subsequent strings
            if index + 1 < len(html_tags) and 'colspan="' in html_tags[index + 1]:
                colspan = int(html_tags[index + 1].strip().split('colspan="')[1].split('"')[0])
                index += 1  # Skip the colspan string
            if index + 1 < len(html_tags) and 'rowspan="' in html_tags[index + 1]:
                rowspan = int(html_tags[index + 1].strip().split('rowspan="')[1].split('"')[0])
                index += 1  # Skip the rowspan string

            # Increment columns count
            current_columns += colspan

            # Track rowspans for subsequent rows
            if rowspan > 1:
                for _ in range(colspan):
                    rowspan_columns[current_columns - _] = rowspan

        elif tag == '</tr>':
            print(f"Row {rows} has {current_columns} columns")
            columns_cnt[current_columns] += 1
            max_columns = max(max_columns, current_columns)

        index += 1
    columns = max(columns_cnt, key=columns_cnt.get)
    return rows, columns


if __name__ == '__main__':
    image_file_list = ["./tmp.png"]
    print(count_rows_and_columns(['<table>', '<thead>', '<tr>', '<td></td>', '<td></td>', '</tr>', '</thead>', '<tbody>', '<tr>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '</tr>', '</tbody>', '</table>']))
    result = tsr(image_file_list)
    for s, i in result:
        print(s)
        print(count_rows_and_columns(s))
        # save i
        cv2.imwrite("result.jpg", i)
