import os

import cv2
import numpy as np
from paddleocr.paddleocr import parse_args
from paddleocr.ppstructure.predict_system import StructureSystem

OCR_BASE_DIR = "./OCRCache"


class OCR(StructureSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)

        params.structure_version = "PP-StructureV2"
        params.use_gpu = False
        params.mode = "structure"

        params.det_model_dir = os.path.join(OCR_BASE_DIR, "whl", "det", "en", "en_PP-OCRv3_det_infer")
        params.rec_model_dir = os.path.join(OCR_BASE_DIR, "whl", "rec", "en", "en_PP-OCRv4_rec_infer")
        params.table_model_dir = os.path.join(OCR_BASE_DIR, "whl", "table", "en_ppstructure_mobile_v2.0_SLANet_infer")
        # params.layout_model_dir = os.path.join(BASE_DIR, "whl", "layout")

        params.rec_char_dict_path = os.path.join(OCR_BASE_DIR, "dict", "en_dict.txt")
        params.table_char_dict_path = os.path.join(OCR_BASE_DIR, "dict", "table_structure_dict.txt")
        # params.layout_dict_path = os.path.join(BASE_DIR, "dict", "layout_publaynet_dict.txt")

        super().__init__(params)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        res, table_time_dict = self.table_system(
            img, return_ocr_result_in_table
        )
        return res


img_path = 'img/table1.png'
img = cv2.imread(img_path)
table_engine = OCR(layout=False, show_log=False, lang="en")
res = table_engine(img, True)
cell_bbox = res['cell_bbox']
html = res['html']
# print(res["rec_res"])
print(res["boxes"])
print(html)


def draw_bbox(img, boxes, color=(0, 0, 255), thickness=2):
    boxes = np.array(boxes)
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img


img = draw_bbox(img, res["boxes"])
cv2.imwrite("img/table1_bbox.png", img)
# cv2.imshow("img", img)
