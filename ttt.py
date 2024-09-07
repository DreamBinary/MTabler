import copy
import os
import warnings
from collections import defaultdict

import cv2
import numpy as np
from paddleocr.paddleocr import parse_args
from paddleocr.ppocr.data.imaug import transform
from paddleocr.ppstructure.table.predict_structure import TableStructurer
from paddleocr.ppstructure.table.predict_table import TableSystem

warnings.filterwarnings("ignore")
OCR_BASE_DIR = "./OCRCache"

class OCR(TableSystem):
    def __init__(self, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)

        params.structure_version = "PP-StructureV2"
        params.use_gpu = False
        params.mode = "structure"

        params.table_max_len = 488
        params.precision = "fp32"
        params.enable_mkldnn = True
        params.merge_no_span_structure = True
        params.cpu_threads = 16

        params.table_algorithm = "SLANet"

        params.use_mp = True
        params.total_process_num = 4

        params.det_model_dir = os.path.join(OCR_BASE_DIR, "whl", "det", "en", "en_PP-OCRv3_det_infer")
        params.rec_model_dir = os.path.join(OCR_BASE_DIR, "whl", "rec", "en", "en_PP-OCRv4_rec_infer")
        params.table_model_dir = os.path.join(OCR_BASE_DIR, "whl", "table", "en_ppstructure_mobile_v2.0_SLANet_infer")
        # params.layout_model_dir = os.path.join(BASE_DIR, "whl", "layout")

        params.rec_char_dict_path = os.path.join(OCR_BASE_DIR, "dict", "en_dict.txt")
        params.table_char_dict_path = os.path.join(OCR_BASE_DIR, "dict", "table_structure_dict.txt")
        # params.layout_dict_path = os.path.join(BASE_DIR, "dict", "layout_publaynet_dict.txt")

        super().__init__(params)

    def run(self, img, path):
        # result = dict()
        structure_res, elapse = self._structure(copy.deepcopy(img))
        # result["cell_bbox"] = structure_res[1].tolist()
        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(copy.deepcopy(img))
        # result["boxes"] = [x.tolist() for x in dt_boxes]
        boxes = [x.tolist() for x in dt_boxes]
        # result["rec_res"] = rec_res
        # pred_html = self.match(structure_res, dt_boxes, rec_res)
        # result["html"] = pred_html
        # print(boxes)
        img = self.draw_bbox(img, boxes)
        cv2.imwrite(path, img)
        return structure_res[0]

    def draw_bbox(self, img, boxes):
        # img = copy.deepcopy(img)
        boxes = np.array(boxes).astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

        return img


ocr = OCR()
img = cv2.imread("img/table.jpg")
path = "img/output.jpg"
print(ocr.run(img, path))