# -*- coding:utf-8 -*-
# @FileName : thresholding.py
# @Time : 2024/8/12 19:22
# @Author : fiv


from PIL import Image


def thresholding(image_path, threshold=127):
    image = Image.open(image_path).convert('L')
    threshold = 127
    binary_image = image.point(lambda p: 255 if p > threshold else 0)
    binary_image.show()
    return binary_image


if __name__ == "__main__":
    img = thresholding("../table.jpg")
    import numpy

    print(numpy.array(img))
    img.save("tmp.jpg")
