# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from PIL import Image

from pix2tex.cli import LatexOCR

img = Image.open("images/2.png")
model = LatexOCR()
print(model(img))
