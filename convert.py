# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shutil

from PIL import Image

from pix2tex.cli import LatexOCR

img = Image.open("images/1.png")
model = LatexOCR()
print(model(img))

tokenizer_path = "pix2tex/model/dataset/tokenizer.json"
save_dir = "models"
shutil.copy(tokenizer_path, save_dir)
