# Convert LaTeX-OCR To ONNX

### 1. Clone the source code.
```bash
git clone https://github.com/SWHL/ConvertLaTeXOCRToONNX.git
```

### 2. Install env.
#### Anaconda
```bash
conda env create -f environment.yml
```

#### Pip
```bash
$ conda create -n cvt_latex python=3.10.13

$ conda activate cvt_latex

$ pip install -r requirements.txt
```


### 3. Run the convert.py, and the converted model is located in the `models` directory.
```bash
$ python convert.py

# /Users/xxx/miniconda3/envs/convert_latex_ocr/lib/python3.10/site-packages/torch/onnx/symbolic_helper.py:1513: UserWarning: ONNX export mode is set to TrainingMode.EVAL, but operator 'batch_norm' is set to train=True. Exporting with train=True.
#   warnings.warn(
# Exported model has been tested with ONNXRuntime, and the result looks good!
# ONNX Model has been saved /Users/xxx/projects/_self/ConvertLaTeXOCRToONNX/models/image_resizer.onnx
# Exported model has been tested with ONNXRuntime, and the result looks good!
# ONNX Model has been saved /Users/xxx/projects/_self/ConvertLaTeXOCRToONNX/models/encoder.onnx
# Exported model has been tested with ONNXRuntime, and the result looks good!
# ONNX Model has been saved /Users/xxx/projects/_self/ConvertLaTeXOCRToONNX/models/decoder.onnx
# \exp\left[\int d^{4}x g\phi\bar{\psi}\psi\right]=\sum_{n=0}^{\infty}\frac{g^{n}}{n!}\left(\int d^{4}x\phi\bar{\psi}\psi\right)^{n}.
```


### 4. Used in [RapidLaTeXOCR](https://github.com/RapidAI/RapidLaTeXOCR).

```python
from rapid_latex_ocr import LatexOCR

image_resizer_path = 'models/image_resizer.onnx'
encoder_path = 'models/encoder.onnx'
decoder_path = 'models/decoder.onnx'
tokenizer_json = 'models/tokenizer.json'
model = LatexOCR(image_resizer_path=image_resizer_path,
                 encoder_path=encoder_path,
                 decoder_path=decoder_path,
                 tokenizer_json=tokenizer_json)

img_path = "tests/test_files/6.png"
with open(img_path, "rb") as f:
    data = f. read()

result, elapse = model(data)

print(result)
# {\frac{x^{2}}{a^{2}}}-{\frac{y^{2}}{b^{2}}}=1

print(elapse)
# 0.4131628000000003
```
