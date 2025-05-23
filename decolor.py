import os
from PIL import Image

input_dir = 'test'
output_dir = 'gray'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    try:
        with Image.open(input_path) as img:
            gray_img = img.convert('L')
            gray_img.save(output_path)
    except:
        pass  # 忽略非图片文件
