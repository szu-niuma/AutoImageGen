import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from sklearn.preprocessing import binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from auto_image_edit.utils.image_sift_visualization import quick_alignment_visualization

# 快速可视化
result = quick_alignment_visualization(
    "/home/yuyangxin/data/AutoImageGen/resource/clock_real.png",
    "/home/yuyangxin/data/AutoImageGen/resource/clock_fake.png",
    "./output_dir",
)
