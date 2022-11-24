import cv2
from PIL import Image
import numpy as np
import pandas as pd

radius = 7
sigma_spatial = 5
sigma_range = 0.51

source = "input/bkf_sparse_depth_0.png"
reference = "input/rgb_0.png"


source_image = Image.open(source)
reference_image = Image.open(reference)

source_image_upsampled = source_image.resize(reference_image.size, Image.BILINEAR)

reference = np.array(reference_image)
# reference = pd.DataFrame(reference)
# reference.to_csv('input/JBI_reference_no_pad.csv', index=False, header=False)

source_upsampled = np.array(source_image_upsampled, dtype='float16')
# source_upsampled = pd.DataFrame(source_upsampled)
# source_upsampled.to_csv('input/JBI_source_no_pad.csv', index=False, header=False)

scale = source_image_upsampled.width / reference_image.width

radius = int(radius)
diameter = 2 * radius + 1
step = int(np.ceil(1 / scale))
padding = radius * step
sigma_spatial = float(sigma_spatial)
sigma_range = float(sigma_range)
print("padding", padding)
reference = np.pad(reference, ((padding, padding), (padding, padding)), 'symmetric').astype(np.float32)
source_upsampled = np.pad(source_upsampled, ((padding, padding), (padding, padding)), 'symmetric').astype(
    np.float32)
print("source_image", source_upsampled.shape)
source_upsampled[source_upsampled == 0] = np.nan
# save = pd.DataFrame(source_upsampled)
# save.to_csv('input/JBI_source_upsampled.csv', index=False, header=False)
result = np.zeros((reference_image.height,reference_image.width))
print("result", result.shape)
for y in range(padding, reference.shape[0] - padding):
    for x in range(padding, reference.shape[1] - padding):
        curr_pixel = 0
        k_p = 0
        for q in range(diameter):
            for s in range(diameter):
                value_x = x - (radius - s)
                value_y = y - (radius - q)
                I_p = reference[y, x]
                patch_reference = reference[value_y, value_x]
                kernel_range = np.exp(-1.0 * (np.abs(patch_reference - I_p).astype(int)) ** 2 / (2 * sigma_range ** 2))
                kernel_spatial = np.exp(-1.0 * ((value_x - x) ** 2 + (value_y - y) ** 2) / (2 * sigma_spatial ** 2))
                if not np.isnan(source_upsampled[value_y, value_x]):
                    weight = kernel_range * kernel_spatial
                    patch_source_upsampled = source_upsampled[value_y, value_x]
                    #print("patch_source_upsampled",patch_source_upsampled)
                    #print("value_y,value_x",value_y,value_x)
                    curr_pixel += patch_source_upsampled * weight
                    k_p += weight
        if not np.isnan(source_upsampled[y,x]):
            result[y - padding,x - padding] = source_upsampled[y, x]
        elif k_p == 0:
            result[y - padding, x - padding] = source_upsampled[y, x]
        else:
            result[y - padding,x - padding] = np.round(curr_pixel / k_p)

output = "output/out_bestX_0"
np.save(output, result)
npy_dataset = np.load("output/out_bestX_0.npy")
npy_nparray = np.asarray(npy_dataset)
#print(npy_nparray)
data = cv2.resize(npy_nparray, (320, 240))
save = pd.DataFrame(npy_dataset)
save.to_csv('/Users/Vision/01_Project/02_STC-SONY/03_RGBD-Fusion/20210816_VisionToolbox/3D/DepthEnhancer/tofmark'
            '/tgvl2/out_bestX_0.csv', index=False, header=False)
