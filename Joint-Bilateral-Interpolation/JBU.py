import cv2
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

radius = 1
sigma_spatial = 1
sigma_range = 1
source = "input/depth_0.png"
reference = "input/rgb_0.png"
output = "output/out_0"

source_image = Image.open(source)
reference_image = Image.open(reference)

source_image_upsampled = source_image.resize(reference_image.size, Image.BILINEAR)

reference = np.array(reference_image)
source_upsampled = np.array(source_image_upsampled, dtype='float16')

save = pd.DataFrame(source_upsampled)
save.to_csv('input/JBI_source_upsampled_no_pad.csv', index=False, header=False)

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
# for i in range(0, 322):
#     for j in range(0, 242):
#         if not np.isnan(source_upsampled[j, i]):
#             print("source_image", source_upsampled[j, i])

save = pd.DataFrame(source_upsampled)
save.to_csv('input/JBI_source_upsampled.csv', index=False, header=False)
# Spatial Gaussian function.
x, y = np.meshgrid(np.arange(diameter) - radius, np.arange(diameter) - radius)
print(x, y)
kernel_spatial = np.exp(-1.0 * (x ** 2 + y ** 2) / (2 * sigma_spatial ** 2))

kernel_spatial = kernel_spatial.reshape(-1, 1)
print("kernel_spatial.shape", kernel_spatial)
# Lookup table for range kernel.
lut_range = np.exp(-1.0 * np.arange(256) ** 2 / (2 * sigma_range ** 2))


def process_row(y):
    result = np.zeros((reference_image.width, 1))
    y += padding#[1-->240]

    for x in range(padding, reference.shape[1] - padding):# reference.shape[1]=322
        print("y,x",y,x)
        I_p = reference[y, x]
        patch_reference = reference[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 1)
        kernel_range = lut_range[np.abs(patch_reference - I_p).astype(int)]
        weight = kernel_range * kernel_spatial
        k_p = weight.sum(axis=0)
        patch_source_upsampled = source_upsampled[y - padding:y + padding + 1:step,
                                 x - padding:x + padding + 1:step].reshape(-1, 1)
        result[x - padding] = np.round(np.nansum(weight * patch_source_upsampled, axis=0) / k_p)
        result = result.flatten()

    return result


executor = ProcessPoolExecutor()
result = executor.map(process_row, range(reference_image.height))
executor.shutdown(True)
np.save(output, list(result))

npy_dataset = np.load("output/out_0.npy")
npy_nparray = np.asarray(npy_dataset, dtype='float64')
data = cv2.resize(npy_nparray, (320, 240))
save = pd.DataFrame(data)
save.to_csv('output/out_0.csv', index=False, header=False)

'''
reference = np.pad(reference, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)
source_upsampled = np.pad(source_upsampled, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)

# Spatial Gaussian function.
x, y = np.meshgrid(np.arange(diameter) - radius, np.arange(diameter) - radius)
kernel_spatial = np.exp(-1.0 * (x**2 + y**2) /  (2 * sigma_spatial**2))
kernel_spatial = np.repeat(kernel_spatial, 3).reshape(-1, 3)
print(kernel_spatial.shape)

# Lookup table for range kernel.
lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * sigma_range**2))

def process_row(y):
   result = np.zeros((reference_image.width, 3))
   y += padding
   for x in range(padding, reference.shape[1] - padding):
      I_p = reference[y, x]
      patch_reference = reference[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)
      print(patch_reference.shape)
      patch_source_upsampled = source_upsampled[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)

      kernel_range = lut_range[np.abs(patch_reference - I_p).astype(int)]

      weight = kernel_range * kernel_spatial

      #print(weight.shape, kernel_range.shape, kernel_spatial.shape)
      k_p = weight.sum(axis=0)
      #print(k_p.shape)
      result[x - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=0) / k_p)
      print(result.shape)
   return result

executor = ProcessPoolExecutor()
result = executor.map(process_row, range(reference_image.height))
executor.shutdown(True)
Image.fromarray(np.array(list(result)).astype(np.uint8)).save(output)
'''
