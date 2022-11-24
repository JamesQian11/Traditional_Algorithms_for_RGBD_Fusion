import json
import open3d as o3d
from PIL import Image
import numpy as np

for a in range(3, 4):
    for b in range(15,16):
        for c in range(15,16):
            d = 0.01 * c
            radius = a
            sigma_spatial = b
            sigma_range = d
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
            result = np.zeros((reference_image.height, reference_image.width))
            print("result", result.shape)
            for y in range(padding, reference.shape[0] - padding):
                for x in range(padding, reference.shape[1] - padding):  # reference.shape[1]=322
                    curr_pixel = 0
                    k_p = 0
                    for q in range(diameter):
                        for s in range(diameter):
                            value_x = x - (radius - s)
                            value_y = y - (radius - q)
                            I_p = reference[y, x]
                            patch_reference = reference[value_y, value_x]
                            kernel_range = np.exp(
                                -1.0 * (np.abs(patch_reference - I_p).astype(int)) ** 2 / (2 * sigma_range ** 2))
                            kernel_spatial = np.exp(
                                -1.0 * ((value_x - x) ** 2 + (value_y - y) ** 2) / (2 * sigma_spatial ** 2))

                            if not np.isnan(source_upsampled[value_y, value_x]):
                                weight = kernel_range * kernel_spatial
                                patch_source_upsampled = source_upsampled[value_y, value_x]
                                curr_pixel += patch_source_upsampled * weight
                                k_p += weight

                    if not np.isnan(source_upsampled[y, x]):
                        result[y - padding, x - padding] = source_upsampled[y, x]
                    elif k_p == 0:
                        result[y - padding, x - padding] = source_upsampled[y, x]
                    else:
                        result[y - padding, x - padding] = np.round(curr_pixel / k_p)


            calib_file = open(
                "/Users/Vision/Desktop/03_STC/99_Data/RGBD_fusion_records_QianJian_npy/calibration_BKF518_00_P145.json")
            calib_parameter = json.load(calib_file)
            intrinsics = calib_parameter['depth_intrinsics']
            fx = intrinsics.get("fx")
            fy = intrinsics.get("fy")
            cx = intrinsics.get("cx")
            cy = intrinsics.get("cy")
            depth = result
            pointcloud = []
            for v in range(0, 320):
                for u in range(0, 240):
                    if (depth[u, v] > 1) & (depth[u, v] < 32001):
                        z = (depth[u, v])
                        x = (((v * 2 - cx) * z) / fx)
                        y = (((u * 2 - cy) * z) / fy)
                        pointcloud.append([x, y, z])
            np.save("output_best/out_ply_%d_%d_%d" % (a,b,c), pointcloud)

            input_data = "output_best/out_ply_%d_%d_%d.npy" % (a,b,c)
            input_data = np.load(input_data)
            valid = input_data[:, 2] < 32001
            pc = input_data[valid]
            vertices = np.float32(pc)
            output = "output_best/out_ply_%d_%d_%d.ply" % (a,b,c)
            np.savetxt(output, vertices, fmt='%f %f %f')  # 必须先写入，然后利用write()在头部插入ply header
            ply_header = '''ply
format ascii 1.0
obj_info num_cols %(vert_num)d
obj_info num_rows 1
element vertex %(vert_num)d
property float x
property float y
property float z
element range_grid %(vert_num)d
property list uchar int vertex_indices
end_header\n'''
            with open(output, 'r+') as f:
                old = f.read()
                f.seek(0)
                f.write(ply_header % dict(vert_num=len(vertices)))
                f.write(old)

            print("Load a ply point cloud, print it, and render it")
            pcd = o3d.io.read_point_cloud(output)
            o3d.io.write_point_cloud("output_best/out_ply_%d_%d_%d.ply" % (a,b,c), pcd)
            print("a,b,c", a, b, c)