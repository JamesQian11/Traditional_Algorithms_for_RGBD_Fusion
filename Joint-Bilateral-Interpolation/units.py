import csv
import json
import os
import pandas as pd
import torch
from PIL import Image
import cv2
import numpy as np
import open3d as o3d
from torch.distributions import transforms


def floatZ2poingtcloud():
    basedir = "/Users/Vision/Desktop/03_STC/99_Data/RGBD_fusion_records_QianJian_npy/FF_ToF_REFERENCE/"
    calib_file = open(
        "/Users/Vision/Desktop/03_STC/99_Data/RGBD_fusion_records_QianJian_npy/calibration_BKF518_00_P145.json")
    calib_parameter = json.load(calib_file)
    intrinsics = calib_parameter['depth_intrinsics']
    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    print(fx, fy)
    depthScale = 1.0
    depth_datapath = basedir + "float_z.npy"
    depth = np.load(depth_datapath)
    print(depth.shape)
    for i in range(0, 60):
        pointcloud = []
        for v in range(0, 640):
            for u in range(0, 480):
                if (depth[i, u, v] > 5) & (depth[i, u, v] < 2100):
                    z = (depth[i, u, v] / depthScale)
                    x = (((v - cx) * z) / fx)
                    y = (((u - cy) * z) / fy)
                    pointcloud.append([x, y, z])
        # print(pointcloud)
        np.save("FF_PointCloud/FF_depth_org_%d.npy" % i, pointcloud)
    return


def rotate(inpath, outpath):
    # 读取图像
    im = Image.open(inpath)
    # 指定逆时针旋转的角度
    im_rotate = im.rotate(180)
    im_rotate.show()
    im_rotate.save(outpath)
    return


def jointBilateralFilter():
    src = cv2.imread("Fusion/depth_0.jpg")
    joint = cv2.imread("Fusion/RGB_0.jpg")
    radius = 15
    sigma_color = 10
    sigma_space = 10
    dst = cv2.ximgproc.jointBilateralFilter(joint, src, radius, sigma_color, sigma_space)
    cv2.imshow('dst', dst)
    cv2.imwrite("Fusion/output_cv2.png", dst)
    return


def canny(map):
    img = cv2.imread(map, 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, 50, 150)
    cv2.imshow('Canny', canny)
    cv2.waitKey(0)
    return ()


def resize_map(input, output):
    img = cv2.imread(input)
    print(img.shape)
    im_resized = cv2.resize(img, (304, 228), interpolation=cv2.INTER_LINEAR)
    print(im_resized.shape)
    cv2.imwrite(output, im_resized)
    print("finish")
    return ()


def blend_two_images():
    img1 = Image.open("RGB_image/gray_img_0.png")
    img1 = img1.convert('RGBA')
    img2 = Image.open("BKF_Matched_Depth-304-228/depth_0.jpg")
    img2 = img2.convert('RGBA')
    img = Image.blend(img1, img2, 0.4)
    img.show()
    img.save("BKF_Matched_Depth-304-228/depth_rgb_0.png")
    return


def jpg2csv(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 打开图像
    img_nparray = np.asarray(img, dtype='float64')
    data = cv2.resize(img_nparray, (320, 240))
    save = pd.DataFrame(data)
    save.to_csv('/Users/Vision/Desktop/03_STC/03_RGBD_Fusion/20210629_SpotToF/Fusion/out_noscale_0.csv', index=False,
                header=False)


def npy2csv(path):
    npy_dataset = np.load(path)
    npy_nparray = np.asarray(npy_dataset, dtype='float64')
    data = cv2.resize(npy_nparray, (320, 240))
    save = pd.DataFrame(data)
    save.to_csv('output/out_0.csv', index=False, header=False)


def png2jpg(PngPath):
    img = cv2.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = "/Users/Vision/Desktop/03_STC/03_RGBD_Fusion/20210629_SpotToF/Fusion/depth_noscale_0.jpg"
    img = Image.open(infile)
    img = img.resize((int(w), int(h)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


def npy2rgb(path):
    rgb_dataset = np.load(path)
    rgb_dataset = np.array(rgb_dataset, dtype='float16')
    print(rgb_dataset.shape)
    for i in range(0, 60):
        source_upsampled = (rgb_dataset[i, :, :])
        source_upsampled = pd.DataFrame(source_upsampled)
        source_upsampled.to_csv("FF_depthmap/depthmap_%d.csv" % i, index=False, header=False)
        np.save("FF_depthmap/depth_%d.npy" % i, source_upsampled)
        # cv2.imwrite("FF_depthmap/depth_%d.jpg" %i, source_upsampled)
        # image = Image.fromarray(img)
        # image.save("FF_depthmap/depthmap_%d.png" % i)


def csv2rgb():
    rgb_dataset = np.loadtxt("FF_depthmap/depthmap_0.csv")
    cv2.imwrite("FF_depthmap/depth_%d.jpg", rgb_dataset)
    # image = Image.fromarray(rgb_dataset)
    # image.save("FF_depthmap/depthmap_0.png")


def to16bitgrayscale():
    path = "matched_depth/depth_0.npy"
    arr = np.load(path)  # or np.ones etc.
    array_buffer = arr.tobytes()
    img = Image.new("I", arr.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")
    img.save("input/depth_0.png")
    return


def rgb2gray():
    for i in range(0, 60):
        img = cv2.imread("RGB_image/%d.jpg" % i, cv2.IMREAD_UNCHANGED)
        print(img)
        shape = img.shape
        print(shape)
        if shape[2] == 3 or shape[2] == 4:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # cv2.imshow("gray_img", img_gray)
        cv2.imwrite("RGB_image/gray_img_%d.png" % i, img_gray)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return


def csv2pointcloud(csv_path, npy_path, out_path):
    calib_file = open(
        "/Users/Vision/01_Project/02_STC-SONY/99_Data/RGBD_fusion_records_QianJian_npy/calibration_BKF518_00_P145.json")
    calib_parameter = json.load(calib_file)
    intrinsics = calib_parameter['depth_intrinsics']
    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    data = pd.read_csv(csv_path, header=None)
    print(data.shape)
    # 再使用numpy保存为npy
    np.save(npy_path, data)
    depth = np.load(npy_path)  # , dtype=np.float, delimiter=',') #lodtxt
    print(depth.shape)
    pointcloud = []
    for v in range(0, 320):
        for u in range(0, 240):
            if (depth[u, v] > 1) & (depth[u, v] < 32001):
                z = (depth[u, v])
                x = (((v * 2 - cx) * z) / fx)
                y = (((u * 2 - cy) * z) / fy)
                pointcloud.append([x, y, z])
    np.save(out_path, pointcloud)


def npy2ply(input_data, output):
    input_data = np.load(input_data)
    valid = input_data[:, 2] < 32001
    pc = input_data[valid]
    vertices = np.float32(pc)
    # colors = colors.reshape(-1, 3)
    # vertices = np.hstack([vertices.reshape(-1, 3), colors])
    #   one = np.ones((195529,3))
    #   points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
    #   colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    #   Generate point cloud
    #   create_output(points_3D, colors, output_file)
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
end_header\n
'''
    with open(output, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(output)
    o3d.io.write_point_cloud(output, pcd)
    o3d.visualization.draw_geometries([pcd])
    print('finish show')



def pcshow(path):
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])
    print('finish show')


def confidence_decoder(path):
    confidence_dataset = np.load(path)
    print(confidence_dataset)
    for i in range(0, 60):
        confidence = (confidence_dataset[i, :, :])
        confidence = pd.DataFrame(confidence)
        confidence.to_csv("FF_Confidence/ff_confidence_%d.csv" % i, index=False, header=False)
        np.save("FF_Confidence/ff_confidence_%d.npy" % i, confidence)


def confidence_filter():
    confidence_path = "FF_Confidence/ff_confidence_0.npy"
    confidence = np.load(confidence_path)
    depth_path = "FF_Depthmap/depth_0.npy"
    depth = np.load(depth_path)
    print(depth.shape[0])
    for i in range(0, depth.shape[0]):
        for j in range(0, depth.shape[1]):
            scale = confidence[i, j]
            print(scale)
            if scale < 60:
                depth[i, j] = 0.0
    np.save("FF_Depthmap/ff_depth_0.npy", depth)
    depth = pd.DataFrame(depth)
    depth.to_csv("FF_Depthmap/ff_depth_0.csv", index=False, header=False)


def depthnpy2plynpy():
    calib_file = open(
        "/Users/Vision/Desktop/03_STC/99_Data/RGBD_fusion_records_QianJian_npy/calibration_BKF518_00_P145.json")
    calib_parameter = json.load(calib_file)
    intrinsics = calib_parameter['depth_intrinsics']
    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    print(fx, fy)
    depthScale = 1.0
    depth_datapath = "/Users/Vision/Desktop/03_STC/03_RGBD_Fusion/20210629_SpotToF/FF_Depthmap/ff_depth_0.npy"
    depth = np.load(depth_datapath)
    print(depth.shape)
    pointcloud = []
    for v in range(0, 640):
        for u in range(0, 480):
            if (depth[u, v] > 5) & (depth[u, v] < 2100):
                z = (depth[u, v] / depthScale)
                x = (((v - cx) * z) / fx)
                y = (((u - cy) * z) / fy)
                pointcloud.append([x, y, z])
    print(pointcloud)
    np.save("FF_PointCloud/FF_depth_0.npy", pointcloud)


def save_as_open3d():
    for a in range(3, 11):
        for b in range(3, 11):
            for c in range(1, 11):
                output = "output_best/out_ply_%d_%d_%d.ply" % (a, b, c)
                pcd = o3d.io.read_point_cloud(output)
                o3d.io.write_point_cloud("output_best/out_ply_%d_%d_%d.ply" % (a, b, c), pcd)


def csv2npy():
    # 先用pandas读入csv
    data = pd.read_csv("spot/upsampling_result_1000.csv", header=None)
    print(data.shape)
    # 再使用numpy保存为npy
    np.save("spot/upsampling_result_1000.npy", data)


def csv2png():
    input_fn = "FF_depthmap/depthmap_0.csv"
    output_fn = "FF_depthmap/depthmap_0.png"

    csv_file = open(input_fn, "r")
    csv_image = csv.reader(csv_file, delimiter=',')
    image_width = 0
    image_height = 0
    raster = []
    for row in csv_image:
        w = len(row)
        raster.append(row)
        image_height += 1
        image_width = max(image_width, w)
    image = Image.new('RGB', (image_width, image_height))
    # print("raster",raster)
    for y, r in enumerate(raster):
        for x, p in enumerate(r):
            p = float(p)
            c = (p * 255) / 3000
            print(c)
            c = int(c)
            # if p.strip() == '0':
            #     p = 0
            # else:
            #     p = 255
            # print(p)
            image.putpixel((x, y), (c, c, c))
    image.save(output_fn)


def npy2tensor():
    rgb = "spot/reference_rgb_304_228.jpg"
    rgb = Image.open(rgb)

    transf = transforms.ToTensor()
    img_tensor = transf(rgb)  # tensor数据格式是torch(C,H,W)
    print(img_tensor.size())

    depth = "BKF_Matched_Depth-304-228/depth_0.npy"
    depth = np.load(depth)
    depth_tensor = torch.from_numpy(depth / 1000)
    print(depth_tensor[31, 0])

    depth_tensor = torch.unsqueeze(depth_tensor, dim=2).permute(2, 0, 1)
    print(depth_tensor.size())

    d2 = torch.vstack((img_tensor, depth_tensor))
    print(d2.size())
    d3 = torch.unsqueeze(d2, dim=3).permute(3, 0, 1, 2)
    print(d3.size())

    torch.save(d3, "spot/rgb_depth-304-228.pth")


def depth_filter():
    csv = "input/spot_result_1000_alph3.csv"
    csv_depth = np.loadtxt(csv, dtype=str, delimiter=",")
    pointcloud = np.zeros((240, 320))
    for i in range(0, 240):
        for j in range(0, 320):
            if float(csv_depth[i, j]) > 1080.0:
                csv_depth[i, j] = 0.0
            else:
                csv_depth[i, j] = float(csv_depth[i, j])

            pointcloud[i, j] = csv_depth[i, j]

    save = pd.DataFrame(pointcloud)
    save.to_csv('input/spot_result_1000_alph3_filter.csv', index=False, header=False)


if __name__ == '__main__':
    # csv2npy()
    # rgb2gray()
    # resize_map("RGB_image/0.jpg","spot/reference_rgb_304_228.jpg")
   # blend_two_images()

    # toRGB("/Users/Vision/Desktop/03_STC/03_RGBD_Fusion/20210629_SpotToF/Fusion/out_0.jpg")
    # rotate("PointCloudmap/depthmap_new11_1.png","PointCloudmap/depthmap_new_rotate11_1.png")
    # png2jpg("/Users/Vision/Desktop/03_STC/03_RGBD_Fusion/20210629_SpotToF/Fusion/depth_noscale_0.png")
    # jpg2csv("/Users/Vision/Desktop/03_STC/03_RGBD_Fusion/20210629_SpotToF/Fusion/out_noscale_0.jpg")
    # npy2csv("spot/result_matlab.npy")
    # csv2pointcloud("spot/upsampling_result_1000.npy")
    # confidence_filter()
    # depthnpy2plynpy()
    # npy2ply("spot/result_matlab.npy", "spot/result_matlab.ply")
    # floatZ2poingtcloud()
    # to16bitgrayscale()
    #pcshow("output_best/bkf_pc_out_sparse_0.ply")
    # csv2rgb()
    # npy2rgb("/Users/Vision/Desktop/03_STC/99_Data/RGBD_fusion_records_QianJian_npy/FF_ToF_REFERENCE/float_z.npy")
    csv_path = "/Users/Vision/01_Project/02_STC-SONY/03_RGBD-Fusion/20210816_VisionToolbox/3D/DepthEnhancer/tofmark" \
               "/tgvl2/out_bestX_0.csv"
    # csv_path = "input/spot_result_1000_alph5_filter.csv"
    npy_path = "spot/spot_result_x.npy"
    out_path = "spot/spot_result_x.npy"
    output = "spot/spot_result_x.ply"
    csv2pointcloud(csv_path, npy_path, out_path)
    npy2ply(out_path, output)
