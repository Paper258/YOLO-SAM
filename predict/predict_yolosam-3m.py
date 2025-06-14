from pathlib import Path

from osgeo import gdal
import datetime
import math

import shapely
from PIL import Image
import gc
from segment_anything import sam_model_registry, SamPredictor
import rasterio
import geopandas as gpd
from multiprocessing import Pool

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.vit.sam import PromptPredictor, build_sam
from ultralytics.yolo.utils.torch_utils import select_device
from osgeo import ogr, gdal, osr
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print(torch.cuda.is_available())


def raster_to_vector(source, output, simplify_tolerance=None, **kwargs):
    from rasterio import features

    with rasterio.open(source) as src:
        band = src.read()

        mask = band != 0
        shapes = features.shapes(band, mask=mask, transform=src.transform)

    fc = [
        {"geometry": shapely.geometry.shape(shape), "properties": {"value": value}}
        for shape, value in shapes
    ]
    if simplify_tolerance is not None:
        for i in fc:
            i["geometry"] = i["geometry"].simplify(tolerance=simplify_tolerance)

    gdf = gpd.GeoDataFrame.from_features(fc)
    if src.crs is not None:
        gdf.set_crs(crs=src.crs, inplace=True)
    gdf.to_file(output, **kwargs)



imagesize = 1024


# batch = 20


modelpath = r'/home/neaucs2/usr/xrx/yolo8-sam/yolov8x.pt'
model = YOLO(modelpath)

src_tif = r"/home/neaucs2/usr/xrx/yolo8-sam/test/3m/final-3band-3m.tif"
#src_tif = r"/home/neaucs2/usr/xrx/yolo8-sam/test/3m/groundtruth/33.tif"
outtif = r"/home/neaucs2/usr/xrx/yolo8-sam/test/3m/result/ground-val-191-1024.tif"
# outtif = r"/home/neaucs2/usr/xrx/yolo8-sam/test/test_quantu/result/ground-val.tif"

area_perc = 0.5

#  读取tif数据集
def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount

    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (imagesize - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (imagesize - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (imagesize - SideLength * 2): i * (imagesize - SideLength * 2) + imagesize,
                      j * (imagesize - SideLength * 2): j * (imagesize - SideLength * 2) + imagesize]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (imagesize - SideLength * 2): i * (imagesize - SideLength * 2) + imagesize,
                  (img.shape[1] - imagesize): img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - imagesize): img.shape[0],
                  j * (imagesize - SideLength * 2): j * (imagesize - SideLength * 2) + imagesize]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - imagesize): img.shape[0],
              (img.shape[1] - imagesize): img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (imagesize - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (imagesize - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


def cal_area(path_shp):
    """
    :param path_shp: 输入矢量文件
    :return:
    """
    print("正在获取矢量面积。。。")
    driver = ogr.GetDriverByName("ESRI Shapefile")  # 创建数据驱动
    ds = driver.Open(path_shp, 1)  # 创建数据资源
    layer = ds.GetLayer()
    new_field = ogr.FieldDefn("Area", ogr.OFTReal)  # 创建新的字段
    new_field.SetWidth(32)
    new_field.SetPrecision(16)
    layer.CreateField(new_field)
    number = 0
    for feature in layer:
        fid = feature.GetFID()
        geom = feature.GetGeometryRef()
        geom2 = geom.Clone()
        area = geom2.GetArea() * 3 / 2000  # 默认为平方米 改亩
        # area = area / 1000000 # 转化为平方公里
        feature.SetField("Area", area)
        # feature.GetField('Area')
        if area > 0.5 and area < 1000:
            layer.SetFeature(feature)
        else:
            layer.DeleteFeature(int(fid))
        geometry = feature.GetGeometryRef()
        # x = geometry.GetX()
        extent = geometry.GetEnvelope()
        # extent = layer.GetExtent()
        extentArea = math.fabs(extent[0] - extent[1]) * math.fabs(extent[2] - extent[3])
        
        if extentArea > 3 * geom2.GetArea():
            layer.DeleteFeature(int(fid))
        
    ds = None


#  标签可视化，即为第n类赋上n值
def labelVisualize(img):
    img_out = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #  为第n类赋上n值
            img_out[i][j] = np.argmax(img[i][j])

    return img_out


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, item in enumerate(npyfile):
        # img = labelVisualize(item)
        img = item.astype(np.uint8)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: imagesize - RepetitiveLength, 0: imagesize - RepetitiveLength] = img[
                                                                                           0: imagesize - RepetitiveLength,
                                                                                           0: imagesize - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                # result[shape[0] - ColumnOver : shape[0], 0 : imagesize - RepetitiveLength] = img[0 : ColumnOver, 0 : imagesize - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: imagesize - RepetitiveLength] = img[
                                                                                                              imagesize - ColumnOver - RepetitiveLength: imagesize,
                                                                                                              0: imagesize - RepetitiveLength]
            else:
                result[j * (imagesize - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        imagesize - 2 * RepetitiveLength) + RepetitiveLength,
                0:imagesize - RepetitiveLength] = img[RepetitiveLength: imagesize - RepetitiveLength,
                                                  0: imagesize - RepetitiveLength]
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: imagesize - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[
                                                                                        0: imagesize - RepetitiveLength,
                                                                                        imagesize - RowOver: imagesize]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[
                                                                                        imagesize - ColumnOver: imagesize,
                                                                                        imagesize - RowOver: imagesize]
            else:
                result[j * (imagesize - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        imagesize - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: imagesize - RepetitiveLength,
                                                imagesize - RowOver: imagesize]
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: imagesize - RepetitiveLength,
                (i - j * len(TifArray[0])) * (imagesize - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (imagesize - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: imagesize - RepetitiveLength, RepetitiveLength: imagesize - RepetitiveLength]
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                (i - j * len(TifArray[0])) * (imagesize - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (imagesize - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[imagesize - ColumnOver: imagesize, RepetitiveLength: imagesize - RepetitiveLength]
            else:
                result[j * (imagesize - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                        imagesize - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (imagesize - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (imagesize - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: imagesize - RepetitiveLength, RepetitiveLength: imagesize - RepetitiveLength]
    return result


def yolosam(image_data, x, y):
    
    save_path1 = r"/home/neaucs2/usr/xrx/yolo8-sam/test/3m/process" + '/' + str(x) + '_' + str(y) + '.tif'
    cv2.imwrite(save_path1.replace(".tif", ".jpg"), image_data)
    print(modelpath)
    results = model.predict(save_path1.replace(".tif", ".jpg"))
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    img_save = np.zeros((imagesize, imagesize, 1), np.uint8)
    bounding_boxes = results[0].boxes.xyxy
    annotated_frame = results[0].plot()
    # cv2.imshow(winname="YOLOV8", mat=annotated_frame)
    # cv2.waitKey(0)
    # cv2.imwrite(save_path1.replace(".tif", ".JPG"), annotated_frame)
    if results[0].masks != None:
        # bounding_boxes = []
        # bounding_boxes = torch.tensor(bounding_boxes, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(bounding_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)

        for index, mask in enumerate(masks):
            mask = mask.cpu().numpy()
            color = np.array([1])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            img_save = img_save + mask_image

        for i in range(0, imagesize):
            for j in range(0, imagesize):
                grayPixel = img_save[i, j]
                if grayPixel == 0:
                    # dst[i, j] = 255 - grayPixel
                    img_save[i, j] = 0
                else:
                    img_save[i, j] = 255

    save_path = r"/home/neaucs2/usr/xrx/yolo8-sam/test/3m/process" + '/' + str(x) + '_' + str(y) + '.tif'
    cv2.imwrite(save_path.replace(".tif", ".png"), img_save)
    img = cv2.imread(save_path.replace(".tif", ".png"), 0)
    # # 膨胀
    # kernel = np.zeros((3, 3), dtype=np.uint8)
    # dilate = cv2.dilate(img, kernel, 1)
    # img = dilate
    # 先利用二值化去除图片噪声
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    target_img_save = np.zeros((imagesize, imagesize, 1), np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # print(str(area))
        if area > 0:
            # 多边形拟合
            approx = cv2.approxPolyDP(contour, 3, True)

            # cv2.drawContours(target_img_save, [approx], -1, (0, 0, 0), 3)

            cv_contours.append(approx)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue

    cv2.fillPoly(target_img_save, cv_contours, (255, 255, 255))
    # # 腐蚀
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # ss = np.hstack((img, erosion))
    # img = ss
    cv2.imwrite(save_path, target_img_save)
    print(target_img_save.shape)
    target_img_save = np.squeeze(target_img_save)
    return target_img_save


#  获取当前时间
starttime = datetime.datetime.now()

testtime = []

RepetitiveLength = int((1 - math.sqrt(area_perc)) * imagesize / 2)

# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# path = "ESPCN_x4.pb"
# sr.readModel(path)
# sr.setModel("espcn", 4)
#
# #  记录测试消耗时间


im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(src_tif)
im_data = im_data.swapaxes(1, 0)
im_data = im_data.swapaxes(1, 2)
# im_data = im_data.swapaxes(0, 1)
# im_data = im_data.swapaxes(0, 2)
print("tif读取完毕")
print("im_width：" + str(im_width))
print("im_height：" + str(im_height))
TifArray, RowOver, ColumnOver = TifCroppingArray(im_data, RepetitiveLength)
endtime = datetime.datetime.now()
text = "读取tif并裁剪预处理完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
# image_size = 512

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predicts = []
current = 1
total = len(TifArray) * len(TifArray[0])
for i in range(len(TifArray)):
    for j in range(len(TifArray[0])):
        image = TifArray[i][j]
        img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        #
        print("当前:" + str(current) + "/" + str(total))
        current += 1
        img_cv_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        non_zero_ratio = sum(img_cv_gray[img_cv_gray != 0]) / len(img_cv_gray)

        if non_zero_ratio < 0.05:
            print('该图跳过')
            img_np = np.zeros((imagesize, imagesize)).astype(
                np.uint8)

        else:
            # img_np = yolosam(data=image)
            img_np = yolosam(img_cv, i, j)
        predicts.append((img_np))

# 保存结果
result_shape = (im_data.shape[0], im_data.shape[1])
print("result_shape：{}".format(result_shape))
result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)

writeTiff(result_data, im_geotrans, im_proj, outtif)
endtime = datetime.datetime.now()
text = "---结果拼接完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"

print(text)

outshp = outtif.replace(".tif", ".shp")
#
raster_to_vector(outtif, outshp)
# cal_area(outshp)

endtime = datetime.datetime.now()
# text = "面积计算完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
