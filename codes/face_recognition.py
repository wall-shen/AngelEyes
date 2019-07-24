# -*- coding: UTF-8 -*-
import cv2 as cv
import os
from face_pretreate import endwith, pretreat_img_internet, pretreat_img_chinese
import sklearn.metrics.pairwise as pw
import torch
import numpy as np
from torch.legacy import nn
from numpy import loadtxt
from fileConfig import *
from face_alignment import warpAffilneByAngle


# 从处理过的图像集中获取特征
def save_all_feature(fileName, targetName, *suffix):
    # 文件存在
    if os.path.exists(fileName):

        # 加载vgg模型
        vgg_net = torch.load(VGG_FACE_LOCATION)
        vgg_net.modules[31] = nn.View(1, 25088)

        # 读取一级目录
        f = os.listdir(fileName)
        for name in f:
            # 读取二级目录
            images = os.listdir(fileName + os.sep + name)
            for image in images:
                if endwith(image, suffix):  # 后缀名判断
                    # 文件读取，获取特征
                    img = cv.imread(fileName + os.sep + name + os.sep + image)
                    feature = list(get_feature(img, vgg_net))
                    # print(feature)

                    # 创建文件
                    if not os.path.exists(targetName):
                        os.makedirs(targetName)

                    # 保存标签文件
                    with open(targetName + os.sep + 'id.dat', 'a') as f:
                        f.write(name)
                        f.write('\n')

                    # 保存特征文件
                    with open(targetName + os.sep + 'feature.dat', 'a') as f:
                        feature = map(str, feature)
                        f.writelines(s+' ' for s in feature)
                        f.write('\n')
                    print(name, " successful")


#存储单个照片特征
def save_feature(fileName, targetName, *suffix):
    if os.path.exists(fileName):
        name = os.path.basename(fileName)
        # 加载文件
        vgg_net = torch.load(VGG_FACE_LOCATION)
        vgg_net.modules[31] = nn.View(1, 25088)
        images = os.listdir(fileName)
        for image in images:
            if endwith(image, suffix):
                img = cv.imread(fileName + os.sep + image)
                feature = list(get_feature(img, vgg_net))
                # print(feature)
                if not os.path.exists(targetName):
                    os.makedirs(targetName)
                # 保存标签文件
                with open(targetName + os.sep + 'id.dat', 'a') as f:
                    f.write(name)
                    f.write('\n')
                # 保存特征文件
                with open(targetName + os.sep + 'feature.dat', 'a') as f:
                    feature = map(str, feature)
                    f.writelines(s + ' ' for s in feature)
                    f.write('\n')
                print(name, " successful")


# 获取图片的矩阵形式，输出特征
def get_feature(file, net):
    img = np.zeros((3, 224, 224))

    # 将224*224*3矩阵转换3*244*244矩阵
    img[0, :, :] = (file[:, :, 2])
    img[1, :, :] = (file[:, :, 1])
    img[2, :, :] = (file[:, :, 0])

    # print(img)
    img = img.astype(float)
    trainingMean = [129.1863, 104.7624, 93.5940]
    for i in range(3):
        img[0, i] = img[0, i] - trainingMean[i]  # 归一化

    # 封装为张量
    input = torch.Tensor(1, 3, 224, 224)
    input[0] = torch.from_numpy(img)
    # 正向传播
    output = net.forward(input)
    output = net.modules[35].output.clone().numpy()
    return output[0]

#比较2张图片的相似度
def compare(feat1, feat2):
    feat1 = feat1.reshape(1, -1)
    feat2 = feat2.reshape(1, -1)
    prediction = pw.cosine_similarity(feat1, feat2)
    return prediction

"""
接收的图片地址，
返回图片对应的id号，
如果返回为fail说明数据库中不存在匹配的id
"""
def compare_feature(fileName, vgg_net, feature):
    # print(feature)
    img, ret = pretreat_img_internet(fileName)  # 图片预处理
    sim = 0  # 相似度
    num = -1  # 特征位置
    count = 0  # 标签位置
    if ret == 1:  # 普片预处理成功
        # 获取特征
        img_feature = get_feature(img, vgg_net)
        # 寻找最高相似度
        for i in range(feature.shape[0]):
            comp = compare(img_feature, feature[i])
            if comp > sim:
                num = i
                sim = comp
            if sim > 0.7:  # 相似度足够高，停止寻找
                break
        if not num == -1:
            if sim > 0.37:  # 相似度足够时，返回结果
                with open(ID_LOCATION) as f:
                    for line in f.readlines():
                        if num == count:
                            return line[:-1], 2, sim
                        else:
                            count += 1
            else:
                return img, 1, sim
        else:
            return img, 1, sim
    else:
        return img, 0, sim

def findFeatureZeroCount(fileName):
    img = pretreat_img_chinese(fileName)
    vgg_net = torch.load(VGG_FACE_LOCATION)
    vgg_net.modules[31] = nn.View(1, 25088)
    feature = get_feature(img, vgg_net)
    count = 0
    for i in feature:
        if i == 0:
            count += 1
    print(count)

#比较2个图片
def compare_two_img(img1, img2):
    vgg_net = torch.load(VGG_FACE_LOCATION)
    vgg_net.modules[31] = nn.View(1, 25088)
    feature1 = get_feature(img1, vgg_net)
    feature2 = get_feature(img2, vgg_net)
    result = compare(feature1, feature2)
    return result

#确定阈值
def simdrop_threshold_test(filename):
    if os.path.exists(filename):
        images = os.listdir(filename)
        sum_count = len(images)
        test_point = np.arange(0.2, 0.7, 0.1).astype('float32')
        accuracy = np.arange(0.2, 0.7, 0.1).astype('float32')
        vgg_net = torch.load(VGG_FACE_LOCATION)
        print('模型加载成功')
        vgg_net.modules[31] = nn.View(1, 25088)
        feature = loadtxt(FEATURE_LOCATION)
        print('feature加载成功')
        for w in range(test_point.shape[0]):
            right_count = 0
            for image in images:
                num = -1
                sim = -1
                y = os.path.splitext(image)[0]
                img = cv.imread(filename + os.path.sep + image)
                image_feature = get_feature(img, vgg_net)
                # 找到对应的id在第几行
                for i in range(feature.shape[0]):
                    comp = compare(feature[i].reshape(1, -1), image_feature.reshape(1, -1))
                    if comp > sim:
                        num = i
                        sim = comp
                if sim >= test_point[w]:
                    judge = 1
                else:
                    judge = 0

                # 判断找到的id和正确id是否相同
                if not num == -1:
                    count = 0
                    with open(ID_LOCATION) as f:
                        for line in f.readlines():
                            if num == count:
                                if line[:-1] == y:
                                    if judge == 1:
                                        right_count += 1
                                        print(test_point[w], ' | ', line[:-1], ' | ', y, ' right ', sim)
                                        break
                                    else:
                                        print(test_point[w], ' | ', line[:-1], ' | ', y, ' wrong ', sim)
                                        break
                                else:
                                    if judge == 0:
                                        right_count += 1
                                        print(test_point[w], ' | ', line[:-1], ' | ', y, ' right ', sim)
                                        break
                                    else:
                                        print(test_point[w], ' | ', line[:-1], ' | ', y, ' wrong ', sim)
                                        break

                            count += 1
                else:
                    print(test_point[w], ' | ', image, ' | ', y, ' not found')
            print(right_count, ' ', sum_count)
            accuracy[w] = right_count / sum_count
            print(test_point[w], ' | ', accuracy[w])

        for i in range(accuracy.shape[0]):
            print('threshold: ', i+1, ' | accuracy:', accuracy[i])

#确定阈值
def simbreak_threshold_test(filename):
    if os.path.exists(filename):
        images = os.listdir(filename)
        sum_count = len(images)
        test_point = np.arange(0.3, 0.8, 0.1).astype('float32')
        accuracy = np.arange(0.3, 0.8, 0.1).astype('float32')
        vgg_net = torch.load(VGG_FACE_LOCATION)
        print('模型加载成功')
        vgg_net.modules[31] = nn.View(1, 25088)
        feature = loadtxt(FEATURE_LOCATION)
        print('feature加载成功')
        for w in range(test_point.shape[0]):
            right_count = 0
            for image in images:
                num = -1
                sim = -1
                y = os.path.splitext(image)[0]
                img = cv.imread(filename + os.path.sep + image)
                image_feature = get_feature(img, vgg_net)
                # 找到对应的id在第几行
                for i in range(feature.shape[0]):
                    comp = compare(feature[i].reshape(1, -1), image_feature.reshape(1, -1))
                    if comp > sim:
                        num = i
                        sim = comp
                    if sim >= test_point[w]:
                        break

                # 判断找到的id和正确id是否相同
                if not num == -1:
                    count = 0
                    with open(ID_LOCATION) as f:
                        for line in f.readlines():
                            if num == count:
                                if line[:-1] == y:
                                    right_count += 1
                                    print(test_point[w], ' | ', line[:-1], ' | ', y, ' right ', sim)
                                    break
                                else:
                                    print(test_point[w], ' | ', line[:-1], ' | ', y, ' wrong ', sim)
                                    break
                            count += 1
                else:
                    print(test_point[w], ' | ', image, ' | ', y, ' not found')
            print(right_count, ' ', sum_count)
            accuracy[w] = right_count / sum_count
            print(test_point[w], ' | ', accuracy[w])

        for i in range(accuracy.shape[0]):
            print('threshold: ', i+1, ' | accuracy:', accuracy[i])

if __name__ == '__main__':

    save_all_feature('/media/wall/F6F4D817F4D7D7C7/image/20180326_test/image_face', '/media/wall/F6F4D817F4D7D7C7/image/f_test', '.jpg')

    # simbreak_threshold_test('/media/wall/F6F4D817F4D7D7C7/image_test')
