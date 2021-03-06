# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:16:14 2018

@author: 15339
"""

from PIL import Image as image
import numpy as np
from KMeanspp import run_kmeanspp
#%%
def load_data(file_path):
    '''导入数据
    input:  file_path(string):文件的存储位置
    output: data(mat):数据
    '''
    f = open(file_path, "rb")  # 以二进制的方式打开图像文件
    data = []
    im = image.open(f)  # 导入图片
    m, n = im.size  # 得到图片的大小
    print m, n
    for i in xrange(m):
        for j in xrange(n):
            tmp = []
            x, y, z = im.getpixel((i, j))
            tmp.append(x / 256.0)
            tmp.append(y / 256.0)
            tmp.append(z / 256.0)
            data.append(tmp)
    f.close()
    return np.mat(data)

if __name__ == "__main__":
k = 10#聚类中心的个数
# 1、导入数据
print "---------- 1.load data ------------"
data = load_data("001.jpg")
# 2、利用kMeans++聚类
print "---------- 2.run kmeans++ ------------"
run_kmeanspp(data, k)