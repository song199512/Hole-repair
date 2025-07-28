import os
os.environ["OMP_NUM_THREADS"] = '1'
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import numpy as np
import math
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import laspy
import random
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
import pyransac3d as pyrsc
import copy
from pclpy import pcl


#读取pcd 数据
def ReadPcd(path):
    try:
        pcd = o3d.io.read_point_cloud(path)
        # points = np.asarray(pcd.points)
        return pcd
    except Exception as e:
        print(e)
#存储pcd数据
def WritePcd(path, points):
    '''
    description: 将点集写入pcd文件
    param {*} path 写入路径
    param {*} points 要写入的点
    return {*}
    '''
    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
    print('写入完成！')
def WritePcd1(path, points):
    '''
    description: 将点集写入pcd文件
    param {*} path 写入路径
    param {*} points 要写入的点
    return {*}
    '''
    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    o3d.io.write_point_cloud(path, pcd)

    print('写入完成！')
def WritePcd2(path, points, colors):
    '''
    description: 将点集写入pcd文件
    param {*} path 写入路径
    param {*} points 要写入的点
    return {*}
    '''
    points = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

    print('写入完成！')
def Ransac_3d(pcd, t=0.1, max=1500):
    """
    RANSAC拟合三维直线算法
    :param points: 数据点，每行一个点，共n行，列数为3
    :param threshold: 阈值，表示数据点到拟合直线的距离小于该值即为内点，默认为0.1
    :param max_iterations: 最大迭代次数，默认为1000
    :return: 直线参数，即直线上一点和方向向量，形式为[(x1, y1, z1), (a, b, c)]
    """
    points = np.asarray(pcd.points)
    line = pyrsc.Line()
    A, B, inliers = line.fit(points, t, max)
    return A,B,inliers
def project2line(pcd,D,P):
    '''
    :param pcd: 需要进行投影的点云数据
    :param D: 投影直线的方向向量
    :param P: 直线上一点
    :return: 返回pcd数据
    '''
    line_pt = np.array(P)  # 直线上一点
    line_dir = np.array(D)  # 直线轴向的单位法向量
    #点云投影
    line_dir = line_dir.reshape(1, 3)
    project_cloud = o3d.geometry.PointCloud(pcd)
    pt = np.asarray(project_cloud.points)
    k = (pt - line_pt) @ line_dir.T
    points = line_pt + k @ line_dir
    project_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud('kuamian_hole\\project2line.pcd', project_cloud)
    #WritePcd1('test3\\project2line.pcd', project_cloud)
    return project_cloud

#点云投影到平面
def point_cloud_plane_project(cloud, coefficients):
    """
    点云投影到平面
    :param cloud:输入点云
    :param coefficients: 待投影的平面
    :return: 投影后的点云
    """
    # 获取平面系数
    A = coefficients[0]
    B = coefficients[1]
    C = coefficients[2]
    D = coefficients[3]
    # 构建投影函数
    Xcoff = np.array([B * B + C * C, -A * B, -A * C])
    Ycoff = np.array([-B * A, A * A + C * C, -B * C])
    Zcoff = np.array([-A * C, -B * C, A * A + B * B])
    # 三维坐标执行投影
    points = np.asarray(cloud.points)
    xp = np.dot(points, Xcoff) - A * D
    yp = np.dot(points, Ycoff) - B * D
    zp = np.dot(points, Zcoff) - C * D
    project_points = np.c_[xp, yp, zp]  # 投影后的三维坐标
    project_cloud = o3d.geometry.PointCloud()  # 使用numpy生成点云
    project_cloud.points = o3d.utility.Vector3dVector(project_points)
    project_cloud.colors = pcd.colors  # 获取投影前对应的颜色赋值给投影后的点
    return project_cloud
#-----------------------------二次切片--------------------------------------------

#中心线提取
def CentersLine(pcd, pcd1,direct=True):
    '''
    description: 点云中心点生成，若是将slice函数改成slice_part函数就是指定一段点云数据的中心线生成
    param {*} points 点云集合
    param {*} direct 切片方向
    param {*} writepath 写入路径
    return {*} 中心点
    '''
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    flag, slicelists = Slice(points, direct)
    qiepian_data = []
    if pcd1:
        points1 = np.asarray(pcd1.points)
        flag, slicelists = SliceinLine(points, points1, colors, direct)  # 切片列表，这里可以对点云数据进行旋转，多个角度进行切割
        i = 0
        for slicelist1 in slicelists:
            normalpath = 'kuamian_hole\\qiepian_color\\'
            filename = f'qiepian_{i}.pcd'
            path = normalpath + str(filename)

            WritePcd1(path, slicelist1 )
            qiepian_data=[]
            i = i + 1
    else:
        flag, slicelists = Slice(points, direct)
        print('1')
    print('切片完成,长度如下')                        # 生成多个角度切片的中心点，之后将中心点序列进行合并
    print(len(slicelists) )
    #writer = pcl.io.PCDWriter()
    for n in range(0, len(slicelists)):     #完整去噪范围
    #for n in range(60, 70):  # 测试范围设置
        normalpath = 'kuamian_hole\\qiepian_color\\'
        filename = f'qiepian_{n}.pcd'
        path = normalpath + str(filename)
        pcd = o3d.io.read_point_cloud(path)
        plane_model, _ = pcd.segment_plane(distance_threshold=1,
                                           ransac_n=5,
                                           num_iterations=30)
        projected_cloud = point_cloud_plane_project(pcd, plane_model)  #将切片拟合为二维平面
        normalpath_p = 'kuamian_hole\\sources_color\\'
        filename_p = f'source_{n}.pcd'
        path_p = normalpath_p + str(filename_p)
        o3d.io.write_point_cloud(path_p, projected_cloud)

    return
#中心点生成
def Slice(points, direction=True, scale=3, interval=3):
    slicelists = []  # 返回数组
    try:
        # 打开文件
        ''' pcd = o3d.io.read_point_cloud(sourcepath)
        points = np.asarray(pcd.points) '''
        # 判断点云总体走向
        miny = int(points[:, 1].min())  # y坐标最小值
        maxy = int(points[:, 1].max())  # y坐标最大值
        minx = int(points[:, 0].min())  # x坐标最小值
        maxx = int(points[:, 0].max())  # x坐标最大值
        max = 0
        min = 0
        flag = 0  # 走向标志
        if(maxx - minx) <= (maxy - miny):
            max = maxy
            min = miny
            flag = 0  # 0代表y走向
        else:
            max = maxx
            min = minx
            flag = 1  # 1代表x走向
        if direction == False:  # 改变方向
            flag = (flag + 1) % 2  # 正交方向切割
            if(max == maxx):
                max = maxy
                min = miny
            else:
                max = maxx
                min = minx
        slicelist = []  # 切片名称列表
        length = (max - min)*scale*2  # 迭代次数
        for num in range(0, length):  # 初始化数组
            array = []
            slicelist.append(array)
        print("数组初始化完成！！！")
        k = 0
        # 迭代文件块
        if(flag == 0):
            listpoints = points[:, 1]
        else:
            listpoints = points[:, 0]
        for i in range(0, len(listpoints)):
            local = int((listpoints[i]-min)*scale)  # 切片编号
            if(local % interval == 0):
                #local = int((listpoints[i]-min)*(scale/2))
                slicelist[local].append(points[i])
        for i in range(len(slicelist)):
            if len(slicelist[i]) != 0:
                slicelists.append(slicelist[i])
        return flag, slicelists  # 返回切片列表

    except Exception as e:
        print(e)

def SliceinLine(points, points1,colors, direction=True, scale=35, interval=1):
    print("开始执行二次切片")
    '''
    description: 切片算法
    param {*} points 点云数据的点集集合
    param {*} points1 投影到直线的点云数据用于切分获取索引
    param {*} direction 切片的方向，沿点云数据主方向direction为True,与主方向垂直方向为False
    param {*} scale 切片步长系数，与切片步长成反比关系
    return {*} flag, slicelists flag为切片的方向，flag为1为x方向，0为y方向
    '''
    slicelists = []  # 返回数组
    try:
        # 打开文件
        ''' pcd = o3d.io.read_point_cloud(sourcepath)
        points = np.asarray(pcd.points) '''
        # 判断点云总体走向
        miny = int(points1[:, 1].min())  # y坐标最小值
        maxy = int(points1[:, 1].max())  # y坐标最大值
        minx = int(points1[:, 0].min())  # x坐标最小值
        maxx = int(points1[:, 0].max())  # x坐标最大值
        max = 0
        min = 0
        flag = 0  # 走向标志
        if(maxx - minx) <= (maxy - miny):
            max = maxy
            min = miny
            flag = 0  # 0代表y走向
        else:
            max = maxx
            min = minx
            flag = 1  # 1代表x走向
        if direction == False:  # 改变方向
            flag = (flag + 1) % 2  # 正交方向切割
            if(max == maxx):
                max = maxy
                min = miny
            else:
                max = maxx
                min = minx
        slicelist = []  # 切片名称列表
        length = (max - min)*scale*2  # 迭代次数
        for num in range(0, length):  # 初始化数组
            array = []
            slicelist.append(array)
        print("数组初始化完成！！！")
        k = 0
        # 迭代文件块
        if(flag == 0):
            listpoints = points1[:, 1]
        else:
            listpoints = points1[:, 0]

        for i in range(0, len(listpoints)):
            local = int((listpoints[i]-min)*scale)  # 切片编号
            if(local % interval == 0):  #-------------------------------------0,1两次切片改为了无间隔一次切片
                #local = int((listpoints[i]-min)*(scale/2))
                xyz_color = np.hstack((points[i], colors[i]))
                slicelist[local].append(xyz_color)
        for i in range(len(slicelist)):
            if len(slicelist[i]) != 0:
                slicelists.append(slicelist[i])
        return flag, slicelists  # 返回切片列表
    except Exception as e:
        print(e)

#dbscan聚类
def Cluster_Slice(slice, radius=0.25, min_points=5):
    '''
    description: 对切片聚类分类算法，dbscan算法会给每个点打上一个正整数标签，同一类的点的标签一样
    -1代表噪声点
    param {*} slice 切片点云集合
    param {*} radius 扫描半径
    param {*} min_samples 最小样本点数
    return {*} clusters_points 子切片序列
    '''
    slice = np.asarray(slice)
    clusters = DBSCAN(eps=radius, min_samples=min_points, n_jobs=-
                      1).fit_predict(slice)  # dbscan聚类 邻域范围 最小点数
    clusters_points = []
    nums = clusters.max()+1  # 所得聚类个数
    for i in range(0, nums):
        a = []
        clusters_points.append(a)
    # 获取子切片序列
    for i in range(0, len(clusters)):
        if(clusters[i] >= 0):
            clusters_points[clusters[i]].append(slice[i])
    return clusters_points


#二维平面点集中心，两种计算方法，center_1以点集第一个点为基点，center_2以点集均值为中心点
def center_1(Listpoints):
    '''
    description: 二维平面点集中心，两种计算方法，center_1以点集第一个点为基点，center_2以点集均值为中心点
    param {*} Listpoints
    return {*} center 返回二维平面点集中心点
    '''
    try:
        # 第一个点基准点
        y_ = Listpoints[0][0]
        z_ = Listpoints[0][1]
        Listpoints = np.vstack((Listpoints, Listpoints[0])) #按垂直方向（行顺序）堆叠数组构成一个新的数组
        area = 0
        yarea = 0
        zarea = 0
        for i in range(1, len(Listpoints)-2):
            unit_area = math.fabs((Listpoints[i][0]-y_) * (Listpoints[i+1][1]-z_) - (
                Listpoints[i][1]-z_) * (Listpoints[i+1][0]-y_)) / 2
            area = area + unit_area
            xy = Tricenter([Listpoints[i], Listpoints[i+1], [y_, z_]])#返回三角形圆心坐标
            yarea = yarea + xy[0]*unit_area
            zarea = zarea + xy[1]*unit_area
        return [yarea/(3*area), zarea/(3*area)]
    except Exception as e:
        print(e)

#返回三角形圆心坐标
def Tricenter(trilist):
    '''
    description: 返回三角形圆心坐标
    param {*} trilist
    return {*} center 返回三角形圆心坐标
    '''
    try:
        x = 0
        y = 0
        for point in trilist:
            x = x + point[0]
            y = y + point[1]
        return [x, y]
    except Exception as e:
        print(e)

def distancePToL(point,line1,line2):
    p = np.asarray(point)
    a = np.asarray(line1)
    b = np.asarray(line2)
    ab = b - a
    ap = p - a
    bp = p - b
    # 计算投影长度，并做正则化处理
    r = np.dot(ap, ab) / (np.linalg.norm(ab)) ** 2
    # 分了三种情况
    if r > 0 and r < 1:
        dis = math.sqrt((np.linalg.norm(ap)) ** 2 - (r * np.linalg.norm(ab)) ** 2)
    elif r >= 1:
        dis = np.linalg.norm(bp)
    else:
        dis = np.linalg.norm(ap)
    return dis

if __name__ == '__main__':
    start = time.time()
    pcd = ReadPcd('kuamian_hole\\kuamian.pcd')
    center_points1 = ReadPcd('kuamian_hole\\kuamian_center.pcd')
    D,P,inp = Ransac_3d(center_points1)
    pcd1 = project2line(pcd,D,P)
    CentersLine(pcd, pcd1)
    end = time.time()
    print(str(end - start))