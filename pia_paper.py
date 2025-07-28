import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from b_spline_basis import b_spline_basis
from bspline_basis import bspline_basis
from plot_bspline import plot_bspline
import open3d as o3d
import os
import scipy.spatial as spt
import numpy as np

#根据三维点云使用追赶法反求三次B样条控制点
def LU_B1(CPnum, V):
    a, b, c, d, e = 1, 4, 1, 1, 1

    f = np.zeros(CPnum - 1)
    g = np.zeros(CPnum - 2)
    h = np.zeros(CPnum)
    k = np.zeros(CPnum - 1)

    # get h[] & f[]
    h[0] = b
    for i in range(CPnum - 1):
        f[i] = a / h[i]
        h[i + 1] = b - f[i] * c

    # get g[] & f[n-1]
    g[0] = d / h[0]
    for i in range(CPnum - 3):
        g[i + 1] = -g[i] * c / h[i + 1]
    f[CPnum - 2] = (a - g[CPnum - 3] * c) / h[CPnum - 2]

    # get k[] & h[n]
    k[0] = e
    for i in range(CPnum - 2):
        k[i + 1] = -f[i] * k[i]
    k[CPnum - 2] = c - f[CPnum - 3] * k[CPnum - 3]

    # gksum = 0
    # for i in range(CPnum - 2):
    #     gksum = gksum + g(i) * k(i)
    gksum = np.sum(g * k[:CPnum - 2])
    h[CPnum - 1] = b - gksum - f[CPnum - 2] * c

    # forward elimination矩阵求解过程，追的过程
    x = np.zeros(CPnum)
    x[0] = 6 * V[-1]

    for i in range(CPnum - 2):
        x[i + 1] = 6 * V[i] - f[i] * x[i]

    # gxsum = 0
    # for i in range(CPnum - 2):
    #     gksum = gksum + g(i) * k(i)
    gxsum = np.sum(g * x[:CPnum - 2])
    x[CPnum - 1] = 6 * V[CPnum - 2] - gxsum - f[CPnum - 2] * x[CPnum - 2]

    # backward substitution赶的过程
    px = np.zeros(CPnum + 2)
    px[CPnum-1] = x[CPnum-1] / h[CPnum-1]
    px[CPnum - 2] = (x[CPnum - 2] - k[CPnum - 2] * px[CPnum-1]) / h[CPnum - 2]

    for i in range(CPnum - 3, -1, -1):
        px[i] = (x[i] - c * px[i + 1] - k[i] * px[CPnum-1]) / h[i]

    px[CPnum] = px[0]
    px[CPnum + 1] = px[1]

    return px
#PIA渐进迭代逼近算法优化控制点
def PIA(px,py,pz,bs):
    x_cps = px
    y_cps = py
    z_cps = pz

    # Then we have a difference between the target values and the corresponding values of our B-spline.
    dx = px - np.transpose(np.dot(np.transpose(bs), x_cps))
    dy = py - np.transpose(np.dot(np.transpose(bs), y_cps))
    dz = pz - np.transpose(np.dot(np.transpose(bs), z_cps))

    # norm is going to store the norm of (dx, dy)范数将存储（dx，dy）的范数
    norm = np.zeros(n)
    for i in range(len(dx)):
        norm[i] = math.sqrt(dx[i] ** 2 + dy[i] ** 2 + dz[i] ** 2)

    # make the biggest norm to be the error
    err = max(norm)

    iteration = 0
    print('iteration #', iteration, ', err = ', err)

    # set the threshold for the algorithm to stop设置算法停止的阈值
    #tol = 1e-5
    tol = 0.001*err

    # in while loop, calculate the difference in each loop, until error is smaller than the threshold
    while err > tol and iteration < 100:
        iteration = iteration + 1
        # We change the control points ...
        x_cps = x_cps + dx
        y_cps = y_cps + dy
        z_cps = z_cps + dz
        # ... and compute the difference from the target (which is constant)!
        dx = px - np.transpose(np.dot(np.transpose(bs), x_cps))
        dy = py - np.transpose(np.dot(np.transpose(bs), y_cps))
        dz = pz - np.transpose(np.dot(np.transpose(bs), z_cps))
        for i in range(len(dx)):
            norm[i] = math.sqrt(dx[i] ** 2 + dy[i] ** 2 + dz[i] ** 2)
        err = max(norm)
        print('iteration #', iteration, ', err = ', err)

    x_inter = np.transpose(np.dot(np.transpose(bs), x_cps))
    y_inter = np.transpose(np.dot(np.transpose(bs), y_cps))
    z_inter = np.transpose(np.dot(np.transpose(bs), z_cps))

    return x_inter, y_inter, z_inter

def cubic_bsp(u, Xi):
    """Cubic B-spline basis function"""
    return (1/6) * ((1-u)**3 * Xi[0] + (3*u**3 - 6*u**2 + 4) * Xi[1] +
                    (-3*u**3 + 3*u**2 + 3*u + 1) * Xi[2] + u**3 * Xi[3])
def count_files_in_folder(folder_path):
    # 获取文件夹中的文件列表
    files = os.listdir(folder_path)

    # 统计文件的数量
    file_count = len(files)

    return file_count
if __name__ == '__main__':
    normalpath = 'qiepian_compare\\'
    filename = f'qiepian_91sorted_Cloud.txt'
    file_path = normalpath + str(filename)
    points = np.loadtxt(file_path)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    colors_r = points[:, 3]
    colors_g = points[:, 4]
    colors_b = points[:, 5]
    rgb = points[:, 3:6]
    point_nocolor = points[:, :3]

    CPnum = len(x)

    # 反求控制点
    px = LU_B1(CPnum, x)
    py = LU_B1(CPnum, y)
    pz = LU_B1(CPnum, z)
    xyz_combined = np.column_stack((px, py, pz))
    np.savetxt("qiepian_compare\\qiepian_91sorted_Cloud_combined.txt", xyz_combined)

    # PIA渐进迭代逼近算法优化控制点
    degree = 3
    n = len(px)
    delta = np.linspace(0, 2 * np.pi, num=n)  # 在指定的起始点和结束点之间生成等间隔的数值序列
    a = min(delta)
    b = max(delta)
    knotVector = [a, a, a, a, *delta[2:-2], b, b, b, b]
    bs = bspline_basis(n, degree, knotVector, delta)  # basis matrix is stored in bs基矩阵存储在bs中
    x_inter, y_inter, z_inter = PIA(px, py, pz, bs)
    xyz_inters = np.column_stack((x_inter, y_inter, z_inter))
    # np.savetxt("sd_mine\\test_result\\xyz_inters.txt", xyz_inters)
    # 首尾相连，曲线闭合
    # 要添加的值
    x_add = x[0]
    y_add = y[0]
    z_add = z[0]
    rgb_add = rgb[0]
    # 在数组末尾增加值
    x = np.append(x, x_add)
    y = np.append(y, y_add)
    z = np.append(z, z_add)

    rgb = np.vstack((rgb, rgb_add))
    px_add = x_inter[2]
    py_add = y_inter[2]
    pz_add = z_inter[2]
    x_inter = np.append(x_inter, px_add)
    y_inter = np.append(y_inter, py_add)
    z_inter = np.append(z_inter, pz_add)
    px_add2 = x_inter[3]
    py_add2 = y_inter[3]
    pz_add2 = z_inter[3]
    x_inter = np.append(x_inter, px_add2)
    y_inter = np.append(y_inter, py_add2)
    z_inter = np.append(z_inter, pz_add2)

    # 两点之间对10个点进行插值计算
    # h = 10
    # delta = 1.0 / h
    xx = x.tolist()
    yy = y.tolist()
    zz = z.tolist()
    for j in range(CPnum):
        u = 0.0
        #dist = np.sqrt(((px[j + 2] - px[j + 1]) ** 2 + (py[j + 2] - py[j + 1]) ** 2 + (pz[j + 2] - pz[j + 1]) ** 2))
        #dist = np.sqrt(((x_inter[j] - x_inter[j+4]) ** 2 + (y_inter[j] - y_inter[j+4]) ** 2 + (z_inter[j] - z_inter[j+4]) ** 2))
        dist = np.sqrt(((x[j+1] - x[j]) ** 2 + (y[j+1] - y[j]) ** 2 + (z[j+1] - z[j]) ** 2))
        nP = int(np.ceil(dist / 0.03))
        delta = 1.0 / nP
        if nP > 1:
            for m in range(nP):
                Xi = x_inter[j:j + 4]
                Yi = y_inter[j:j + 4]
                Zi = z_inter[j:j + 4]
                x_value = cubic_bsp(u, Xi)
                y_value = cubic_bsp(u, Yi)
                z_value = cubic_bsp(u, Zi)
                point = np.column_stack((x_value, y_value, z_value))
                if point not in point_nocolor:
                    kt = spt.KDTree(data=point_nocolor, leafsize=10)  # 用于快速查找的KDTree类
                    d, index = kt.query(point, k=5)  # 返回最近邻点的距离d和在数组中的顺序x
                    find_points = points[index]
                    color_rgb = np.mean(find_points[0], axis=0)

                    xx.append(x_value)
                    yy.append(y_value)
                    zz.append(z_value)
                    rgb = np.vstack((rgb, color_rgb[-3:]))
                m = m + 1
                u += delta
    xx2 = np.array(xx)
    yy2 = np.array(yy)
    zz2 = np.array(zz)
    xxyyzz = np.column_stack((xx2, yy2, zz2))
    np.savetxt("qiepian_compare\\qiepian_91sorted_combined.txt", xxyyzz)

    xyz_points = np.hstack([xxyyzz, rgb])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyz_points[:, 3:6])
    normalpath = 'qiepian_compare\\'
    filename = f'qiepian_91sorted_Cloud_fill.pcd'
    write_path = normalpath + str(filename)
    o3d.io.write_point_cloud(write_path, pcd)
    # o3d.visualization.draw_geometries([pcd])
