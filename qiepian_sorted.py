import time

import open3d as o3d
import numpy as np
import os

def count_files_in_folder(folder_path):
    # 获取文件夹中的文件列表
    files = os.listdir(folder_path)

    # 统计文件的数量
    file_count = len(files)

    return file_count
def extract_boundary_points(point_cloud):
    point_clouds = np.asarray(point_cloud.points)
    pcds = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # 计算点云的凸壳
    hull, index = point_cloud.compute_convex_hull()

    # 获取凸壳的顶点
    hull_points = np.asarray(hull.vertices)

    # 计算凸包顶点的中心
    center = np.mean(hull_points, axis=0)

    # 计算每个顶点相对于中心的极坐标角度
    angles = np.arctan2(point_clouds[:, 1] - center[1], point_clouds[:, 0] - center[0])

    # 按照极坐标角度排序凸包的顶点
    sorted_indices = np.argsort(angles)
    sorted_hull_points = pcds[sorted_indices]
    sorted_hull_colors = colors[sorted_indices]
    sorted_points = np.hstack([sorted_hull_points,sorted_hull_colors])
    # 将凸壳的顶点按顺序连接形成线
    # lines = []
    # for i in range(len(sorted_hull_points)):
    #     lines.append([point_clouds[i], point_clouds[(i + 1) % len(point_clouds)]])

    #return lines,sorted_hull_colors,sorted_points
    return sorted_hull_points, sorted_points
def convex_hull(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)

    # 提取孔洞边界的线段
    sorted_hull_points, sorted_points = extract_boundary_points(point_cloud)
    lines = [[i, (i + 1) % len(sorted_hull_points)] for i in range(len(sorted_hull_points))]
    # 创建LineSet对象
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(sorted_hull_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set,sorted_points

if __name__ == "__main__":
    start = time.time()
    file_num = count_files_in_folder('kuamian_hole\\sources_color\\')
    for i in range(0, file_num):
        normalpath = 'kuamian_hole\\sources_color\\'
        filename = f'source_{i}.pcd'
        file_path = normalpath + str(filename)
        normalpath2 = 'kuamian_hole\\qiepian_color\\'
        filename2 = f'qiepian_{i}.pcd'
        pcd_path = normalpath2 + str(filename2)
        pcd = o3d.io.read_point_cloud(pcd_path)
        line_set, sorted_points = convex_hull(file_path)
        normalpath3 = 'kuamian_hole\\qiepiancolor_sorted\\'
        filename3 = f'qiepian{i}_sorted.txt'
        save_path = normalpath3 + str(filename3)
        np.savetxt(save_path, sorted_points)
        #o3d.visualization.draw_geometries([line_set])
        normalpath3 = 'kuamian_hole\\hulls_color\\'
        filename3 = f'boundary_lines{i}.ply'
        file_path3 = normalpath3 + str(filename3)
        o3d.io.write_line_set(file_path3, line_set)

    end = time.time()
    print(str(end - start))

    # file_path = "process_data\\source_182_voxcel.pcd"
    # pcd = o3d.io.read_point_cloud("process_data\\qiepian_182_voxcel.pcd")
    # file_path = "D:\\pythonProject\\ism\\xiadian\\sources0\\source_182.pcd"
    # pcd = o3d.io.read_point_cloud("D:\\pythonProject\\ism\\xiadian\qiepian0\\qiepian_182.pcd")
    # sorted_hull_points =  extract_boundary_points(point_cloud)
    # np.savetxt("process_data\\qiepian182_sorted.txt",sorted_hull_points)
    # o3d.io.write_point_cloud("process_data\\qiepian182_sorted.pcd", sorted_hull_points)
    # folder_path = "./dbscan"
    # line_set, sorted_hull_points = convex_hull(file_path)
    # o3d.visualization.draw_geometries([line_set])
    # np.savetxt("process_data\\qiepian182_sorted.txt", sorted_hull_points)
    # normalpath2 = 'hulls\\'
    # filename2 = f'boundary_lines{i}.ply'
    # file_path2 = normalpath2 + str(filename2)
    # o3d.io.write_line_set(file_path2, line_set)

