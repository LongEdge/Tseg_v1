from multiprocessing import Pool

import os

import json

import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
import open3d as o3d
from tqdm import tqdm


def read_obj_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # 仅处理顶点行
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])  # 提取 x,y,z 坐标
                vertices.append([x,  y, z])
    return np.array(vertices)

def compute_gaussian_curvature(points, normals, radius=0.1):
    """
    计算点云中每个点的高斯曲率（基于局部二次曲面拟合）

    参数:
        points: numpy 数组，形状为 (N, 3)，表示点云中的点
        normals: numpy 数组，形状为 (N, 3)，表示每个点的法向量
        radius: 浮点数，用于确定邻域范围的半径

    返回:
        gaussian_curvatures: numpy 数组，形状为 (N,)，每个点的高斯曲率
    """
    tree = KDTree(points)
    N = points.shape[0]
    gaussian_curvatures = np.zeros(N)


    for i in range(N):
        # 获取邻域点索引
        indices = tree.query_radius(points[i].reshape(1, -1), r=radius)[0]
        neighborhood_points = points[indices]

        if len(neighborhood_points) < 6:
            # 至少需要6个点才能拟合二次曲面
            continue

        # 构建局部坐标系
        normal = normals[i] / np.linalg.norm(normals[i])
        tangent1 = np.random.rand(3)
        tangent1 -= np.dot(tangent1, normal) * normal
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)

        # 将邻域点投影到局部坐标系
        local_coords = neighborhood_points - points[i]
        u = np.dot(local_coords, tangent1)
        v = np.dot(local_coords, tangent2)
        w = np.dot(local_coords, normal)

        # 构造设计矩阵 A 用于拟合 z = ax^2 + bxy + cy^2 + dx + ey + f
        A = np.column_stack([u**2, u*v, v**2, u, v, np.ones_like(u)])
        b = w

        # 最小二乘求解系数
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        a, b_coeff, c, d, e, f = coeffs

        # 高斯曲率 K = (4ac - b²) / (1 + d² + e²)^2
        K = (4 * a * c - b_coeff**2) / ((1 + d**2 + e**2) ** 2)

        gaussian_curvatures[i] = K


    return gaussian_curvatures


def compute_point_curvature_new(points, normals, radius=0.1):
    """
    计算点云中每个点的新定义的点“曲率”

    参数:
    points (numpy.ndarray):  点云数据，形状为 (n_points, 3)，其中 n_points 是点的数量
    normals (numpy.ndarray):  点云的法向量，形状为 (n_points, 3)
    radius (float): 邻域半径，默认为 0.1

    返回:
    numpy.ndarray:  每个点的曲率，形状为 (n_points,)
    """
    n_points = points.shape[0]
    curvatures = np.zeros(n_points)

    # 使用 sklearn 的 NearestNeighbors 来查找每个点的邻域
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(points)

    for i in range(n_points):
        # 查找当前点的邻域点
        distances, indices = nbrs.radius_neighbors([points[i]], return_distance=True)
        neighbor_indices = indices[0]

        if len(neighbor_indices) > 1:
            # 获取当前点的法向量
            current_normal = normals[i]

            # 获取邻域点的法向量
            neighbor_normals = normals[neighbor_indices]

            # 计算当前点法向量与邻域点法向量的夹角余弦值
            cos_angles = np.dot(neighbor_normals, current_normal)

            # 计算曲率，这里定义为夹角余弦值的平均值
            curvature = np.mean(cos_angles)
            curvatures[i] = curvature

    return curvatures

def pc_normalize(pc):
    centroid = np.mean(pc,axis=0)
    pc=pc - centroid
    m=np.max(np.sqrt(np.sum(pc**2,axis=1)))
    return pc/m

def number_covert(original):
    num_map = {
        31:1,
        32:2,
        33:3,
        34:4,
        35:5,
        36:6,
        37:7,
        38:8,
        41:9,
        42:10,
        43:11,
        44:12,
        45:13,
        46:14,
        47:15,
        48:16,
        11:17,
        12:18,
        13:19,
        14:20,
        15:21,
        16:22,
        17:23,
        18:24,
        21:25,
        22:26,
        23:27,
        24:28,
        25:29,
        26:30,
        27:31,
        28:32
    }
    new_list = [num_map.get(x,  x) for x in original]
    return new_list


def process_single_file(args):
    input_file_path, output_dir, file_idx = args

    vertices = []
    normals = []
    labels = []

    with open(input_file_path, 'r') as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertex = parts[1:4]
                normal = parts[4:7]
                label = parts[7:8]

                vertices.append(vertex)
                normals.append(normal)
                labels.append(label)

    # 数据处理部分保持不变
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    labels = [int(float(x[0])) for x in labels]
    vertices = pc_normalize(np.array(vertices))  # (N,3)
    normals = pc_normalize(np.array(normals))  # (N,3)
    labels = number_covert(labels)  # (N)
    labels = np.array(labels).reshape(-1, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.5, max_nn=30))
    normals = np.asarray(pcd.normals)

    gaussian_curvatures = compute_gaussian_curvature(vertices, normals)
    curvature_array = compute_point_curvature_new(vertices, normals)

    features = np.concatenate((
        vertices,
        normals,
        gaussian_curvatures.reshape(-1, 1),
        curvature_array.reshape(-1, 1)
    ), axis=1)

    category = np.tile((0, 1), (vertices.shape[0], 1))

    result = {
        "feature": features.tolist(),
        "label": labels.tolist(),
        "category": category.tolist()
    }

    filename = f"{file_idx}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(result, f)

    return filename

def process(inputPath, outputPath, pool_size=4):
    """
    多进程版本入口函数
    """
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    file_names = os.listdir(inputPath)
    tasks = []
    idx = 0

    for filename in tqdm(file_names, desc="Preparing files"):
        if filename == "log.txt":
            continue
        file_path = os.path.join(inputPath, filename)
        tasks.append((file_path, outputPath, idx))
        idx += 1

    # 使用多进程池
    with Pool(pool_size) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc="Processing files"))

    print(f"Finished {len(results)} files.")

if __name__ == "__main__":
    try:
        process("D:\\TempDataChche\\1st_processed_lower", "D:\\TempDataChche\\jsonData_low", pool_size=7)
    except Exception as e:
        print(e)