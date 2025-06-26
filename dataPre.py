import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
import open3d as o3d
import os
import json
import tqdm

def process_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    lower_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    log_file_path = os.path.join(target_dir, 'log.txt')
    with open(log_file_path, 'w') as log_file:
        for i, lower_dir in tqdm.tqdm(enumerate(lower_dirs), total=len(lower_dirs), desc="Processing directories"):
            obj_file_name = f"{lower_dir}_lower.obj"
            json_file_name = f"{lower_dir}_lower.json"

            obj_file_path = os.path.join(source_dir, lower_dir, obj_file_name)
            json_file_path = os.path.join(source_dir, lower_dir, json_file_name)

            try:
                if not (os.path.exists(obj_file_path) and os.path.exists(json_file_path)):
                    raise FileNotFoundError(f"Missing {obj_file_name} or {json_file_name}")

                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                labels = data['labels']

                new_obj_lines = []
                comment_count = -1
                with open(obj_file_path, 'r') as f:

                    for line_number, line in enumerate(f, start=0):
                        stripped_line = line.strip()
                        if stripped_line.startswith('#'):
                            comment_count += 1
                            continue
                        if stripped_line.startswith('v '):
                            # coordinates = stripped_line.split()[1:]  # Remove the 'v' prefix
                            label = labels[line_number - 1 - comment_count]

                            # Convert label to the required format
                            # new_label = '0.000000' if label == 0 else '1.000000'
                            if label>10 and label<21:
                                label+=10
                            if label>30 and label<41:
                                label+=10

                            new_label = str(label)+".000000"

                            new_line = ' '.join(stripped_line.split() + [new_label])
                            new_obj_lines.append(new_line)
                        elif stripped_line.startswith('f'):
                            line_split =stripped_line.split(" ")
                            for split_id in range(len(line_split)):
                                if line_split[split_id] == "f":
                                    continue
                                else:
                                    line_split[split_id] = line_split[split_id].split("//")[0]
                            new_line = " ".join(line_split)
                            new_obj_lines.append(new_line)



                new_obj_content = '\n'.join(new_obj_lines)
                new_obj_file_path = os.path.join(target_dir, f'teeth_{i + 1}.txt')

                with open(new_obj_file_path, 'w') as f:
                    f.write(new_obj_content)
                    print(f"Processed {lower_dir}, saved as {new_obj_file_path}")

            except Exception as e:
                error_message = f"Error processing {lower_dir}: {str(e)}"
                print(error_message)
                log_file.write(error_message + '\n')

# # Example usage
# try:
#     source_directory = r'D:\TempDataChche\raw_data\lower'
#     target_directory = r'D:\TempDataChche\1st_processed_lower'
#     process_files(source_directory, target_directory)
# except Exception as e:
#     print(f"An error occurred: {str(e)}")


########################################
# 1st process
########################################

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

def process(inputPath,outputPath):
    """
    输入总路径
    循环读出所有txt文件
    返回vertex,normal,label数组的数组
    [
      [[v1,v2,v3], [n1,n2,n3], [l1]],
      [[v1,v2,v3], [n1,n2,n3], [l2]],
      ...
    ]
    每个子项代表一个点，格式是 [vertex, normal, label]
    """
    r=0
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    file_names = os.listdir(inputPath)
    for filename in tqdm.tqdm(file_names, desc="Loading files"):
        if filename == "log.txt":
            continue
        file_path = os.path.join(inputPath, filename)

        vertices = []
        normals = []
        labels = []

        with open(file_path, 'r') as f:

            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertex = parts[1:4]
                    normal = parts[4:7]
                    label = parts[7:8]


                    vertices.append(vertex)
                    normals.append(normal)
                    labels.append(label)

        # 将该文件的所有点组合成 [vertices, normals, labels]
        vertices=np.array(vertices,dtype=np.float32)
        normals=np.array(normals,dtype=np.float32)
        labels=[int(float(x[0])) for x in labels]
        vertices=pc_normalize(np.array(vertices)) #(N,3)
        normals=pc_normalize(np.array(normals)) #(N,3)
        labels=number_covert(labels) #(N)
        labels=np.array(labels).reshape(-1,1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.5, max_nn=30))
        normals= np.asarray(pcd.normals)

        gaussian_curvatures = compute_gaussian_curvature(vertices, normals)
        gaussian_curvatures=gaussian_curvatures.reshape(-1,1)
        curvature_array = compute_point_curvature_new(vertices, normals)
        curvature_array=curvature_array.reshape(-1,1)


        features=np.concatenate((vertices,normals,gaussian_curvatures,curvature_array),axis=1)
        category=np.tile((0,1),(vertices.shape[0],1))

        result = {
            "feature": features.tolist(),
            "label": labels.tolist(),
            "category": category.tolist()
        }

        with open(os.path.join(outputPath, f"{++r}.json"), 'w') as f:
            json.dump(result, f)
        print(f"{r}.json saved")
        r+=1

try:
    process("D:\\TempDataChche\\1st_processed_lower", "D:\\TempDataChche\\jsonData_low")
except Exception as e:
    print(e)

# def data_load(DATA_PATH):
#     """
#     According to the path, load the teeth data from the preprocessed json file.
#     Return: feature (8-d vector), label (int:0-32), category ((1, 0) for mandible / (0, 1) for maxillary)
#     """
#     f = open(DATA_PATH, 'r')
#     teeth_dict = json.load(f)
#     feature = teeth_dict['features']
#     label = teeth_dict['label']
#     category = teeth_dict['category']
#     f.close()
#     feature = np.array(feature).astype(np.float32)
#     label = np.array(label).astype(np.int64)
#     category = np.array(category).astype(np.float32)
#
#     return feature, label, category
#
#
#
# file_path = r"D:\TempDataChche\raw_data\lower\json_data\0.json"
# features, labels, categories = data_load(file_path)
# print("Features shape:", features.shape)
# print("Labels:", labels)
# print("Categories:", categories)
