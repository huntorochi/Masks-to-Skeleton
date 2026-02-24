import open3d as o3d
from sklearn.cluster import KMeans
from scipy.spatial import distance
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import find
from typing import NamedTuple
from tqdm import tqdm
import torch.nn.functional as F

from plyfile import PlyData
import math
import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import collections
import re
import struct
from gsplat import rasterization
import json
import glob
import itertools

C0 = 0.28209479177387814

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
BaseCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


# 删除远离主点云的点云簇
def kmeans_clustering(points, num_clusters):
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=30,
        algorithm="lloyd"
    )
    labels = kmeans.fit_predict(points)

    rough_centroids = kmeans.cluster_centers_

    centroids = []

    for i in range(num_clusters):
        center = rough_centroids[i]

        centroids.append(center)

    centroids = np.array(centroids)

    # 计算每个簇的平均半径
    radii = []
    for i in range(num_clusters):
        cluster_points = points[labels == i]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        radii.append(np.median(distances) / 2)  # 使用中位数更鲁棒
    return labels, centroids, np.array(radii)


def remove_distant_clusters(pcd, labels, centroids):
    clusters = [pcd.select_by_index(np.where(labels == i)[0]) for i in range(len(centroids))]

    main_cluster = max(clusters, key=lambda c: len(c.points))
    main_centroid = np.mean(np.asarray(main_cluster.points), axis=0)
    distances = np.linalg.norm(centroids - main_centroid, axis=1)

    threshold = np.mean(distances) + 2 * np.std(distances)
    filtered_clusters = [clusters[i] for i in range(len(centroids)) if distances[i] < threshold]
    filtered_centroids = centroids[distances < threshold]
    mask = distances < threshold

    if not filtered_clusters:
        return pcd, centroids, mask

    filtered_pcd = filtered_clusters[0]
    for cluster in filtered_clusters[1:]:
        filtered_pcd += cluster

    return filtered_pcd, np.array(filtered_centroids), mask


def build_mst(centroids):
    dist_matrix = distance.cdist(centroids, centroids)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    mst_edges = np.transpose(np.nonzero(mst))
    return mst_edges


def identify_keypoints(mst_edges, centroids):
    degree_count = defaultdict(int)
    for edge in mst_edges:
        degree_count[edge[0]] += 1
        degree_count[edge[1]] += 1

    endpoints = [i for i, degree in degree_count.items() if degree == 1]
    branch_points = [i for i, degree in degree_count.items() if degree > 2]
    pass_through_points = [i for i, degree in degree_count.items() if degree == 2]

    return endpoints, branch_points, pass_through_points


def fetchPly(path, scene_scale=1.0):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T * scene_scale
    colors = np.ones_like(positions)  # Placeholder for colors
    normals = np.ones_like(positions)  # Placeholder for normals
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def get_clusters(pcd, num_clusters=100, nb_points=64, radius=0.05, nb_neighbors=100, std_ratio=0.2):
    '''
    :param pcd:  点云
    :param num_clusters:  聚类数
    :param nb_points:  每个点的邻居数 含义：每个点需要在 radius 范围内至少有 nb_points 个邻居，否则判为离群点。
    :param radius:  检查半径
    :param nb_neighbors:  邻居数 每个点考虑的邻居数，用于计算局部平均距离。
    :param std_ratio:  标准差比率
    :return:
    '''
    points = pcd.points
    # print("原始点云总点数:", len(points))
    # 将 numpy 数组转换为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用 remove_radius_outlier 删除局部稀疏点
    pcd, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

    # 使用统计滤波器去除孤立点
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_filtered = pcd.select_by_index(ind)

    # 打印点的总数
    print("去除孤立点后的总点数:", len(pcd_filtered.points))

    labels, centroids, radii = kmeans_clustering(np.asarray(pcd_filtered.points), num_clusters)
    # 删除远离主点云的点云簇
    pcd_filtered, centroids, mask = remove_distant_clusters(pcd_filtered, labels, centroids)
    # 计算最小生成树
    mst_edges = build_mst(centroids)
    endpoints, branch_points, pass_through_points = identify_keypoints(mst_edges, centroids)

    radii = radii[mask]  # 同步过滤半径

    print("骨架点数:", len(centroids))
    print("骨架边数:", len(mst_edges))
    print("端点数:", len(endpoints))
    print("分叉点数:", len(branch_points))
    return centroids, mst_edges, radii


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class Camera(BaseCamera):
    @property
    def K(self):
        K = np.eye(3)
        if self.model == "SIMPLE_PINHOLE" or self.model == "SIMPLE_RADIAL" or self.model == "RADIAL" or self.model == "SIMPLE_RADIAL_FISHEYE" or self.model == "RADIAL_FISHEYE":
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[0]
            K[0, 2] = self.params[1]
            K[1, 2] = self.params[2]
        elif self.model == "PINHOLE" or self.model == "OPENCV" or self.model == "OPENCV_FISHEYE" or self.model == "FULL_OPENCV" or self.model == "FOV" or self.model == "THIN_PRISM_FISHEYE":
            K[0, 0] = self.params[0]
            K[1, 1] = self.params[1]
            K[0, 2] = self.params[2]
            K[1, 2] = self.params[3]
        else:
            raise NotImplementedError
        return K


def camera_to_intrinsic(camera):
    '''
    camera object to intrinsic matrix
    fx 0  cx
    0  fy cy
    0  0  1
    '''
    return np.array([
        [camera.params[0], 0, camera.params[2]],
        [0, camera.params[1], camera.params[3]],
        [0, 0, 1]
    ])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def get_next_test_name(father_path, prefix):
    """
    自动检测父路径下是否有符合前缀的目录，自动编号生成下一个可用名称。
    :param father_path: 保存测试结果的路径，例如 root_path/save_test/synthetic_tree
    :param prefix: 前缀，例如 "gsplat_adj_20250402"
    :return: (new_test_name, full_path)
    """
    os.makedirs(father_path, exist_ok=True)

    # 匹配形如 gsplat_adj_20250402_00005 的文件夹
    pattern = re.compile(f"^{re.escape(prefix)}_(\\d{{5}})$")
    existing_nums = []

    for name in os.listdir(father_path):
        match = pattern.match(name)
        if match:
            existing_nums.append(int(match.group(1)))

    next_num = 1 if not existing_nums else max(existing_nums) + 1
    new_test_name = f"{prefix}_{next_num:05d}"
    full_path = os.path.join(father_path, new_test_name)

    return new_test_name, full_path


###########################################################################################

def read_blender(root_path, resize_ratio=1.0):
    train_path = os.path.join(root_path, 'train')
    image4_path = os.path.join(train_path, 'image4')
    data_np_path = os.path.join(train_path, 'data_np')

    img_list = os.listdir(image4_path)
    print(f"all data images: {len(img_list)}")

    img_name0 = img_list[0]
    img_path0 = os.path.join(image4_path, img_name0)
    # img size
    img0 = cv2.imread(img_path0)
    cam_height, cam_width = img0.shape[:2]
    # 缩放后的分辨率
    cam_height = int(cam_height * resize_ratio)
    cam_width = int(cam_width * resize_ratio)

    np_path0 = os.path.join(data_np_path, img_name0.split('.')[0] + '.npz')
    np_data0 = np.load(np_path0, allow_pickle=True)
    contents0 = np_data0['Cameras'].item()
    cam_model = 'PINHOLE'
    cam_id = 1
    cam_cx = cam_width / 2
    cam_cy = cam_height / 2

    fovx = contents0["camera_angle_x"]
    fovy = focal2fov(fov2focal(fovx, cam_height), cam_width)
    focal_length_x = fov2focal(fovx, cam_width)
    focal_length_y = fov2focal(fovy, cam_height)

    cam_params = np.array([focal_length_x, focal_length_y, cam_cx, cam_cy])

    colmap_camera = Camera(cam_id, cam_model, cam_width, cam_height, cam_params)

    # self.poses
    poses = []
    for image_name in img_list:
        np_path = os.path.join(data_np_path, image_name.split('.')[0] + '.npz')
        np_data = np.load(np_path, allow_pickle=True)
        Cameras = np_data['Cameras'].item()
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(Cameras['transform_matrix'])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        poses.append(pose)

    return poses, colmap_camera, img_list, image4_path


def read_nerf(root_path, resize_ratio=1.0):
    json_path = os.path.join(root_path, 'transforms_train.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data['frames']

    img_path0 = os.path.join(root_path, frames[0]['file_path'])
    # img size
    img0 = cv2.imread(img_path0)
    cam_height, cam_width = img0.shape[:2]
    # 缩放后的分辨率
    cam_height = int(cam_height * resize_ratio)
    cam_width = int(cam_width * resize_ratio)

    # camera parameters
    fovx = data['camera_angle_x']
    fovy = focal2fov(fov2focal(fovx, cam_height), cam_width)
    focal_length_x = fov2focal(fovx, cam_width)
    focal_length_y = fov2focal(fovy, cam_height)

    cam_model = 'PINHOLE'
    cam_id = 1
    cam_cx = cam_width / 2
    cam_cy = cam_height / 2
    cam_params = np.array([focal_length_x, focal_length_y, cam_cx, cam_cy])
    colmap_camera = Camera(cam_id, cam_model, cam_width, cam_height, cam_params)

    # self.poses and self.image_list
    poses = []
    img_list = []
    for frame in frames:
        file_path = frame['file_path']
        img_name = file_path.split('/')[-1]
        img_list.append(img_name)

        # get the world-to-camera transform and set R, T
        c2w = np.array(frame['transform_matrix'])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        poses.append(pose)
    image4_path = os.path.join(root_path, 'train')

    return poses, colmap_camera, img_list, image4_path


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_metashape(root_path, resize_ratio=1.0):
    try:
        cameras_extrinsic_file = os.path.join(root_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(root_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(root_path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(root_path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 读取相机内参
    camera0 = cam_intrinsics[0]
    original_width = camera0.width
    original_height = camera0.height
    cam_height = int(original_height * resize_ratio)
    cam_width = int(original_width * resize_ratio)

    cam_model = camera0.model
    if cam_model == "PINHOLE":
        orig_flx = camera0.params[0]
        orig_fly = camera0.params[1]
        orig_FovX = focal2fov(orig_flx, original_width)
        orig_FovY = focal2fov(orig_fly, original_height)

        focal_length_x = fov2focal(orig_FovX, cam_width)
        focal_length_y = fov2focal(orig_FovY, cam_height)
        cam_id = 1
        cam_cx = cam_width / 2
        cam_cy = cam_height / 2
    else:
        orig_flx = camera0.params[0]
        orig_FovX = focal2fov(orig_flx, original_width)
        orig_FovY = focal2fov(orig_flx, original_height)

        focal_length_x = fov2focal(orig_FovX, cam_width)
        focal_length_y = fov2focal(orig_FovY, cam_height)
        cam_id = 1
        cam_cx = cam_width / 2
        cam_cy = cam_height / 2

    cam_params = np.array([focal_length_x, focal_length_y, cam_cx, cam_cy])
    colmap_camera = Camera(cam_id, cam_model, cam_width, cam_height, cam_params)

    # 读取相机外参
    poses = []
    img_list = []
    for idx, key in enumerate(cam_extrinsics):
        cam_extrinsic = cam_extrinsics[key]
        img_name = os.path.basename(cam_extrinsic.name)
        img_list.append(img_name)

        R = qvec2rotmat(cam_extrinsic.qvec)
        T = cam_extrinsic.tvec
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        poses.append(pose)
    image4_path = os.path.join(root_path, "images")

    return poses, colmap_camera, img_list, image4_path


def get_normal_img_mask(image_lst, image4_path, img_height, img_width, device):
    """
    获取正常的图像列表和掩码
    """
    # 读取 RGB 图像（BGR → RGB）
    gt_image_lst = []
    gt_tensor_lst = []
    gt_mask_lst = []
    for i, frame_ in enumerate(tqdm(image_lst)):
        gt_image_path = os.path.join(image4_path, frame_)
        gt_image_all = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
        # ========== 提取 RGB 和 mask ==========
        if gt_image_all.shape[2] == 4:
            # 有 alpha 通道
            gt_image = gt_image_all[:, :, :3]
            gt_mask = gt_image_all[:, :, 3]
            gt_mask = (gt_mask / 255.0).astype(np.uint8)
            gt_image = gt_image * gt_mask[:, :, None]
        else:
            gt_image = gt_image_all
            # mask = 非黑即白
            gray_img = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            threshold = 10  # 0~255 之间调节
            gt_mask = np.where(gray_img < threshold, 0, 1).astype(np.uint8)
            # gt_mask = np.ones_like(gt_image[:, :, 0])
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # ========== 缩放 RGB 图像和 mask ==========
        if gt_image.shape[:2] != (img_height, img_width):
            gt_image = cv2.resize(gt_image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        gt_image_lst.append(gt_image)
        # 归一化到 [0, 1] 并转换为 tensor
        gt_tensor = torch.tensor(gt_image, dtype=torch.float32) / 255.0
        gt_tensor = gt_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)
        gt_tensor_lst.append(gt_tensor)
        gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        gt_mask_lst.append(gt_mask_tensor)

    return gt_image_lst, gt_tensor_lst, gt_mask_lst


def get_metashape_img_mask(image_lst, image4_path, img_height, img_width, device):
    """
        获取tree图像列表和掩码
    """
    # 读取 RGB 图像（BGR → RGB）
    gt_image_lst = []
    gt_tensor_lst = []
    gt_mask_lst = []

    mask_path = image4_path.replace('images', 'masks')

    for i, frame_ in enumerate(tqdm(image_lst)):
        gt_image_path = os.path.join(image4_path, frame_)
        gt_image_all = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
        # ========== 提取 RGB 和 mask ==========
        gt_image = gt_image_all
        # mask = 非黑即白
        mask_frame_path = os.path.join(mask_path, frame_.replace(".JPG", ".jpg"))
        gt_mask_all = cv2.imread(mask_frame_path, cv2.IMREAD_UNCHANGED)
        if gt_mask_all.max() > 1:
            gt_mask_all = gt_mask_all / 255.0
        gt_mask = gt_mask_all.astype(np.uint8)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # ========== 缩放 RGB 图像和 mask ==========
        if gt_image.shape[:2] != (img_height, img_width):
            gt_image = cv2.resize(gt_image, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        gt_image_lst.append(gt_image)
        # 归一化到 [0, 1] 并转换为 tensor
        gt_tensor = torch.tensor(gt_image, dtype=torch.float32) / 255.0
        gt_tensor = gt_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)
        gt_tensor_lst.append(gt_tensor)
        gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        gt_mask_lst.append(gt_mask_tensor)

    return gt_image_lst, gt_tensor_lst, gt_mask_lst


def xyz2uv(xyz_world, pose_w2c, intrinsic):
    """
    输入:
        xyz_world: (N, 3) Tensor or ndarray，世界坐标点
        pose_w2c: (4, 4) ndarray，相机世界到相机的变换矩阵（已处理轴变换）
        intrinsic: (3, 3) ndarray，相机内参矩阵
    输出:
        uv: (N, 2) Tensor，像素坐标
    """
    if isinstance(xyz_world, np.ndarray):
        xyz_world = torch.tensor(xyz_world, dtype=torch.float32)
    if isinstance(pose_w2c, np.ndarray):
        pose_w2c = torch.tensor(pose_w2c, dtype=torch.float32)
    if isinstance(intrinsic, np.ndarray):
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)

    N = xyz_world.shape[0]

    # 增加齐次维
    xyz_world_h = torch.cat([xyz_world, torch.ones((N, 1))], dim=1)  # (N, 4)

    # 世界 → 相机坐标
    xyz_cam = (pose_w2c @ xyz_world_h.T).T  # (N, 4) → (N, 3)
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    z = z.clamp(min=1e-6)  # 避免除以 0

    # 相机坐标 → 像素坐标
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    return torch.stack([u, v], dim=1)  # (N, 2)


class BranchModelSigmoidMST_adj:
    def __init__(self, root_path, centroids, mst_edges, radii, sh_degree, device, xyz_lr, r_lr, eig_lr, num_samples_per_edge,
                 resize_ratio=1.0, use_metashape=False, scene_scale=1.0):
        self.branch_num = centroids.shape[0]

        self.device = device
        self.sh_degree = sh_degree
        self.num_samples_per_edge = num_samples_per_edge
        self.root_path = root_path

        self.resize_ratio = resize_ratio
        self.scene_scale = scene_scale

        if use_metashape:  # tree
            poses, colmap_camera, image_list, image4_path = read_metashape(root_path, resize_ratio=resize_ratio)
            # 删除不在mask中的index
            masks_path = os.path.join(image4_path.replace('images', 'masks'))
            masks_name_list = os.listdir(masks_path)
            masks_name_list = [name.replace('.jpg', '.JPG') for name in masks_name_list]
            print('masks_name_list:', masks_name_list)

            # 过滤掉不在mask中的图片
            filtered_image_idx = []
            for idx, image_name in enumerate(image_list):
                if image_name in masks_name_list:
                    filtered_image_idx.append(idx)
                else:
                    print(f"Image {image_name} not found in masks, skipping.")

            # 过滤 poses 和 image_list
            poses = [poses[i] for i in filtered_image_idx]
            image_list = [image_list[i] for i in filtered_image_idx]
        else:
            poses, colmap_camera, image_list, image4_path = read_blender(root_path, resize_ratio=resize_ratio)

        self.poses = poses
        self.colmap_camera = colmap_camera
        self.image_list = image_list

        self.poses = np.array(self.poses)
        # 在所有位姿的第四列缩放平移分量 (每个位姿矩阵的前三行第四列)
        self.poses[:, :3, 3] *= self.scene_scale
        self.img_height, self.img_width = self.colmap_camera.height, self.colmap_camera.width
        self.intrinsic_mat = camera_to_intrinsic(self.colmap_camera)
        print('Intrinsic Matrix:', self.intrinsic_mat)
        print('Poses:', self.poses.shape)

        if use_metashape:
            gt_image_lst, gt_tensor_lst, gt_mask_lst = get_metashape_img_mask(self.image_list, image4_path, self.img_height,
                                                                              self.img_width, self.device)
        else:
            gt_image_lst, gt_tensor_lst, gt_mask_lst = get_normal_img_mask(self.image_list, image4_path, self.img_height,
                                                                           self.img_width, self.device)

        self.gt_image = gt_image_lst
        self.gt_mask = gt_mask_lst

        self.gt_image = np.array(self.gt_image)
        self.gt_mask_tensor = torch.cat(self.gt_mask, dim=0).to(self.device).unsqueeze(-1)

        print('GT Image:', len(self.gt_image), self.gt_image.shape)

        self.num_imgs = len(self.image_list)

        #######################################################################
        # 模型数据
        self._centroids = nn.Parameter(torch.Tensor(centroids).to(self.device).requires_grad_(True))  # 簇的中心点 N, 3
        print('num_centroids:', self._centroids.shape, self._centroids.device)
        print('radii:', radii.shape, type(radii))
        self.radii = torch.log(torch.tensor(radii, device=device, dtype=torch.float32)).squeeze(-1)
        self._r = nn.Parameter(
            torch.ones((self.branch_num,), device=self.device).requires_grad_(True) * self.radii)  # 簇的中心点半径 N, 1

        MIN = 1e-6
        MAX = 1 - MIN

        # 最小生成树的边
        adj = torch.ones((self.branch_num, self.branch_num), dtype=torch.float32) * MIN  # 邻接矩阵
        for i, (a, b) in enumerate(mst_edges):
            adj[a, b] = MAX
            adj[b, a] = MAX

        # 反 Sigmoid 得到可训练参数
        inverse_A = torch.logit(adj, eps=1e-8)

        self._adj_logits = nn.Parameter(inverse_A.to(self.device).requires_grad_(True))  # 邻接矩阵的对数几率

        self.xyz_lr = xyz_lr
        self.r_lr = r_lr
        self.eig_lr = eig_lr

        self.training_step(xyz_lr=xyz_lr, r_lr=r_lr, eig_lr=eig_lr)

    def training_step(self, xyz_lr=0.016, r_lr=0.0025, eig_lr=0.0025):
        l = [
            {'params': [self._centroids], 'lr': xyz_lr, "name": "centroids"},
            {'params': [self._r], 'lr': r_lr, "name": "r"},
            {'params': [self._adj_logits], 'lr': eig_lr, "name": "adj_logits"},
        ]
        # 创建optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0)

    def get_graph(self):
        A_reconstructed = self._adj_logits
        sigmoid_A_reconstructed = torch.sigmoid(A_reconstructed)
        # 获取上三角矩阵（不包括对角线）的索引
        triu_indices = torch.triu_indices(A_reconstructed.shape[0], A_reconstructed.shape[1], offset=1)

        # 生成 edges
        edges = torch.stack((triu_indices[0], triu_indices[1]), dim=1)

        # 计算概率
        prob_xyz = sigmoid_A_reconstructed[triu_indices[0], triu_indices[1]]

        # 提取边的起始和终止顶点的坐标
        xyz1 = self._centroids[edges[:, 0]]
        xyz2 = self._centroids[edges[:, 1]]

        # 计算边起点和终点的半径
        r1 = torch.exp(self._r[edges[:, 0]])
        r2 = torch.exp(self._r[edges[:, 1]])

        return xyz1, xyz2, r1, r2, prob_xyz

    def update_A_reconstructed_with_MST(self, A_reconstructed, big_shift=10.0):
        """
        - A_reconstructed: N x N, 网络预测出来的邻接矩阵(可微)
        - prob_matrix_real: 对 A_reconstructed 做 sigmoid 得到边存在的概率(可微)
        - prob_matrix_fake: = prob_matrix_real.detach() (不带梯度，用于 MST)
        - 利用 MST 结果，对 prob_matrix_real / A_reconstructed 做进一步操作(带梯度)。
        - 这里演示了如何将 MST 的信息反过来影响最终的损失或其他张量 (比如 -epsilon)。
        """
        # -------------------------------------------------
        # 1) 构建 fake branch, 用于 MST
        # -------------------------------------------------
        N = A_reconstructed.shape[0]

        with torch.no_grad():
            # prob_matrix_fake 不带梯度
            prob_matrix_fake = torch.sigmoid(A_reconstructed).detach()

        # 转 numpy (不带梯度), 用于 MST
        prob_np = prob_matrix_fake.cpu().numpy()

        # 构造代价矩阵(1 - p)，只保留上三角(不含对角线)
        cost_mat = 1.0 - prob_np
        np.fill_diagonal(cost_mat, 0.0)
        cost_mat = np.triu(cost_mat, k=1)

        # 做 MST
        mst_csr = minimum_spanning_tree(cost_mat)

        # 取出 MST 的边
        mst_rows, mst_cols, _ = find(mst_csr)

        # 把 (row, col) 放进一个 set，用于快速判断
        mst_edges_set = set(zip(mst_rows, mst_cols))
        # 如果需要无向对称，可加这行
        mst_edges_set.update(zip(mst_cols, mst_rows))

        # -------------------------------------------------
        # 2) 根据 MST 和预测概率，判断 E+, E-, 一致
        # -------------------------------------------------
        #   - E+ : MST要的边，但 A<0 => prob<0.5
        #   - E- : MST不要的边，但 A>0 => prob>0.5
        #   - 其余不动
        #
        #   这里仍只遍历上三角(不含对角线)，对称部分一起更新
        # -------------------------------------------------

        # 拿到上三角的所有 (i, j)
        triu_indices = torch.triu_indices(N, N, offset=1)  # [2, E], E = N*(N-1)/2
        edges = triu_indices.t()  # [E, 2], each row = (i, j)

        # 预测概率 > 0.5 ?
        # 注意这里也可以直接看 A_reconstructed>0
        # 只拿上三角
        pred_prob_vals = prob_matrix_fake[edges[:, 0], edges[:, 1]].cpu()

        # 是否在 MST 中
        edges_np = edges.cpu().numpy()
        in_mst = np.array([tuple(e) in mst_edges_set for e in edges_np], dtype=bool)

        # 转回 torch
        in_mst_torch = torch.from_numpy(in_mst)

        # E+ : MST = 1, prob<0.5 => A<0
        # E- : MST = 0, prob>0.5 => A>0
        # (prob>0.5) 用 pred_prob_vals>0.5
        # (prob<0.5) 用 pred_prob_vals<=0.5
        # 当然也可以直接看 A_reconstructed>0 / <0, 这里用 prob 做演示
        mask_prob_small = (pred_prob_vals <= 0.5)  # A<0
        mask_prob_large = (pred_prob_vals > 0.5)  # A>0

        # E+ mask
        mask_Ep = in_mst_torch & mask_prob_small  # E+ ,mst yes prob no
        # E- mask
        mask_Em = (~in_mst_torch) & mask_prob_large  # E- ,mst no prob yes

        # 取出满足 E+ 的 (i, j)
        Ep_idx = edges[mask_Ep]  # shape [k, 2]
        # 取出满足 E- 的 (i, j)
        Em_idx = edges[mask_Em]  # shape [k, 2]

        # -------------------------------------------------
        # 3) 对 A_reconstructed 做加/减 big_shift
        # -------------------------------------------------

        A_reconstructed_update = A_reconstructed.clone()
        # 向量化操作（不需要 for）
        if Ep_idx.shape[0] > 0:  # E+ edges
            iE, jE = Ep_idx[:, 0], Ep_idx[:, 1]
            A_reconstructed_update[iE, jE] += big_shift
            A_reconstructed_update[jE, iE] += big_shift

        if Em_idx.shape[0] > 0:  # E- edges
            iE, jE = Em_idx[:, 0], Em_idx[:, 1]
            A_reconstructed_update[iE, jE] -= big_shift
            A_reconstructed_update[jE, iE] -= big_shift

        return A_reconstructed_update

    def get_graph_mst(self, big_shift=10.0):
        A_reconstructed = self._adj_logits
        A_reconstructed = self.update_A_reconstructed_with_MST(A_reconstructed, big_shift=big_shift)

        sigmoid_A_reconstructed = torch.sigmoid(A_reconstructed)
        # 获取上三角矩阵（不包括对角线）的索引
        triu_indices = torch.triu_indices(A_reconstructed.shape[0], A_reconstructed.shape[1], offset=1)

        # 生成 edges
        edges = torch.stack((triu_indices[0], triu_indices[1]), dim=1)

        # 计算概率
        prob_xyz = sigmoid_A_reconstructed[triu_indices[0], triu_indices[1]]

        # 提取边的起始和终止顶点的坐标
        xyz1 = self._centroids[edges[:, 0]]
        xyz2 = self._centroids[edges[:, 1]]

        # 计算边起点和终点的半径
        r1 = torch.exp(self._r[edges[:, 0]])
        r2 = torch.exp(self._r[edges[:, 1]])

        return xyz1, xyz2, r1, r2, prob_xyz

    def render_GS_MST(self, indices):
        # xyz1, xyz2, r1, r2, prob_xyz = self.get_graph()
        xyz1, xyz2, r1, r2, prob_xyz = self.get_graph_mst(big_shift=big_shift)

        t = torch.linspace(0, 1, steps=self.num_samples_per_edge, device=xyz1.device).unsqueeze(-1)  # (n_samples, 1)
        # 插值生成采样点
        sampled_positions = xyz1.unsqueeze(1) + t * (xyz2.unsqueeze(1) - xyz1.unsqueeze(1))  # (num_edges, n_samples, 3)
        sampled_radii = r1.unsqueeze(1) + t.squeeze(-1) * (r2.unsqueeze(1) - r1.unsqueeze(1))  # (num_edges, n_samples)
        xyz = sampled_positions.reshape(-1, 3)
        radii = sampled_radii.reshape(-1)
        num_xyz = xyz.shape[0]
        scales = radii[..., None].repeat(1, 3)

        # 定义 opacities
        sampled_opacities = prob_xyz.unsqueeze(-1).repeat(1, self.num_samples_per_edge)  # (num_edges, n_samples)
        opacities = sampled_opacities.reshape(-1, 1)  # (num_edges * n_samples, 1)

        # 旋转四元数初始化
        rots = torch.zeros((num_xyz, 4), device=xyz.device)
        rots[:, 0] = 1  # 四元数单位旋转

        # 定义颜色特征
        RGB_color = torch.ones((num_xyz, 3), device=xyz.device).float()
        fused_color = RGB2SH(RGB_color)
        features = torch.zeros((fused_color.shape[0], 3, (self.sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        _features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        _features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        features = torch.cat([_features_dc, _features_rest], dim=1)

        # 应用筛选到关键参数
        selected_poses = self.poses[indices.cpu().numpy()]  # 形状变为 (selected_num, 4, 4)

        # 进行光栅化处理
        colors, alphas, meta = rasterization(
            means=xyz, quats=rots, scales=scales, opacities=opacities.squeeze(-1),
            colors=features,
            viewmats=torch.tensor(selected_poses, dtype=torch.float32).unsqueeze(0).to(self.device),
            Ks=torch.tensor(self.intrinsic_mat, dtype=torch.float32).unsqueeze(0).to(self.device),
            width=self.img_width, height=self.img_height,
            render_mode='RGB', sh_degree=self.sh_degree,
            packed=True, sparse_grad=False, channel_chunk=32, tile_size=32,
            near_plane=0.01, far_plane=1e10)

        rgb = colors[..., :3]  # 提取前 3 个通道 torch.Size([1, 1440, 1920, 3]) hw
        # alpha torch.Size([1, 1440, 1920, 1])

        return rgb, alphas, meta

    def render_GS_batch(self, indices):
        xyz1, xyz2, r1, r2, prob_xyz = self.get_graph()

        t = torch.linspace(0, 1, steps=self.num_samples_per_edge, device=xyz1.device).unsqueeze(-1)  # (n_samples, 1)
        # 插值生成采样点
        sampled_positions = xyz1.unsqueeze(1) + t * (xyz2.unsqueeze(1) - xyz1.unsqueeze(1))  # (num_edges, n_samples, 3)
        sampled_radii = r1.unsqueeze(1) + t.squeeze(-1) * (r2.unsqueeze(1) - r1.unsqueeze(1))  # (num_edges, n_samples)
        xyz = sampled_positions.reshape(-1, 3)
        radii = sampled_radii.reshape(-1)
        num_xyz = xyz.shape[0]
        scales = radii[..., None].repeat(1, 3)

        # 定义 opacities
        sampled_opacities = prob_xyz.unsqueeze(-1).repeat(1, self.num_samples_per_edge)  # (num_edges, n_samples)
        opacities = sampled_opacities.reshape(-1, 1)  # (num_edges * n_samples, 1)

        # 旋转四元数初始化
        rots = torch.zeros((num_xyz, 4), device=xyz.device)
        rots[:, 0] = 1  # 四元数单位旋转

        # 定义颜色特征
        RGB_color = torch.ones((num_xyz, 3), device=xyz.device).float()
        fused_color = RGB2SH(RGB_color)
        features = torch.zeros((fused_color.shape[0], 3, (self.sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        _features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        _features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        features = torch.cat([_features_dc, _features_rest], dim=1)

        # 应用筛选到关键参数
        selected_poses = self.poses[indices.cpu().numpy()]  # 形状变为 (selected_num, 4, 4)
        selected_num_imgs = len(indices)

        # 进行光栅化处理
        colors, alphas, meta = rasterization(
            means=xyz, quats=rots, scales=scales, opacities=opacities.squeeze(-1),
            colors=features,
            viewmats=torch.tensor(selected_poses, dtype=torch.float32).to(self.device),
            Ks=torch.tensor(self.intrinsic_mat, dtype=torch.float32).repeat(selected_num_imgs, 1, 1).to(self.device),
            width=self.img_width, height=self.img_height,
            render_mode='RGB', sh_degree=self.sh_degree,
            packed=True, sparse_grad=False, channel_chunk=32, tile_size=32,
            near_plane=0.01, far_plane=1e10)

        rgb = colors[..., :3]  # 提取前 3 个通道 torch.Size([1, 1440, 1920, 3]) hw
        # alpha torch.Size([1, 1440, 1920, 1])

        return rgb, alphas, meta

    def render_GS_MST_batch(self, indices):
        # xyz1, xyz2, r1, r2, prob_xyz = self.get_graph()
        xyz1, xyz2, r1, r2, prob_xyz = self.get_graph_mst(big_shift=big_shift)

        t = torch.linspace(0, 1, steps=self.num_samples_per_edge, device=xyz1.device).unsqueeze(-1)  # (n_samples, 1)
        # 插值生成采样点
        sampled_positions = xyz1.unsqueeze(1) + t * (xyz2.unsqueeze(1) - xyz1.unsqueeze(1))  # (num_edges, n_samples, 3)
        sampled_radii = r1.unsqueeze(1) + t.squeeze(-1) * (r2.unsqueeze(1) - r1.unsqueeze(1))  # (num_edges, n_samples)
        xyz = sampled_positions.reshape(-1, 3)
        radii = sampled_radii.reshape(-1)
        num_xyz = xyz.shape[0]
        scales = radii[..., None].repeat(1, 3)

        # 定义 opacities
        sampled_opacities = prob_xyz.unsqueeze(-1).repeat(1, self.num_samples_per_edge)  # (num_edges, n_samples)
        opacities = sampled_opacities.reshape(-1, 1)  # (num_edges * n_samples, 1)

        # 旋转四元数初始化
        rots = torch.zeros((num_xyz, 4), device=xyz.device)
        rots[:, 0] = 1  # 四元数单位旋转

        # 定义颜色特征
        RGB_color = torch.ones((num_xyz, 3), device=xyz.device).float()
        fused_color = RGB2SH(RGB_color)
        features = torch.zeros((fused_color.shape[0], 3, (self.sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        _features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        _features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        features = torch.cat([_features_dc, _features_rest], dim=1)

        # 应用筛选到关键参数
        selected_poses = self.poses[indices.cpu().numpy()]  # 形状变为 (selected_num, 4, 4)
        selected_num_imgs = len(indices)

        # 进行光栅化处理
        colors, alphas, meta = rasterization(
            means=xyz, quats=rots, scales=scales, opacities=opacities.squeeze(-1),
            colors=features,
            viewmats=torch.tensor(selected_poses, dtype=torch.float32).to(self.device),
            Ks=torch.tensor(self.intrinsic_mat, dtype=torch.float32).repeat(selected_num_imgs, 1, 1).to(self.device),
            width=self.img_width, height=self.img_height,
            render_mode='RGB', sh_degree=self.sh_degree,
            packed=True, sparse_grad=False, channel_chunk=32, tile_size=32,
            near_plane=0.01, far_plane=1e10)

        rgb = colors[..., :3]  # 提取前 3 个通道 torch.Size([1, 1440, 1920, 3]) hw
        # alpha torch.Size([1, 1440, 1920, 1])

        return rgb, alphas, meta

    def save_data(self, path):
        save_dict = {
            # 模型可学习参数
            'centroids': self._centroids.detach().cpu().numpy(),
            'r': self._r.detach().cpu().numpy(),
            'adj_logits': self._adj_logits.detach().cpu().numpy(),

            # 超参数/训练配置
            'sh_degree': self.sh_degree,
            'num_samples_per_edge': self.num_samples_per_edge,
            'branch_num': self.branch_num,
            'xyz_lr': self.xyz_lr,
            'r_lr': self.r_lr,
            'eig_lr': self.eig_lr,

            # optimizer 状态
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        torch.save(save_dict, path)
        print(f"[INFO] Model data and config saved to {path}")

    def load_data(self, path):
        load_dict = torch.load(path, map_location=self.device)

        # 加载可学习参数
        self._centroids.data = torch.tensor(load_dict['centroids'], dtype=torch.float32, device=self.device)
        self._r.data = torch.tensor(load_dict['r'], dtype=torch.float32, device=self.device)
        self._adj_logits.data = torch.tensor(load_dict['adj_logits'], dtype=torch.float32, device=self.device)

        # 恢复训练配置
        self.sh_degree = load_dict['sh_degree']
        self.num_samples_per_edge = load_dict['num_samples_per_edge']
        self.branch_num = load_dict['branch_num']
        self.xyz_lr = load_dict['xyz_lr']
        self.r_lr = load_dict['r_lr']
        self.eig_lr = load_dict['eig_lr']

        # 重新设置 optimizer 的学习率（如果想继续训练）
        self.training_step(xyz_lr=self.xyz_lr, r_lr=self.r_lr, eig_lr=self.eig_lr)
        self.optimizer.load_state_dict(load_dict['optimizer_state_dict'])

        print(f"[INFO] Model data and config loaded from {path}")


def find_ply_path(base_dir):
    """自动查找包含随机数字的dense.ply文件"""
    pattern = os.path.join(base_dir, "pointcloud_*_dense.ply")
    matches = glob.glob(pattern)

    # 使用正则表达式过滤有效文件
    valid_matches = []
    for path in matches:
        # 匹配 pointcloud_<NUMBER>_dense.ply 格式
        if re.match(r".*pointcloud_\d+_dense\.ply$", path):
            valid_matches.append(path)

    if not valid_matches:
        raise FileNotFoundError(f"No valid PLY files in {base_dir}")

    # 提取数字并排序
    def extract_number(path):
        return int(re.search(r"pointcloud_(\d+)_dense\.ply$", path).group(1))

    valid_matches.sort(key=extract_number)
    return valid_matches[-1]


#######################################
if __name__ == "__main__":
    # real world
    tree_name = 'tree200'
    root_path = rf".\github_mdpi\dataset"
    nerf_tree_path = os.path.join(root_path, tree_name)
    base_dir = rf".\github_mdpi\dataset\{tree_name}"

    dense_name = 'pointcloud_41_dense'
    ply_path = os.path.join(base_dir, f"{dense_name}.ply")
    save_path = rf".\github_mdpi\ours\real\{tree_name}\{dense_name}_github"
    device = torch.device("cuda:0")
    resize_ratio = 0.1
    use_metashape = True
    num_clusters = 150
    nb_points = 5
    radius = 0.1
    nb_neighbors = 5
    std_ratio = 0.1
    scene_scale = 1.1
    size_rate = 1.0

    more_training = 1

    epochs = int(1000 * more_training)
    sub_batch_num = 4

    xyz_lr = 0.0001 * size_rate
    r_lr = 0.1 * size_rate
    eig_lr = 0.001

    silhouette_weight = 2.0
    graph_weight = 0.05

    sh_degree = 3
    num_samples_per_edge = 64

    repruning_graph_radius = 0.04
    big_shift = 1000.0

    alpha_GS = 0.1
    alpha_MST = 0.5

    # ──────────────────────────────────────────────────────────────────────────────
    # Hyper‑parameters  (can be exposed as args / yaml later)
    # ──────────────────────────────────────────────────────────────────────────────
    SIGMA_REP = 0.1  # global centroid repulsion σ 节点之间最小距离
    SIGMA_EDGE_SHORT = 0.02 * size_rate  # edge‑length repulsion σ (acts on edges only) 最小边长
    THETA_MIN_DEG = 20.0  # minimum angle (degrees) for edge pruning  最小角度
    DIR_SIM_THRESH = 0.90  # cosine similarity threshold for “similar direction”  判断两条边的方向是否相似的阈值
    MID_DIST_MAX = 2 * SIGMA_EDGE_SHORT  # desired max distance between mid points  两条方向相似的边的中点之间的最大距离
    R_MIN = 0.01 * size_rate  # radius lower bound (pixel units) 最小半径

    # Loss weights (Phase‑1 will重设) ------------------------------------------------
    W_REP = 1.0
    W_EDGE_SHORT = 1.0
    W_ANGLE = 1.0
    W_MID = 1.0
    W_R_MIN = 1.0


    # ──────────────────────────────────────────────────────────────────────────────
    # Geometry losses
    # ──────────────────────────────────────────────────────────────────────────────

    def gaussian_repulsion(dist: torch.Tensor, sigma: float):
        """Element‑wise :math:`\exp(-d^2/\sigma^2)` used by both repulsion terms."""
        # return torch.exp(-dist_squared / (sigma ** 2))
        return torch.exp(-(dist / sigma) ** 2)


    # 抑制节点之间的重叠
    def loss_centroid_repulsion(centroids: torch.Tensor) -> torch.Tensor:
        r"""Gaussian repulsion between *all* centroid pairs.

        .. math::
            \mathcal{L}_{\text{rep}} = \sum_{i<j} \exp\!\left( -\frac{\lVert c_i-c_j \rVert_2^{2}}{\sigma_{\text{rep}}^{2}} \right)

        where :math:`c_i` is the *i*-th centroid and
        :pydata:`SIGMA_REP` provides :math:`\sigma_{\text{rep}}`.

        This smooth penalty prevents node collapse while keeping gradients
        non‑zero even when centroids are far apart (vanishing exponentially).
        """
        # Compute upper‑triangular pairwise squared distances (no self pairs)
        with torch.no_grad():
            pair_idx = torch.triu_indices(centroids.size(0), centroids.size(0), offset=1)
        diffs = centroids[pair_idx[0]] - centroids[pair_idx[1]]
        dist = torch.norm(diffs, dim=-1)
        return gaussian_repulsion(dist, SIGMA_REP).mean()


    # 抑制边的最短长度
    def loss_edge_short(centroids: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        v = centroids[edges[:, 0]] - centroids[edges[:, 1]]
        dist = torch.norm(v, dim=-1)
        return gaussian_repulsion(dist, SIGMA_EDGE_SHORT).mean()


    # 抑制边之间的夹角 防止z fold的形成
    def loss_angle_fold(centroids: torch.Tensor, edges: torch.Tensor, theta_min_deg: float = THETA_MIN_DEG) -> torch.Tensor:
        """Suppress angles tighter than *theta_min_deg*.

        For each triple *(i,j,k)* with common pivot *j*:
        :math:`\theta_{ijk}` is the angle between the two incident edges.
        We apply

        .. math::
            \operatorname{softplus}\bigl(\cos\theta_{ijk}-\cos\theta_{\min}\bigr)
        """
        cos_thr = math.cos(math.radians(theta_min_deg))
        n = centroids.size(0)
        # --- CPU adjacency list --------------------------------------------------
        adj_list = [[] for _ in range(n)]
        for i, j in edges.cpu().tolist():
            adj_list[i].append(j)
            adj_list[j].append(i)

        triplets = []
        for j in range(n):
            nbrs = adj_list[j]
            if len(nbrs) < 2:
                continue
            triplets.extend((j, i, k) for i, k in itertools.combinations(nbrs, 2))

        if not triplets:
            return centroids.new_tensor(0.0)

        j_idx, i_idx, k_idx = torch.tensor(triplets, device=centroids.device).unbind(1)
        cj = centroids[j_idx]
        vi = F.normalize(centroids[i_idx] - cj, dim=1)
        vk = F.normalize(centroids[k_idx] - cj, dim=1)
        cos_theta = (vi * vk).sum(dim=1)
        return F.softplus(cos_theta - cos_thr).mean()


    # 抑制方向相近的边的中心点距离过近
    def loss_midpoint_direction(
            centroids: torch.Tensor,
            edges: torch.Tensor,
            dist_max: float = MID_DIST_MAX,
            dir_sim_thresh: float = DIR_SIM_THRESH,
    ) -> torch.Tensor:
        """Fully vectorised midpoint‑direction loss.

        Complexity :math:`\mathcal O(E^2)` with E≤5k (≤100 nodes) – still negligible on GPU.
        Step list:
          1. direction & mid‑point for every edge.
          2. broadcast equality to get shared‑node mask.
          3. cosine similarity matrix via matmul.
          4. select valid edge pairs and apply softplus(dist-d_max)^2.
        """
        m = edges.size(0)
        if m < 2:
            return centroids.new_tensor(0.0)

        ci, cj = centroids[edges[:, 0]], centroids[edges[:, 1]]
        vec = F.normalize(cj - ci, dim=1)
        mid = 0.5 * (ci + cj)

        # shared node mask via broadcasting
        n0 = edges[:, 0:1]
        n1 = edges[:, 1:2]
        shared = (n0 == n0.T) | (n0 == n1.T) | (n1 == n0.T) | (n1 == n1.T)

        iu, ju = torch.triu_indices(m, m, 1, device=edges.device)
        shared_ut = shared[iu, ju]
        if shared_ut.sum() == 0:
            return centroids.new_tensor(0.0)

        # direction similarity upper‑triangular
        cos_ut = (vec @ vec.T)[iu, ju]
        dir_mask = cos_ut > dir_sim_thresh
        valid = shared_ut & dir_mask
        if valid.sum() == 0:
            return centroids.new_tensor(0.0)

        iu = iu[valid]
        ju = ju[valid]
        dist = (mid[iu] - mid[ju]).norm(dim=1)
        return F.softplus(dist - dist_max).mean()


    # 抑制边的直径过细
    def loss_radius_lower(r_log: torch.Tensor) -> torch.Tensor:
        r_lin = torch.exp(r_log)
        return F.softplus(R_MIN - r_lin).mean()


    pcd = fetchPly(ply_path, scene_scale=scene_scale)
    centroids, mst_edges, radii = get_clusters(pcd, num_clusters=num_clusters, nb_points=nb_points, radius=radius,
                                               nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    branch_model_sigmoid_mst = BranchModelSigmoidMST_adj(nerf_tree_path, centroids, mst_edges, radii,
                                                         sh_degree, device, xyz_lr, r_lr, eig_lr,
                                                         num_samples_per_edge, resize_ratio=resize_ratio,
                                                         use_metashape=use_metashape, scene_scale=scene_scale)

    num_all_data = branch_model_sigmoid_mst.num_imgs
    rand_indices = torch.randperm(num_all_data, device=device)
    rand_batch_num = num_all_data // sub_batch_num
    print(f"rand_batch_num: {rand_batch_num}")
    #######################################################################
    # save_log
    father_path = os.path.join(save_path)
    prefix = "gsplat_adj_sigmoid_mst_new_pruning_step_20260224_github"

    new_test_name, save_img_path = get_next_test_name(father_path, prefix)
    print(f"save_img_path: {save_img_path}")
    os.makedirs(save_img_path, exist_ok=True)

    # 记录要写的话
    txt_path = os.path.join(save_img_path, "log.txt")
    with open(txt_path, 'w') as f:
        f.write(f"num_clusters: {num_clusters}\n")
        f.write(f"nb_points: {nb_points}\n")
        f.write(f"radius: {radius}\n")
        f.write(f"nb_neighbors: {nb_neighbors}\n")
        f.write(f"std_ratio: {std_ratio}\n")
        f.write(f"more_training: {more_training}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"xyz_lr: {xyz_lr}\n")
        f.write(f"r_lr: {r_lr}\n")
        f.write(f"eig_lr: {eig_lr}\n")
        f.write(f"silhouette_weight: {silhouette_weight}\n")
        f.write(f"graph_weight: {graph_weight}\n")
        f.write(f"sh_degree: {sh_degree}\n")
        f.write(f"num_samples_per_edge: {num_samples_per_edge}\n")
        f.write(f"ply_path: {ply_path}\n")
        f.write(f"num_all_data: {num_all_data}\n")
        f.write(f"repruning_graph_radius : {repruning_graph_radius}\n")
        f.write(f"big_shift: {big_shift}\n")
        f.write(f"alpha_GS: {alpha_GS}\n")
        f.write(f"alpha_MST: {alpha_MST}\n")
        f.write(f"resize_ratio: {resize_ratio}\n")
        f.write(f"use_metashape: {use_metashape}\n")
        f.write(f"sub_batch_num: {sub_batch_num}\n")

        f.write(f"SIGMA_REP: {SIGMA_REP}\n")
        f.write(f"SIGMA_EDGE_SHORT: {SIGMA_EDGE_SHORT}\n")
        f.write(f"THETA_MIN_DEG: {THETA_MIN_DEG}\n")
        f.write(f"R_MIN: {R_MIN}\n")
        f.write(f"DIR_SIM_THRESH: {DIR_SIM_THRESH}\n")
        f.write(f"MID_DIST_MAX: {MID_DIST_MAX}\n")
        f.write(f"W_REP: {W_REP}\n")
        f.write(f"W_EDGE_SHORT: {W_EDGE_SHORT}\n")
        f.write(f"W_ANGLE: {W_ANGLE}\n")
        f.write(f"W_MID: {W_MID}\n")
        f.write(f"W_R_MIN: {W_R_MIN}\n")
        f.write(f"Scene scale: {scene_scale}\n")
        f.write(f'size_rate: {size_rate}\n')

    # tqdm 进度条
    progress_bar = tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True)
    total_loss_check = 1000000
    losses = []
    silhouette_losses = []
    graph_losses = []

    final_print_step_npz_path = ''
    final_print_best_npz_path = ""

    for epoch in progress_bar:
        total_silhouette_loss = 0
        total_loss = 0
        total_graph_loss = 0

        all_imgs_num = num_all_data

        global_indices = torch.randperm(num_all_data, device=device)
        for sub_idx in range(sub_batch_num):
            # 计算当前子批次的索引范围
            start = sub_idx * rand_batch_num
            end = min((sub_idx + 1) * rand_batch_num, num_all_data)
            rand_idx = global_indices[start:end]

            branch_model_sigmoid_mst.optimizer.zero_grad()
            # 计算MST
            rendered_images_MST, rendered_alpha_MST, _ = branch_model_sigmoid_mst.render_GS_MST_batch(indices=rand_idx)

            gt_mask = branch_model_sigmoid_mst.gt_mask_tensor[rand_idx].unsqueeze(0)
            gt_mask = gt_mask.squeeze(0)
            gt_rgb = gt_mask.repeat(1, 1, 1, 3)

            loss_silhouette_MST = F.mse_loss(rendered_alpha_MST, gt_mask)

            # 计算Original Loss
            rendered_images_Original, rendered_alpha_Original, _ = branch_model_sigmoid_mst.render_GS_batch(indices=rand_idx)
            loss_silhouette_Original = F.mse_loss(rendered_alpha_Original, gt_mask)
            loss_silhouette = alpha_GS * loss_silhouette_Original + alpha_MST * loss_silhouette_MST

            # 计算几何 Loss
            loss_A = branch_model_sigmoid_mst._adj_logits
            loss_A_mst = branch_model_sigmoid_mst.update_A_reconstructed_with_MST(loss_A, big_shift=big_shift)

            sigmoid_A_mst = torch.sigmoid(loss_A_mst)
            tau = 0.5  # threshold
            edges_idx = (sigmoid_A_mst > tau).nonzero(as_tuple=False)
            edges_idx = edges_idx[edges_idx[:, 0] < edges_idx[:, 1]].to(device)
            L_rep = loss_centroid_repulsion(branch_model_sigmoid_mst._centroids)
            L_edge_short = loss_edge_short(branch_model_sigmoid_mst._centroids, edges_idx)
            L_angle_fold = loss_angle_fold(branch_model_sigmoid_mst._centroids, edges_idx)
            L_mid = loss_midpoint_direction(branch_model_sigmoid_mst._centroids, edges_idx)
            L_r_min = loss_radius_lower(branch_model_sigmoid_mst._r)

            loss_graph = W_REP * L_rep + W_EDGE_SHORT * L_edge_short + W_ANGLE * L_angle_fold + W_MID * L_mid + W_R_MIN * L_r_min

            # 计算最终 Loss
            loss = silhouette_weight * loss_silhouette + graph_weight * loss_graph

            log_dict = {
                "Epoch": epoch,
                "Loss": loss.item(),
                "loss_silhouette": loss_silhouette.item(),
                "loss_graph": loss_graph.item(),
            }
            # 累加 Loss（用于统计和打印）
            total_loss += loss.item()
            total_silhouette_loss += loss_silhouette.item()
            total_graph_loss += loss_graph.item()

            loss.backward()
            branch_model_sigmoid_mst.optimizer.step()

            progress_bar.set_postfix(log_dict)

        # 计算平均 Loss
        total_loss = total_loss / all_imgs_num
        total_silhouette_loss = total_silhouette_loss / all_imgs_num
        total_graph_loss = total_graph_loss / all_imgs_num

        losses.append(total_loss)
        silhouette_losses.append(total_silhouette_loss)
        graph_losses.append(total_graph_loss)

        if total_loss < total_loss_check and epoch > int(epochs * 0.8):
            total_loss_check = total_loss
            best_txt_path = os.path.join(save_img_path, "best.txt")
            with open(best_txt_path, "a") as f:
                f.write(
                    f"epoch: {epoch}, total_loss: {total_loss}, total_silhouette_loss: {total_silhouette_loss}, total_graph_loss: {total_graph_loss}\n")

            best_chk_path = os.path.join(save_img_path, "best.pth")
            branch_model_sigmoid_mst.save_data(best_chk_path)

            with torch.no_grad():
                A_reconstructed = branch_model_sigmoid_mst._adj_logits
                prob_matrix_fake = torch.sigmoid(A_reconstructed).detach()

                # 转 numpy (不带梯度), 用于 MST
                prob_np = prob_matrix_fake.cpu().numpy()

                # 构造代价矩阵(1 - p)，只保留上三角(不含对角线)
                cost_mat = 1.0 - prob_np
                np.fill_diagonal(cost_mat, 0.0)
                cost_mat = np.triu(cost_mat, k=1)

                # 做 MST
                mst_csr = minimum_spanning_tree(cost_mat)
                mst_rows, mst_cols, _ = find(mst_csr)
                orig_edges = np.stack([mst_rows, mst_cols], axis=1).astype(np.int32)
                xyz = branch_model_sigmoid_mst._centroids
                num_points = xyz.shape[0]
                adj = np.zeros((num_points, num_points), dtype=np.uint8)
                for edge in orig_edges:
                    src, dst = edge
                    adj[src, dst] = 1
                    adj[dst, src] = 1

                # 取上半邻接矩阵
                upper_triangular = np.triu(adj, k=1)
                edges = np.argwhere(upper_triangular)
                vertices = xyz.cpu().detach().numpy()
                best_npz_path = os.path.join(save_img_path, "best.npz")
                final_print_best_npz_path = best_npz_path
                np.savez(best_npz_path, nodes=vertices, edges=edges, scene_scale=scene_scale)

        if (epoch + 1) % int(epochs // 10) == 0 or epoch == 0:
            model_chk_path = os.path.join(save_img_path, f"model_{epoch}.pth")
            branch_model_sigmoid_mst.save_data(model_chk_path)

            losses_chk_path = os.path.join(save_img_path, f"losses_{epoch}.npz")
            np.savez(losses_chk_path,
                     total_loss=np.array(losses),
                     silhouette_loss=np.array(silhouette_losses),
                     graph_loss=np.array(graph_losses),
                     )

            with torch.no_grad():
                check_num = all_imgs_num // 2
                indices50 = torch.tensor(check_num, device=device)
                _, rendered_alpha, _ = branch_model_sigmoid_mst.render_GS_MST(indices=indices50)

                rendered_alpha_np = rendered_alpha.squeeze().cpu().numpy()
                rendered_alpha_np = rendered_alpha_np * 255
                rendered_alpha_np = rendered_alpha_np.astype(np.uint8)

                GT_alpha_np = branch_model_sigmoid_mst.gt_mask_tensor[check_num].squeeze().cpu().numpy()
                GT_alpha_np = GT_alpha_np * 255
                GT_alpha_np = GT_alpha_np.astype(np.uint8)

                GT_img_np = branch_model_sigmoid_mst.gt_image[check_num]
                GT_img_np = cv2.cvtColor(GT_img_np, cv2.COLOR_RGB2BGR)

                # render edges
                A_reconstructed = branch_model_sigmoid_mst._adj_logits
                prob_matrix_fake = torch.sigmoid(A_reconstructed).detach()
                prob_np = prob_matrix_fake.cpu().numpy()

                cost_mat = 1.0 - prob_np
                np.fill_diagonal(cost_mat, 0.0)
                cost_mat = np.triu(cost_mat, k=1)

                # 做 MST
                mst_csr = minimum_spanning_tree(cost_mat)
                mst_rows, mst_cols, _ = find(mst_csr)
                mst_edges = torch.from_numpy(np.stack([mst_rows, mst_cols], axis=1)).long().to(
                    branch_model_sigmoid_mst._centroids.device)

                xyz1 = branch_model_sigmoid_mst._centroids[mst_edges[:, 0]]  # shape: [k, 3]
                xyz2 = branch_model_sigmoid_mst._centroids[mst_edges[:, 1]]  # shape: [k, 3]

                r1 = torch.exp(branch_model_sigmoid_mst._r[mst_edges[:, 0]])  # shape: [k]
                r2 = torch.exp(branch_model_sigmoid_mst._r[mst_edges[:, 1]])  # shape: [k]

                edge_xyz1_xyz2 = torch.cat([xyz1, xyz2], dim=1)  # shape: [k, 6]
                intrinsic = branch_model_sigmoid_mst.intrinsic_mat
                poses_w2c = branch_model_sigmoid_mst.poses

                pose_w2c_50 = poses_w2c[check_num]
                start_xyz = edge_xyz1_xyz2[:, :3].cpu()
                end_xyz = edge_xyz1_xyz2[:, 3:].cpu()

                start_uv = xyz2uv(start_xyz, pose_w2c_50, intrinsic)
                end_uv = xyz2uv(end_xyz, pose_w2c_50, intrinsic)

                for i in range(start_uv.shape[0]):
                    start = (int(start_uv[i][0].item()), int(start_uv[i][1].item()))
                    end = (int(end_uv[i][0].item()), int(end_uv[i][1].item()))
                    cv2.circle(GT_img_np, start, 2, (255, 0, 0), -1)
                    cv2.circle(GT_img_np, end, 2, (255, 0, 0), -1)
                    cv2.line(GT_img_np, start, end, (0, 0, 255), 1)

                rendered_alpha_np_3 = np.stack([rendered_alpha_np] * 3, axis=-1)
                GT_alpha_np_3 = np.stack([GT_alpha_np] * 3, axis=-1)

                big_img = np.concatenate([GT_img_np, GT_alpha_np_3, rendered_alpha_np_3], axis=1)

                save_img_path_50 = os.path.join(save_img_path, f"render50_{epoch}.png")

                cv2.imwrite(save_img_path_50, big_img)

                orig_edges = np.stack([mst_rows, mst_cols], axis=1).astype(np.int32)
                xyz = branch_model_sigmoid_mst._centroids
                num_points = xyz.shape[0]
                adj = np.zeros((num_points, num_points), dtype=np.uint8)
                # 构建无向邻接矩阵
                for edge in orig_edges:
                    src, dst = edge
                    adj[src, dst] = 1
                    adj[dst, src] = 1

                # 取上半邻接矩阵
                upper_triangular = np.triu(adj, k=1)
                edges = np.argwhere(upper_triangular)
                vertices = xyz.cpu().detach().numpy()
                chk_npz_path = os.path.join(save_img_path, f"render50_{epoch}.npz")
                np.savez(chk_npz_path, nodes=vertices, edges=edges, scene_scale=scene_scale)
                final_print_step_npz_path = chk_npz_path

    # 保存数据
    final_model_path = os.path.join(save_img_path, "final.pth")
    branch_model_sigmoid_mst.save_data(final_model_path)

    npz_path = os.path.join(save_img_path, "losses.npz")
    np.savez(npz_path,
             total_loss=np.array(losses),
             silhouette_loss=np.array(silhouette_losses),
             graph_loss=np.array(graph_losses),
             )

    # rander images
    Ours_save_gt_path = os.path.join(save_img_path, "gt")
    Ours_save_graph_path = os.path.join(save_img_path, "graph")
    Ours_save_graph_mask_path = os.path.join(save_img_path, "graph_mask")

    os.makedirs(Ours_save_gt_path, exist_ok=True)
    os.makedirs(Ours_save_graph_path, exist_ok=True)
    os.makedirs(Ours_save_graph_mask_path, exist_ok=True)

    print('Loading best data....')
    if final_print_best_npz_path:  # 读取最佳数据
        best_data = np.load(final_print_best_npz_path)
    else:
        best_data = np.load(final_print_step_npz_path)

    print('best_data:', best_data.keys())
    nodes = best_data['nodes']
    edges = best_data['edges']

    intrinsic = branch_model_sigmoid_mst.intrinsic_mat
    poses = branch_model_sigmoid_mst.poses
    img_names = branch_model_sigmoid_mst.image_list
    H = branch_model_sigmoid_mst.img_height
    W = branch_model_sigmoid_mst.img_width

    gt_image = branch_model_sigmoid_mst.gt_image
    gt_mask = branch_model_sigmoid_mst.gt_mask_tensor

    print('gt_image:', gt_image.max(), gt_image.min())
    print('gt_mask:', gt_mask.max(), gt_mask.min())

    for i in tqdm(range(branch_model_sigmoid_mst.num_imgs)):
        pose = poses[i]
        xyz_world = torch.from_numpy(nodes).to(torch.float32)
        uv = xyz2uv(xyz_world, pose, intrinsic)

        GT_img_np = gt_image[i]
        GT_img_np = cv2.cvtColor(GT_img_np, cv2.COLOR_BGR2RGB)
        GT_img_np = GT_img_np.astype(np.uint8)

        raw_img_np = GT_img_np.copy()

        gt_mask_np = gt_mask[i].squeeze(-1).cpu().numpy()
        gt_mask_np = gt_mask_np.astype(np.uint8)

        for edge in edges:
            start_idx = edge[0]
            end_idx = edge[1]

            start_point = (int(uv[start_idx][0]), int(uv[start_idx][1]))
            end_point = (int(uv[end_idx][0]), int(uv[end_idx][1]))

            cv2.circle(GT_img_np, start_point, 2, (255, 0, 0), -1)
            cv2.circle(GT_img_np, end_point, 2, (255, 0, 0), -1)
            cv2.line(GT_img_np, start_point, end_point, (0, 0, 255), 1)

        # 保存graph orig 图像
        img_name = img_names[i]
        save_path = os.path.join(Ours_save_graph_path, img_name)
        cv2.imwrite(save_path, GT_img_np.astype(np.uint8))

        # 保存原始图像
        save_raw_path = os.path.join(Ours_save_gt_path, img_name)
        cv2.imwrite(save_raw_path, raw_img_np.astype(np.uint8))

        # save gt mask graph
        # 创建x色透明图层，只有在 mask 区域才显示
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[..., 0] = 0  # B
        overlay[..., 1] = 255  # G
        overlay[..., 2] = 0  # R
        overlay[..., 3] = (gt_mask_np * 255).astype(np.uint8)  # 显示区域的透明度为 255

        # 转换 RGB 为 RGBA，便于合成
        GT_img_np_bgra = cv2.cvtColor(GT_img_np, cv2.COLOR_RGB2RGBA)

        # Alpha blending
        alpha_mask = overlay[..., 3] / 255.0 * 0.2  # 控制透明度强弱
        alpha_mask = alpha_mask[..., np.newaxis]  # (H, W, 1) 才能 broadcast 到 RGB 通道

        img_blend = GT_img_np_bgra.copy()
        img_blend[..., :3] = (1 - alpha_mask) * GT_img_np_bgra[..., :3] + alpha_mask * overlay[..., :3]

        save_blend_path = os.path.join(Ours_save_graph_mask_path, img_name)
        cv2.imwrite(save_blend_path, img_blend.astype(np.uint8))
