"""
Last modified date: 2023.02.23
Author: Ruicheng Wang, Jialiang Zhang
Description: Class ObjectModel
"""

import os
import trimesh as tm
import plotly.graph_objects as go
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np

from torchsdf import index_vertices_by_faces, compute_sdf


class ObjectModel:

    def __init__(self, data_root_path, batch_size_each, num_samples=2000, device="cuda"):
        """
        Create a Object Model
        
        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_face_verts_list = None
        self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float, device=self.device)

    def initialize(self, object_code_list):
        """
        Initialize Object Model with list of objects
        
        Choose scales, load meshes, sample surface points
        
        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        # 检查输入的 object_code_list 是否是列表；如果不是（例如单个字符串），将其转换为列表
        # 这样可以统一处理单个对象或多个对象的情况
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
            
        # 将处理后的对象代码列表保存到实例变量中
        self.object_code_list = object_code_list

        # 初始化空列表，用于存储每个对象的尺度张量、网格、面顶点和表面点
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        
        # 遍历每个对象代码
        for object_code in object_code_list:
            # 从预定义的尺度选择列表 (self.scale_choice) 中随机选择一个尺度
            # torch.randint 生成随机索引，self.batch_size_each 是每个对象的批次大小
            # 结果是一个张量，表示该对象的尺度（用于缩放网格）
            self.object_scale_tensor.append(self.scale_choice[torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)])
            
            # 使用 trimesh 加载对象的网格文件
            # 路径拼接：data_root_path / object_code / "coacd" / "decomposed.obj"
            # force="mesh" 确保加载为网格
            # process=False 避免额外处理，明确禁止 trimesh：平移 mesh，重心对齐，PCA 旋转，单位缩放
            self.object_mesh_list.append(tm.load(os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False))
            # 将网格的顶点转换为 PyTorch 张量，并移动到指定设备 (self.device)
            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            # 将网格的面（faces）转换为长整型张量，并移动到设备
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
             # 使用 index_vertices_by_faces 函数根据面索引顶点，生成面顶点列表
            # 这用于后续的 SDF（符号距离场）计算
            self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            # 如果需要采样表面点 (self.num_samples != 0)
            if self.num_samples != 0:
                # 将顶点和面再次转换为张量（确保数据类型和设备）
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                # 创建 PyTorch3D 的 Meshes 对象，用于几何操作
                mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                # 从网格采样密集点云：采样 100 * self.num_samples 个点
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)
                # 使用最远点采样 (FPS) 从密集点云中选择 self.num_samples 个点
                # [0][0] 提取第一个批次的第一个网格的点
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
                # 将表面点转换为 float 类型并移动到设备
                surface_points.to(dtype=float, device=self.device)
                # 将采样到的表面点添加到列表
                self.surface_points_tensor.append(surface_points)
        # 将所有对象的尺度张量堆叠成一个张量：形状为 (n_objects, batch_size_each)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        
        # 如果采样了表面点，将它们堆叠并重复扩展
        # repeat_interleave 用于每个对象重复 batch_size_each 次
        # 最终形状：(n_objects * batch_size_each, num_samples, 3)
        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals
        
        Interiors are positive, exteriors are negative
        
        Use our modified Kaolin package
        
        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes
        
        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

    def get_plotly_data(self, i, color='lightgreen', opacity=0.5, pose=None):
        """
        Get visualization data for plotly.graph_objects
        
        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix
        
        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0],y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]
