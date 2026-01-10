"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))



import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #命令行参数
    parser.add_argument('--object_code', type=str, default='sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='/home/chark/DexGraspNet/data/dataset/')
    args = parser.parse_args()

    device = 'cpu'



    # load results
    data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[args.num]
    qpos = data_dict['qpos']
    # 旋转3自由度转换为矩阵，再转换为所需格式
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    #手部姿态张量：3平移 + 6旋转 + 22关节
    hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
    # 初始手部姿态张量，qpos_st是初始姿态
    if 'qpos_st' in data_dict:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st[name] for name in joint_names], dtype=torch.float, device=device)

    # hand model
    hand_model = HandModel(
        mjcf_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mjcf', 'shadow_hand_wrist_free.xml'),
        mesh_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mjcf', 'meshes'),
        contact_points_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mjcf', 'contact_points.json'),
        penetration_points_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mjcf', 'penetration_points.json'),
        device=device
    )

    # object model
    object_model = ObjectModel(
        data_root_path='/home/chark/DexGraspNet/data/meshdata/meshdata/',
        batch_size_each=1,
        num_samples=2000, 
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)

    #打印物体边界框的顶点信息，axis=0 表示沿每个坐标轴计算最小值、最大值和均值；
    #axis=1 表示沿每个顶点计算,即计算每个顶点的3个坐标的最小值、最大值和均值，没有几何意义
    mesh = object_model.object_mesh_list[-1]

    print(mesh.vertices.min(axis=0))
    print(mesh.vertices.max(axis=0))
    print(mesh.vertices.mean(axis=0))


    # visualize

    if 'qpos_st' in data_dict:
        hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        #获取初始手部姿态的 Plotly 可视化数据
        hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
    else:
        hand_st_plotly = []
    #添加批次维度 unsqueeze(0)，使其成为 (1, ...) 形状，用于单样本处理
    hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
    #创建 Plotly 图表对象 fig，将初始手、最终手和物体的可视化数据合并（列表拼接）。这会生成一个 3D 场景，显示手和物体的相对位置。
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
    #如果有，表示有能量计算结果
    #提取能量数据：E_fc（摩擦力能量）、E_dis（距离能量）、E_pen（穿透能量）、E_spen（自穿透能量）、E_joints（关节能量），并四舍五入到指定小数位
    if 'energy' in data_dict:
        energy = data_dict['energy']
        E_fc = round(data_dict['E_fc'], 3)
        E_dis = round(data_dict['E_dis'], 5)
        E_pen = round(data_dict['E_pen'], 5)
        E_spen = round(data_dict['E_spen'], 5)
        E_joints = round(data_dict['E_joints'], 5)
        result = f'Index {args.num}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
        #在图表中添加文本注释，位置在图表的中间下方（x=0.5, y=0.1，基于纸张坐标系），显示能量信息。
        fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
    #更新图表布局，设置 3D 场景的纵横比模式为 'data'（基于数据范围自动调整比例，确保手和物体比例正确）。
    fig.update_layout(scene_aspectmode='data')
    fig.show()
