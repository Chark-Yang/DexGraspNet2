"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: generate grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys

#把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#print(sys.path)
import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method

try:
    #用spawn方法启动多进程，spawn为每个进程初始化CUDA
    set_start_method('spawn')
except RuntimeError:
    pass

#避免OpenMP/MKL重复加载导致崩溃
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')


def generate(args_list):

    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare models

    n_objects = len(object_code_list)

    worker = multiprocessing.current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')

    #HandModel内部包含手模型的关节结构及网格的路径，接触点和穿透点
    #hand_pose是优化变量
    hand_model = HandModel(
        mjcf_path='mjcf/shadow_hand_wrist_free.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device
    )
    #object_model是被抓物体；ObjectModel是一个类
    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_samples=2000, 
        device=device
    )

    object_model.initialize(object_code_list)
    #抓取初始化
    initialize_convex_hull(hand_model, object_model, args)
    
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'step_size': args.step_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device
    }
    #构造模拟退火优化器对象
    # **optim_config 是把字典里的键值对展开成关键字参数传给函数
    optimizer = Annealing(hand_model, **optim_config)

    # optimize
    #定义能量函数，抓取能量的加权系数
    #w_dis是距离能量的权重，w_pen是穿透能量的权重，w_spen是自穿透能量的权重，w_joints是关节能量的权重
    weight_dict = dict(
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
    )
    #计算初始能量，其余是分项能量
    energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    #初始能量的梯度
    #.sum()是把 batch 维度加起来，变成标量，才能调用 backward()
    #.backward()反向传播，计算梯度
    #retain_graph=True 保留计算图，以便后续继续调用 backward()
    energy.sum().backward(retain_graph=True)

    #模拟退火主循环
    for step in range(1, args.n_iter + 1):
        #随机perturb手的姿态
        s = optimizer.try_step()
        
        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            #能量下降则接受，能量升高以一定概率接受
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]


    # save results

    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
        'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
        'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
        'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
        'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    ]
    for i, object_code in enumerate(object_code_list):
        data_list = []
        for j in range(args.batch_size_each):
            idx = i * args.batch_size_each + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos.update(dict(zip(rot_names, euler)))
            qpos.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
            rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
            euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
            qpos_st.update(dict(zip(rot_names, euler)))
            qpos_st.update(dict(zip(translation_names, hand_pose[:3].tolist())))
            data_list.append(dict(
                scale=scale,
                qpos=qpos,
                qpos_st=qpos_st,
                energy=energy[idx].item(),
                E_fc=E_fc[idx].item(),
                E_dis=E_dis[idx].item(),
                E_pen=E_pen[idx].item(),
                E_spen=E_spen[idx].item(),
                E_joints=E_joints[idx].item(),
            ))
        #保存结果（生成数据集）
        np.save(os.path.join(args.result_path, object_code + '.npy'), data_list, allow_pickle=True)


if __name__ == '__main__':
    #定义这个脚本可以接受哪些命令行参数，把在终端输入的--xxx 变成 Python 里的 args.xxx
    #创建一个“参数说明书”对象，告诉 Python：这个程序可以接收哪些命令行参数
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/graspdata", type=str)
    parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--batch_size_each', default=500, type=int)
    parser.add_argument('--max_total_batch_size', default=1000, type=int)
    parser.add_argument('--n_iter', default=6000, type=int)
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.2, type=float)
    parser.add_argument('--distance_upper', default=0.3, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)

    #从命令行读取你输入的--xxx参数，并存到args里
    args = parser.parse_args()

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f'gpu_list: {gpu_list}')

    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #创建结果保存目录
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    #检查数据集路径是否存在,否则报错
    if not os.path.exists(args.data_root_path):
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    #object_code_list 和 all 
    if (args.object_code_list is not None) + args.all != 1:
        raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified')
    
    #object_code_list_all是数据集里所有物体的列表;
    #如果指定了--todo参数，则只处理 todo.txt 里的物体,否则处理 data_root_path 目录下的所有物体
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path)
    
    #object_code_list是本次要处理的物体列表; object_code_list_all是数据集里所有物体的列表
    #前者是后者的子集，否则报错；如果没指定前者，则用后者
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path')
    else:
        object_code_list = object_code_list_all
    #指定是否覆盖
    #不覆盖，就移除，即不处理那些已经存在结果文件的物体
    #覆盖，不移除，都生成抓取结果，这样就会覆盖原有文件
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + '.npy')):
                object_code_list.remove(object_code)

    #batch_size_each 应该比 max_total_batch_size 小，否则报错
    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'n_objects: {len(object_code_list)}')
    
    # generate

    random.seed(args.seed)
    random.shuffle(object_code_list)
    #任务切分，控制单个 GPU 同时处理多少个物体
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [object_code_list[i: i + objects_each] for i in range(0, len(object_code_list), objects_each)]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(p.imap(generate, process_args), total=len(process_args), desc='generating', maxinterval=1000)
        list(it)
