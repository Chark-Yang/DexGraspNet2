"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: Class Annealing optimizer
"""

import torch


class Annealing:
    
    def __init__(self, hand_model, switch_possibility=0.5, starting_temperature=18, temperature_decay=0.95, annealing_period=30,
                 step_size=0.005, stepsize_period=50, mu=0.98, device='cpu'):
        """
        Create a optimizer
        
        Use random resampling to update contact point indices
        
        Use RMSProp to update translation, rotation, and joint angles, use step size decay
        
        Use Annealing to accept / reject parameter updates
        
        Parameters
        ----------
        hand_model: hand_model.HandModel
        switch_possibility: float
            possibility to resample each contact point index each step
        starting_temperature: float
        temperature_decay: float
            temperature decay rate and step size decay rate
        annealing_period: int
        step_size: float
        stepsize_period: int
        mu: float
            `1 - decay_rate` of RMSProp

        hand_model: 手部模型对象，包含手部姿态、接触点等信息

        switch_possibility=0.5: 每次迭代中，每个接触点被重新采样的概率

        starting_temperature=18: 模拟退火的初始温度

        temperature_decay=0.95: 温度衰减率

        annealing_period=30: 每30步衰减一次温度

        step_size=0.005: RMSProp的初始学习率

        stepsize_period=50: 每50步衰减一次学习率

        mu=0.98: RMSProp的动量参数（衰减率）

        device='cpu': 计算设备（CPU/GPU）
        """

        self.hand_model = hand_model
        self.device = device
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(temperature_decay, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)
        self.step = 0

        self.old_hand_pose = None
        self.old_contact_point_indices = None
        self.old_global_transformation = None
        self.old_global_rotation = None
        self.old_current_status = None
        self.old_contact_points = None
        self.old_grad_hand_pose = None
        #ema_grad_hand_pose: 用于RMSProp的梯度平方的指数移动平均
        self.ema_grad_hand_pose = torch.zeros(self.hand_model.n_dofs + 9, dtype=torch.float, device=device)

    def try_step(self):
        """
        Try to update translation, rotation, joint angles, and contact point indices
        
        Returns
        -------
        s: torch.Tensor
            current step size
        """
        #
        s = self.step_size * self.temperature_decay ** torch.div(self.step, self.step_size_period, rounding_mode='floor')
        step_size = torch.zeros(*self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device) + s

        self.ema_grad_hand_pose = self.mu * (self.hand_model.hand_pose.grad ** 2).mean(0) + \
            (1 - self.mu) * self.ema_grad_hand_pose

        hand_pose = self.hand_model.hand_pose - \
            step_size * self.hand_model.hand_pose.grad / (torch.sqrt(self.ema_grad_hand_pose) + 1e-6)
        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(self.hand_model.n_contact_candidates, size=[switch_mask.sum()], device=self.device)

        self.old_hand_pose = self.hand_model.hand_pose
        self.old_contact_point_indices = self.hand_model.contact_point_indices
        self.old_global_transformation = self.hand_model.global_translation
        self.old_global_rotation = self.hand_model.global_rotation
        self.old_current_status = self.hand_model.current_status
        self.old_contact_points = self.hand_model.contact_points
        self.old_grad_hand_pose = self.hand_model.hand_pose.grad
        self.hand_model.set_parameters(hand_pose, contact_point_indices)

        self.step += 1

        return s

    def accept_step(self, energy, new_energy):
        """
        Accept / reject updates using annealing
        
        Returns
        -------
        accept: (N,) torch.BoolTensor
        temperature: torch.Tensor
            current temperature
        """

        batch_size = energy.shape[0]
        #计算当前温度；温度随着步数增加而指数衰减
        #**是指数衰减操作，torch.div用于整数除法并向下取整
        #self.temperature_decay ** ... 是温度衰减率的...次方
        temperature = self.starting_temperature * self.temperature_decay ** torch.div(self.step, self.annealing_period, rounding_mode='floor')
        #alpha是【0,1）之间的随机数
        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        #根据概率决定是否接受更新
        accept = alpha < torch.exp((energy - new_energy) / temperature)

        with torch.no_grad():
            #对于不接受更新的样本，恢复旧的参数值
            #～是按位取反操作，即将accept中的True变为False，False变为True
            reject = ~accept
            #使用布尔索引，只选取reject为True的样本进行恢复
            self.hand_model.hand_pose[reject] = self.old_hand_pose[reject]
            #恢复接触点索引
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]
            #恢复全局平移和旋转
            self.hand_model.global_translation[reject] = self.old_global_transformation[reject]
            self.hand_model.global_rotation[reject] = self.old_global_rotation[reject]
            #更新当前状态，[:, 9:]：取所有行（所有样本），从第9列开始到最后；前9个参数是平移和旋转，后面的参数是手部关节角度
            #因为有些样本的参数被恢复到了旧值，需要重新计算它们的状态
            self.hand_model.current_status = self.hand_model.chain.forward_kinematics(self.hand_model.hand_pose[:, 9:])
            #恢复被拒绝样本的接触点坐标
            self.hand_model.contact_points[reject] = self.old_contact_points[reject]
            #恢复被拒绝样本的梯度
            self.hand_model.hand_pose.grad[reject] = self.old_grad_hand_pose[reject]

        return accept, temperature
    #防止梯度累积，进行梯度清零操作
    def zero_grad(self):
        """
        Sets the gradients of translation, rotation, and joint angles to zero
        """
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()
