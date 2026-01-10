"""

"""

import torch
import pytorch_kinematics as pk
import numpy as np


class LinkerHandModel(torch.nn.Module):
    """
    LinkerHand L20 Hand Model for DexGraspNet-style grasp generation
    """

    def __init__(
        self,
        urdf_path,
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.urdf_path = urdf_path

        # ------------------------------------------------
        # 1. Build kinematic chain
        # ------------------------------------------------
        with open(self.urdf_path, "rb") as f:
            urdf_bytes = f.read()

        self.chain = pk.build_chain_from_urdf(urdf_bytes).to(
            dtype=torch.float, device=self.device
        )

        # ------------------------------------------------
        # 2. Joint information
        # ------------------------------------------------
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = len(self.joint_names)   # = 21

        # ------------------------------------------------
        # 3. Active / Passive DOF split
        # （你可以根据官方文档微调）
        # ------------------------------------------------
        self.passive_joint_names = [
            "index_dip",
            "middle_dip",
            "ring_dip",
            "pinky_dip",
            "thumb_dip",
        ]

        self.active_joint_names = [
            j for j in self.joint_names if j not in self.passive_joint_names
        ]

        self.active_dof_indices = [
            self.joint_names.index(j) for j in self.active_joint_names
        ]
        self.passive_dof_indices = [
            self.joint_names.index(j) for j in self.passive_joint_names
        ]

        # ------------------------------------------------
        # 4. Joint limits
        # （保守设置，后面可以精调）
        # ------------------------------------------------
        lower = []
        upper = []

        for j in self.joint_names:
            if "roll" in j:
                lower.append(-0.5)
                upper.append(0.5)
            elif "yaw" in j:
                lower.append(-1.0)
                upper.append(1.0)
            else:  # pitch / pip / dip
                lower.append(0.0)
                upper.append(1.5)

        self.joint_lower_limits = torch.tensor(lower, device=self.device)
        self.joint_upper_limits = torch.tensor(upper, device=self.device)

        # ------------------------------------------------
        # 5. Fingertip links（极其重要）
        # ------------------------------------------------
        self.fingertip_links = {
            "thumb": "thumb_link3",
            "index": "index_link3",
            "middle": "middle_link3",
            "ring": "ring_link3",
            "pinky": "pinky_link3",
        }

    # ==================================================
    # Forward kinematics
    # ==================================================
    def forward_kinematics(self, q):
        """
        q: (B, 21)
        return: dict {link_name: Transform}
        """
        return self.chain.forward_kinematics(q)

    # ==================================================
    # Fingertip positions
    # ==================================================
    def get_fingertip_positions(self, q):
        fk = self.forward_kinematics(q)

        tips = []
        for name in self.fingertip_links.values():
            T = fk[name].get_matrix()[:, :3, 3]
            tips.append(T)

        return torch.stack(tips, dim=1)  # (B, 5, 3)

    # ==================================================
    # Joint limit penalty（给 optimizer 用）
    # ==================================================
    def joint_limit_loss(self, q):
        lower_violation = torch.relu(self.joint_lower_limits - q)
        upper_violation = torch.relu(q - self.joint_upper_limits)
        return (lower_violation + upper_violation).pow(2).mean()

    # ==================================================
    # Passive DOF regularization
    # ==================================================
    def passive_dof_loss(self, q):
        q_passive = q[:, self.passive_dof_indices]
        return (q_passive ** 2).mean()

