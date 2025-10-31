# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
from humanoid.utils.math import wrap_to_pi


import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  Terrain

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class X1DHStandEnv(LeggedRobot):
    '''
    X1DHStandEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (Terrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_stance_mask(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.fixed_joint_angles = cfg.init_state.fixed_joint_angles
        # self.fixed_joint_angles = cfg.init_state.get_fixed_joint_angles()
        # self.last_feet_z = self.cfg.rewards.feet_to_ankle_distance
        # self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        # self.ref_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)      
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        print("=== 索引创建检查 ===")
        print(f"所有DOF名称: {self.dof_names}")
        fixed_joint_names = list(self.fixed_joint_angles.keys())
        print("固定关节配置:", fixed_joint_names)
        print(f"受控关节配置: {list(self.cfg.init_state.default_joint_angles.keys())}")
        
        # 1. 提取受控关节名称
        controlled_joint_names = [
        name for name in self.cfg.init_state.default_joint_angles.keys() 
        if name not in fixed_joint_names  # 只添加这个条件
        ]
    
        # 2. 获取受控关节索引
        self.controlled_dof_indices = [
            self.dof_names.index(name) for name in controlled_joint_names
        ] 
        # 3. 将索引列表转换为PyTorch张量,以便在GPU上高效使用
        
        self.fixed_dof_indices = [
            self.dof_names.index(name) for name in self.fixed_joint_angles
        ]
        
        # log_p("controlled_joint_names:", controlled_joint_names)
        # log_p("controlled_dof_indices:", self.controlled_dof_indices)
        # log_p("fixed_dof_indices:", self.fixed_dof_indices)
        
        
        self.controlled_dof_indices = torch.tensor(self.controlled_dof_indices, device=self.device, 
                                                   dtype=torch.long)
        # 将可控制的关节索引转换为PyTorch张量
        self.ref_action = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        # 初始化参考动作张量
        self.fixed_dof_indices = torch.tensor(self.fixed_dof_indices, device=self.device,dtype=torch.long) 
        # 将固定关节的索引转换为张量        
        self.controlled_dof_pos = torch.zeros(self.num_envs, len(controlled_joint_names),  device=self.device)
        # 初始化可控关节的位置张量

        super()._init_buffers()  # 调用父类初始化方法

        # 初始化步态时间、相位长度和步态起始相位
        self.gait_time = torch.zeros(self.num_envs, len(self.cfg.commands.gait) ,dtype=torch.int, device=self.device, requires_grad=False)
        self.phase_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.gait_start = torch.randint(0, 2, (self.num_envs,)).to(self.device)*0.5
        
        self.last_feet_z = self.cfg.rewards.feet_to_ankle_distance  # 初始化上一时刻脚部的z坐标
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)  # 初始化脚部高度张量
        self.ref_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)  # 初始化参考关节位置
        
        self.phase = torch.zeros(self.num_envs, device=self.device)

    def _init_joint_properties(self):
        """
        [正确实现] 重写关节属性初始化,支持固定关节和控制关节分离
        """

        # 1. 初始化完整DOF(18维)的默认位置张量
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        all_joint_angles = {**self.cfg.init_state.default_joint_angles, **self.fixed_joint_angles}

        # 2. 设置所有关节的默认位置和PD增益
        for i in range(self.num_dof):
            name = self.dof_names[i]
            
            # 设置默认位置
            if name in all_joint_angles:
                self.default_dof_pos[i] = all_joint_angles[name]
            else:
                self.default_dof_pos[i] = 0.0
                print(f"警告: 关节 {name} 未在关节角度配置中找到, 设置为 0")
            
            # 设置PD增益
            found = False
            for dof_name_in_cfg in self.cfg.control.stiffness.keys():
                if dof_name_in_cfg in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name_in_cfg]
                    self.d_gains[i] = self.cfg.control.damping[dof_name_in_cfg]
                    found = True
                    break
            
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                print(f"PD增益未定义: {name}")
        
        # 4. 创建正确的14维默认关节PD目标
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.default_dof_pos_ctl =  torch.index_select(self.default_dof_pos, 1, self.controlled_dof_indices)
      

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)  #angular vel xyz

        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        if self.cfg.commands.sw_switch:
            stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
            self.phase_length_buf[stand_command.bool()] = 0 # set this as 0 for which env is standing
            # self.gait_start is rand 0 or 0.5
            phase = (self.phase_length_buf * self.dt / cycle_time + self.gait_start) * (~stand_command)
        else:
            phase = self.episode_length_buf * self.dt / cycle_time + self.gait_start

        # phase continue increase，if want robot stand, set 0
        return phase

    def _get_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Add double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        # stand mask == 1 means stand leg 
        return stance_mask

    def generate_gait_time(self,envs):
        if len(envs) == 0:
            return

        # rand sample 
        random_tensor_list = []
        for i in range(len(self.cfg.commands.gait)):
            name = self.cfg.commands.gait[i]
            gait_time_range = self.cfg.commands.gait_time_range[name]
            random_tensor_single = torch_rand_float(gait_time_range[0],
                                            gait_time_range[1],
                                            (len(envs), 1),device=self.device)
            random_tensor_list.append(random_tensor_single)

        random_tensor = torch.cat([random_tensor_list[i] for i in range(len(self.cfg.commands.gait))], dim=1)
        current_sum = torch.sum(random_tensor,dim=1,keepdim=True)
        # scaled_tensor store proportion for each gait type
        scaled_tensor = random_tensor * (self.max_episode_length / current_sum)
        scaled_tensor[:,1:] = scaled_tensor[:,:-1].clone()
        scaled_tensor[:,0] *= 0.0
        # self.gait_time accumulate gait_duration_tick
        # self.gait_time = |__gait1__|__gait2__|__gait3__|
        # self.gait_time triger resample gait command
        self.gait_time[envs] = torch.cumsum(scaled_tensor,dim=1).int()
     
    def _resample_commands(self):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        for i in range(len(self.cfg.commands.gait)):
            # if env finish current gait type, resample command for next gait
            env_ids = (self.episode_length_buf == self.gait_time[:,i]).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                # according to gait type create a name
                name = '_resample_' + self.cfg.commands.gait[i] + '_command'
                # get function from self based on name
                resample_command = getattr(self, name)
                # resample_command stands for _resample_stand_command/_resample_walk_sagittal_command/...
                resample_command(env_ids)

    def _resample_stand_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
            
    def _resample_walk_sagittal_command(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)

    def _resample_walk_lateral_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.zeros(len(env_ids), device=self.device)
        else:
            self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)
    
    def _resample_rotate_command(self, env_ids):
        self.commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)
        self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _resample_walk_omnidirectional_command(self,env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.05).unsqueeze(1)
        
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        self.phase_length_buf += 1
        self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            # get all robot surrounding height
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots:
            i = int(self.common_step_counter/self.cfg.domain_rand.update_step)
            if i >= len(self.cfg.domain_rand.push_duration):
                i = len(self.cfg.domain_rand.push_duration) - 1
            duration = self.cfg.domain_rand.push_duration[i]/self.dt
            if self.common_step_counter % self.cfg.domain_rand.push_interval <= duration:
                self._push_robots()
            else:
                self.rand_push_force.zero_()
                self.rand_push_torque.zero_()


    def _get_arm_swing_angles(self):
        """
        根据步态相位计算手臂摆动角度
        返回左右肩膀pitch关节的目标角度增量
        """
        
        # 计算摆臂幅度,基于行走速度
        speed = torch.norm(self.commands[:, :2], dim=1)
        max_swing_amplitude = 0.4 # 最大摆臂幅度 (弧度)
        swing_amplitude = torch.clamp(speed * 0.5, 0, max_swing_amplitude)
        
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        swing_amplitude[stand_command] = 0.0  # 如果是站立命令,摆臂幅度为0
        
        # 计算相位正弦值
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        
        # 手臂摆动：与腿部相位相反 (自然行走模式)
        left_arm_swing = -sin_pos * swing_amplitude
        right_arm_swing = sin_pos * swing_amplitude
        
        return torch.stack((left_arm_swing, right_arm_swing), dim=1)

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        arm_swing = self._get_arm_swing_angles()

        self.ref_dof_pos = torch.zeros((self.num_envs, len(self.controlled_dof_indices)), device=self.device)
        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 0] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[0]
        self.ref_dof_pos[:, 1] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[1]
        self.ref_dof_pos[:, 2] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[2]
        self.ref_dof_pos[:, 3] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[3]
        self.ref_dof_pos[:, 4] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[4]
        self.ref_dof_pos[:, 5] = -sin_pos_l * self.cfg.rewards.final_swing_joint_delta_pos[5]
        self.ref_dof_pos[:, 6] = arm_swing[:, 0]  # left arm swing
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r *  self.cfg.rewards.final_swing_joint_delta_pos[6]
        self.ref_dof_pos[:, 8] = sin_pos_r *  self.cfg.rewards.final_swing_joint_delta_pos[7]
        self.ref_dof_pos[:, 9] = sin_pos_r *  self.cfg.rewards.final_swing_joint_delta_pos[8]
        self.ref_dof_pos[:, 10] = sin_pos_r *  self.cfg.rewards.final_swing_joint_delta_pos[9]
        self.ref_dof_pos[:, 11] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[10]
        self.ref_dof_pos[:, 12] = sin_pos_r * self.cfg.rewards.final_swing_joint_delta_pos[11]
        self.ref_dof_pos[:, 13] = arm_swing[:, 0]  # right arm swing

        # double support phase ref pos = 0
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.
        
        # if use_ref_actions=True, action += ref_action
        self.ref_action = 2 * self.ref_dof_pos
        
        # self.ref_dof_pos set ref dof pos for swing leg, ref_dof_pos=0 for stance leg
        self.ref_dof_pos += self.default_dof_pos_ctl


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: self.cfg.env.num_commands] = 0.  # commands
        noise_vec[self.cfg.env.num_commands: self.cfg.env.num_commands+self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[self.cfg.env.num_commands+self.num_actions: self.cfg.env.num_commands+2*self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[self.cfg.env.num_commands+2*self.num_actions: self.cfg.env.num_commands+3*self.num_actions] = 0.  # previous actions
        noise_vec[self.cfg.env.num_commands+3*self.num_actions: self.cfg.env.num_commands+3*self.num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[self.cfg.env.num_commands+3*self.num_actions + 3: self.cfg.env.num_commands+3*self.num_actions + 6] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec

    def step(self, actions):
        # 创建18维动作张量
        full_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        
        # 将14维动作映射到对应位置
        full_actions[:, self.controlled_dof_indices] = actions
        
        # 对固定关节设置默认角度
        fixed_joint_positions = torch.zeros((self.num_envs, len(self.fixed_dof_indices)), device=self.device)
        for i, joint_name in enumerate(self.fixed_joint_angles.keys()):
            fixed_joint_positions[:, i] = self.fixed_joint_angles[joint_name]
        full_actions[:, self.fixed_dof_indices] = fixed_joint_positions
    
        if self.cfg.env.use_ref_actions:
            full_actions += self.ref_action  # 如果使用参考动作,将参考动作添加到输入的动作中
        # 调用父类的step方法处理物理仿真
        return super().step(full_actions) 

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        # critic no lag
        diff = self.controlled_dof_pos - self.ref_dof_pos
        # 73+8=81 dim privileged obs
        privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3 0-4
            (self.controlled_dof_pos - self.default_dof_pos_ctl) * self.obs_scales.dof_pos, # 12+2 5-18
            self.controlled_dof_vel * self.obs_scales.dof_vel,  # 12+2 19-32
            self.actions,  # 12+2 33-46
            diff,  # 12+2 47-60
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3 61-63
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3 64-66
            self.base_euler_xyz * self.obs_scales.quat,  # 3 67-69
            self.rand_push_force[:, :2],  # 2 70-71
            self.rand_push_torque,  # 3 72-74
            self.env_frictions,  # 1 75
            self.body_mass / 10.,  # 1 # sum of all fix link mass 76
            stance_mask,  # 2 77-78
            contact_mask,  # 2 79-80
        ), dim=-1)
        
        # random add dof_pos and dof_vel same lag 延迟模拟
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            # 使用索引从完整状态中提取受控关节的状态
            lagged_dof_state = self.dof_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            full_dof_pos = lagged_dof_state[:, :self.num_dof]
            full_dof_vel = lagged_dof_state[:, self.num_dof:]
            
            # log_p(f"\nfull_dof_pos shape: {full_dof_pos.shape}, \nfull_dof_vel shape: {full_dof_vel.shape}")
            
            self.lagged_dof_pos = torch.index_select(full_dof_pos, 1, self.controlled_dof_indices)
            self.lagged_dof_vel = torch.index_select(full_dof_vel, 1, self.controlled_dof_indices)        # random add dof_pos and dof_vel different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()

            # 从18维缓冲区读取,然后切片为14维
            full_lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :, self.dof_pos_lag_timestep.long()]
            self.lagged_dof_pos = torch.index_select(full_lagged_dof_pos, 1, self.controlled_dof_indices)
              
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            # 从18维缓冲区读取,然后切片为14维
            full_lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
            self.lagged_dof_vel = torch.index_select(full_lagged_dof_vel, 1, self.controlled_dof_indices)
            # self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
        
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.controlled_dof_pos
            self.lagged_dof_vel = self.controlled_dof_vel

        # imu lag, including rpy and omega   imu传感器延迟
        if self.cfg.domain_rand.add_imu_lag:    
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.int()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz[:,-3:]
        
        # obs q and dq      构建观测，标准化数据
        q = (self.lagged_dof_pos - self.default_dof_pos_ctl) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel  

        # 47   组合策略观测+6
        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12+2
            dq,  # 12+2
            self.actions,   # 12+2
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.env.num_single_obs == 48:            #站立扩展检测
            stand_command = (torch.norm(self.commands[:, :3], dim=1, keepdim=True) <= self.cfg.commands.stand_com_threshold)
            obs_buf = torch.cat((obs_buf, stand_command),dim=1)
            
        if self.cfg.terrain.measure_heights:         #地形感知特权观测
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            privileged_obs_buf = torch.cat((privileged_obs_buf.clone(), heights), dim=-1)
        
        if self.add_noise:  
            # add obs noise
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.critic_history.append(privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset rand dof_pos and dof_vel=0
        self._reset_dofs(env_ids)

        # reset base position
        self._reset_root_states(env_ids)
        
        # Randomize joint parameters, like torque gain friction ...
        self.randomize_dof_props(env_ids)
        self._refresh_actor_dof_props(env_ids)
        self.randomize_lag_props(env_ids)
        
        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_rigid_state[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.phase_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # rand 0 or 0.5
        self.gait_start[env_ids] = torch.randint(0, 2, (len(env_ids),)).to(self.device)*0.5
        
        #resample command
        self.generate_gait_time(env_ids)
        self._resample_commands()
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # fix reset gravity bug
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.feet_quat = self.rigid_state[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_xyz_tensor(self.feet_quat)
        
        # clear obs history buffer and privileged obs buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0
        

# ================================================ Rewards ================================================== #
    def _reward_ref_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos[:, self.controlled_dof_indices].clone()
        pos_target = self.ref_dof_pos.clone()

        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        pos_target[stand_command] = self.default_dof_pos_ctl.clone()

        diff = joint_pos - pos_target
        # 对腿部关节和手臂关节分别处理
        leg_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # 腿部关节索引
        leg_diff = diff[:, leg_indices]  # 腿部关节差异
        arm_diff = diff[:, [6, 13]]  # 如果索引6和13都是手臂关节
        
        # 腿部关节奖励(权重更高)
        leg_reward = torch.exp(-2 * torch.norm(leg_diff, dim=1)) - 0.2 * torch.norm(leg_diff, dim=1).clamp(0, 0.5)
        
        # 手臂关节奖励(权重较低)
        arm_reward = torch.exp(-1 * torch.norm(arm_diff, dim=1)) - 0.1 * torch.norm(arm_diff, dim=1).clamp(0, 0.3)
        
        # 组合奖励,腿部权重0.8,手臂权重0.2
        total_reward = 0.8 * leg_reward + 0.2 * arm_reward
        
        return total_reward
    
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact conditions.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) < 0.05] = 1
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_stance_mask().clone()
        stance_mask[torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold] = 1
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        # 计算关节位置与默认位置的差异
        default_target = self.default_joint_pd_target.squeeze()
        joint_diff = self.dof_pos - default_target
        # 分别处理腿部和手臂关节
        leg_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # 腿部关节索引
        arm_indices = [6, 13]  # 手臂关节索引
        leg_diff = joint_diff[:, leg_indices]  # 腿部关节差异
        arm_diff = joint_diff[:, arm_indices]  # 手臂关节差异
        left_yaw_roll = leg_diff[:, [1,2,5]]
        right_yaw_roll = leg_diff[:, [7,8,11]]

        # 计算腿部偏航和侧倾的总体偏差
        leg_yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        leg_yaw_roll = torch.clamp(leg_yaw_roll - 0.1, 0, 50)
        
        # 计算手臂偏差
        arm_deviation = torch.norm(arm_diff, dim=1)
        
        # 计算奖励
        leg_reward = torch.exp(-leg_yaw_roll * 100) - 0.01 * torch.norm(leg_diff, dim=1)
        arm_reward = torch.exp(-arm_deviation * 20) - 0.005 * arm_deviation
        
        # 组合奖励
        reward = 0.9 * leg_reward + 0.1 * arm_reward
        
        # --- 修改部分：仅在站立命令时应用此奖励 ---
        # 使用 torch.where，当 stand_command 为 True 时返回计算的奖励，否则返回 0
        final_reward = torch.where(stand_command, reward, torch.zeros_like(reward))
        
        return final_reward
        # yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        # yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        # return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_stance_mask()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.rewards.feet_to_ankle_distance)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        r = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error
        return r
    
    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        lin_vel_error_square = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error_abs = torch.sum(torch.abs(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        r_square = torch.exp(-lin_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-lin_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        ang_vel_error_square = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_abs = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        r_square = torch.exp(-ang_vel_error_square * self.cfg.rewards.tracking_sigma)
        r_abs = torch.exp(-ang_vel_error_abs * self.cfg.rewards.tracking_sigma * 2)
        r = torch.where(stand_command, r_abs, r_square)

        return r 
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.feet_to_ankle_distance
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_stance_mask()

        # feet height should larger than target feet height at the peak
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height) * (self.feet_height < self.cfg.rewards.target_feet_height_max)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.05)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        # 分别处理腿部和手臂扭矩
        leg_torques = self.torques[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]  # 腿部扭矩
        arm_torques = self.torques[:, [6, 13]]  # 手臂扭矩
        
        # 腿部扭矩惩罚权重更高
        leg_penalty = torch.sum(torch.square(leg_torques), dim=1)
        arm_penalty = torch.sum(torch.square(arm_torques), dim=1) * 0.5  # 手臂扭矩惩罚权重较低
        
        return leg_penalty + arm_penalty
        # return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_ankle_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        ankle_idx = [6,7,12,13]  # 踝关节索引
        return torch.sum(torch.square(self.torques[:,ankle_idx]), dim=1)
    
    def _reward_feet_rotation(self):
        feet_euler_xyz = self.feet_euler_xyz
        rotation = torch.sum(torch.square(feet_euler_xyz[:,:,:2]),dim=[1,2])
        # rotation = torch.sum(torch.square(feet_euler_xyz[:,:,1]),dim=1)
        r = torch.exp(-rotation*15)
        return r

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        
        leg_actions = torch.abs(self.actions[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])  # 腿部动作
        arm_actions = torch.abs(self.actions[:, [6, 13]])  # 手臂动作
        # 计算动作绝对值的惩罚项
        term_3_leg = 0.05 * torch.sum(leg_actions, dim=1)
        term_3_arm = 0.02 * torch.sum(arm_actions, dim=1)  # 手臂惩罚较轻
        term_3 = term_3_leg + term_3_arm

        # term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_stand_still(self):
        # penalize motion at zero commands
        stand_command = (torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_com_threshold)
        controlled_dof_pos = self.dof_pos[:, self.controlled_dof_indices] 

        r = torch.exp(-torch.sum(torch.square(controlled_dof_pos- self.default_dof_pos_ctl), dim=1))
        r = torch.where(stand_command, r.clone(),
                        torch.zeros_like(r))
        return r
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_arm_swing_naturalness(self):
        """
        奖励手臂自然摆动,惩罚不自然的手臂运动
        """
        # 获取当前手臂角度(前两个关节)
        left_arm_pos = self.controlled_dof_pos[:, 6]   # 左肩膀pitch
        right_arm_pos = self.controlled_dof_pos[:, 13]  # 右肩膀pitch
        
        # 获取参考手臂角度
        left_arm_ref = self.ref_dof_pos[:, 6]
        right_arm_ref = self.ref_dof_pos[:, 13]
        
        # 计算手臂位置与参考位置的差异
        left_arm_error = torch.abs(left_arm_pos - left_arm_ref)
        right_arm_error = torch.abs(right_arm_pos - right_arm_ref)
        
        # 计算总体手臂摆动误差
        total_arm_error = left_arm_error + right_arm_error
        
        # 使用指数函数计算奖励,误差越小奖励越高
        reward = torch.exp(-total_arm_error * 5.0)
        
        # 只在行走时给予奖励,站立时不考虑
        speed = torch.norm(self.commands[:, :2], dim=1)
        moving_mask = speed > self.cfg.commands.stand_com_threshold
        reward = reward * moving_mask.float()
        
        return reward