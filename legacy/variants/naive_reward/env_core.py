import numpy as np
import math
import os
import torch
import matplotlib
# 强制使用无头模式，防止多线程绘图报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class EnvCore(object):
    """
    【消融实验专属环境】：验证奖励函数的必要性 (Naive Reward)
    - 移除了通信香农公式
    - 移除了雷达感知公式
    - 移除了防碰撞惩罚
    - 仅保留最基础的“距离惩罚”
    """
    def __init__(self):
        # ================= 1. 基础物理设定 =================
        self.agent_num = 3  
        self.user_num = 2   
        
        self.obs_dim = 4 + (self.agent_num - 1) * 2 + self.user_num * 2 + 4
        self.action_dim = 2 

        self.map_size = 100.0
        self.max_speed = 5.0
        # 这里的通信参数虽然定义了，但在这个“天真奖励”实验中不会用到
        self.power_tx = 10.0   
        self.noise_power = 0.1 
        self.safe_dist = 2.0   
        self.max_steps = 200   
        
        # ================= 2. 文件夹强制分流 =================
        # 强制将本实验的图片保存在独立文件夹，防止污染之前的成果
        self.save_img_dir = "Results_TRAJ_Naive_Reward"
        if not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir, exist_ok=True)

        self.episode_counter = 0
        self.history_agents = []
        self.history_users =[]

    def reset(self):
        self.step_count = 0
        self.agents_pos = np.random.uniform(10, self.map_size - 10, (self.agent_num, 2))
        self.agents_v = np.zeros((self.agent_num, 2))
        self.users_pos = np.random.uniform(20, self.map_size - 20, (self.user_num, 2))
        
        self.history_agents =[self.agents_pos.copy()]
        self.history_users = [self.users_pos.copy()]
        return self.get_obs()

    def step(self, actions):
        self.step_count += 1
        actions = np.array(actions) 
        
        # 1. 物理更新
        for i in range(self.agent_num):
            self.agents_v[i] += actions[i] * 1.0 
            speed = np.linalg.norm(self.agents_v[i])
            if speed > self.max_speed:
                self.agents_v[i] = (self.agents_v[i] / speed) * self.max_speed
            self.agents_pos[i] += self.agents_v[i] * 0.1 
            self.agents_pos[i] = np.clip(self.agents_pos[i], 0, self.map_size)
            
        self.history_agents.append(self.agents_pos.copy())
        self.history_users.append(self.users_pos.copy())

        # ================= 核心修改：天真奖励 (Naive Reward) =================
        rewards =[]
        for i in range(self.agent_num):
            reward_i = 0.0
            # 传统算法只看距离，越近越好（简单的负距离惩罚）
            # 不考虑香农公式，不考虑雷达4次方衰减，直接去掉了防撞惩罚！
            for j in range(self.user_num):
                dist = np.linalg.norm(self.agents_pos[i] - self.users_pos[j])
                reward_i -= dist * 0.1  # 离用户越远，扣分越多
                
            rewards.append([reward_i]) 
        # =====================================================================

        # 3. 定期绘图 (每200回合)
        if self.step_count >= self.max_steps:
            self.episode_counter += 1
            if self.episode_counter % 200 == 0:
                self.render_trajectory()

        obs = self.get_obs()
        dones = [self.step_count >= self.max_steps] * self.agent_num
        infos = [{"rew": 0}] * self.agent_num
        return obs, rewards, dones, infos

    def get_obs(self):
        """生成观测向量 (植入4维环境干扰噪声)"""
        obs =[]
        for i in range(self.agent_num):
            ob_i = list(self.agents_pos[i]) + list(self.agents_v[i]) 
            for k in range(self.agent_num): 
                if i != k: ob_i += list(self.agents_pos[k])
            for j in range(self.user_num): 
                ob_i += list(self.users_pos[j])
            
            # 保留噪声，控制变量
            noise = np.random.uniform(0, self.map_size, 4).tolist()
            ob_i += noise
            
            obs.append(np.array(ob_i))
        return np.array(obs)

    def render_trajectory(self):
        plt.figure(figsize=(8, 8))
        hist_a = np.array(self.history_agents) 
        colors =['#d62728', '#1f77b4', '#2ca02c'] 
        for i in range(self.agent_num):
            plt.plot(hist_a[:, i, 0], hist_a[:, i, 1], color=colors[i], alpha=0.5, linestyle='--')
            plt.scatter(hist_a[-1, i, 0], hist_a[-1, i, 1], marker='^', color=colors[i], s=150, label=f'UAV {i+1}')
        for j in range(self.user_num):
            plt.scatter(self.users_pos[j,0], self.users_pos[j,1], marker='*', color='#ff7f0e', s=250, edgecolor='black', label=f'UE {j+1}' if j==0 else "")
        plt.xlim(0, self.map_size); plt.ylim(0, self.map_size)
        plt.title(f"Naive Reward Trajectory - Episode {self.episode_counter}"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.save_img_dir, f"ep{self.episode_counter}_pid{os.getpid()}.png"), dpi=300)
        plt.close()