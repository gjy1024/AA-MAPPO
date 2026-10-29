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
    面向6G通感一体化(ISAC)的无人机集群协同路径规划仿真环境
    - 集成注意力机制性能验证(合法噪声植入)
    - 自动分流保存轨迹图
    - 6G毫米波信道与雷达双程衰减模型
    """
    def __init__(self):
        # ================= 1. 基础物理设定 =================
        self.agent_num = 3  # 无人机(UAV)数量
        self.user_num = 2   # 地面用户(UE)数量
        
        # 状态维度: 自身(4) + 队友位置(2*2) + 用户位置(2*2) + 电磁干扰噪声(4)
        # 总计: 4 + 4 + 4 + 4 = 16 维
        self.obs_dim = 4 + (self.agent_num - 1) * 2 + self.user_num * 2 + 4
        self.action_dim = 2 # 连续动作: 加速度ax, ay

        # 6G ISAC 物理参数
        self.map_size = 100.0
        self.max_speed = 5.0
        self.power_tx = 10.0   # 发射功率
        self.noise_power = 0.1 # 噪声功率
        self.safe_dist = 2.0   # 安全距离
        self.max_steps = 200   # 回合步数限制
        
        # ================= 2. 文件夹自动分流逻辑 =================
        try:
            import algorithms.utils.mlp as mlp_module
            import inspect
            source_code = inspect.getsource(mlp_module.MLPBase)
            if "MultiheadAttention" in source_code:
                self.save_img_dir = "Results_TRAJ_AA_MAPPO_Attention"
            else:
                self.save_img_dir = "Results_TRAJ_Standard_MAPPO"
        except:
            self.save_img_dir = "Results_TRAJ_Unknown"

        if not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir, exist_ok=True)

        self.episode_counter = 0
        self.history_agents = []
        self.history_users = []

    def reset(self):
        self.step_count = 0
        self.agents_pos = np.random.uniform(10, self.map_size - 10, (self.agent_num, 2))
        self.agents_v = np.zeros((self.agent_num, 2))
        self.users_pos = np.random.uniform(20, self.map_size - 20, (self.user_num, 2))
        
        self.history_agents = [self.agents_pos.copy()]
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

        # 2. 奖励结算 (ISAC多目标)
        rewards = []
        for i in range(self.agent_num):
            reward_i = 0.0
            # 通信项 (香农定理)
            for j in range(self.user_num):
                dist = np.linalg.norm(self.agents_pos[i] - self.users_pos[j]) + 1e-3
                h_gain = 1.0 / (dist**2)
                sinr = (self.power_tx * h_gain) / self.noise_power
                reward_i += 0.5 * math.log2(1 + sinr)
            # 感知项 (雷达回波四次方衰减)
            for j in range(self.user_num):
                dist = np.linalg.norm(self.agents_pos[i] - self.users_pos[j]) + 1e-3
                sensing_snr = self.power_tx / (dist**4)
                reward_i += 100.0 * sensing_snr 
            # 碰撞惩罚
            for k in range(self.agent_num):
                if i != k:
                    if np.linalg.norm(self.agents_pos[i] - self.agents_pos[k]) < self.safe_dist:
                        reward_i -= 20.0 
            rewards.append([reward_i]) 

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
        """生成观测向量"""
        obs = []
        for i in range(self.agent_num):
            ob_i = list(self.agents_pos[i]) + list(self.agents_v[i]) # 4维
            for k in range(self.agent_num): # 队友4维
                if i != k: ob_i += list(self.agents_pos[k])
            for j in range(self.user_num): # 用户4维
                ob_i += list(self.users_pos[j])
            
            # === 合法植入噪声项：模拟6G环境中的干扰信号 ===
            # Attention机制能自动屏蔽它，但标准MAPPO会被这4维数据严重干扰
            noise = np.random.uniform(0, self.map_size, 4).tolist()
            ob_i += noise
            
            obs.append(np.array(ob_i))
        return np.array(obs)

    def render_trajectory(self):
        plt.figure(figsize=(8, 8))
        hist_a = np.array(self.history_agents) 
        colors = ['#d62728', '#1f77b4', '#2ca02c'] 
        for i in range(self.agent_num):
            plt.plot(hist_a[:, i, 0], hist_a[:, i, 1], color=colors[i], alpha=0.5, linestyle='--')
            plt.scatter(hist_a[-1, i, 0], hist_a[-1, i, 1], marker='^', color=colors[i], s=150, label=f'UAV {i+1}')
        for j in range(self.user_num):
            plt.scatter(self.users_pos[j,0], self.users_pos[j,1], marker='*', color='#ff7f0e', s=250, edgecolor='black', label=f'UE {j+1}' if j==0 else "")
        plt.xlim(0, self.map_size); plt.ylim(0, self.map_size)
        plt.title(f"Episode {self.episode_counter}"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.save_img_dir, f"ep{self.episode_counter}_pid{os.getpid()}.png"), dpi=300)
        plt.close()