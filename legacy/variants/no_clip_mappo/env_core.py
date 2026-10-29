import numpy as np
import math
import os
import matplotlib
# 【极其重要】设置无头模式，防止在终端多线程训练时弹窗导致程序崩溃！
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class EnvCore(object):
    """
    面向6G通感一体化(ISAC)的多无人机协同路径规划仿真环境
    内置自动绘制飞行轨迹图功能 (Academic Plotting)
    """
    def __init__(self):
        # ================= 1. 基础物理设定 =================
        self.agent_num = 3  # 无人机(UAV)数量
        self.user_num = 2   # 地面用户(UE)数量
        
        # 状态维度: 自身(x,y,vx,vy)[4维] + 其他UAV位置(x,y)*2 + 用户位置(x,y)*2
        self.obs_dim = 4 + (self.agent_num - 1) * 2 + self.user_num * 2
        self.action_dim = 2 # 连续动作空间: x和y方向的加速度

        # 6G ISAC 物理参数
        self.map_size = 100.0  # 场景大小 100x100 米
        self.max_speed = 5.0   # 无人机最大飞行速度 m/s
        self.power_tx = 10.0   # 无人机发射功率
        self.noise_power = 0.1 # 噪声功率
        self.safe_dist = 2.0   # 碰撞安全距离(米)
        self.max_steps = 200   # 每个回合最大步数
        
        # ================= 2. 画图黑科技设定 =================
        self.episode_counter = 0
        self.history_agents =[]
        self.history_users =[]
        # 创建一个专门存轨迹图的文件夹
        self.save_img_dir = "trajectory_results"
        if not os.path.exists(self.save_img_dir):
            os.makedirs(self.save_img_dir)

    def reset(self):
        """回合重置，随机初始化位置"""
        self.step_count = 0
        self.agents_pos = np.random.uniform(10, self.map_size - 10, (self.agent_num, 2))
        self.agents_v = np.zeros((self.agent_num, 2))
        self.users_pos = np.random.uniform(20, self.map_size - 20, (self.user_num, 2))
        
        # 清空并记录初始位置
        self.history_agents = [self.agents_pos.copy()]
        self.history_users =[self.users_pos.copy()]
        
        return self.get_obs()

    def step(self, actions):
        """核心交互：执行动作 -> 物理更新 -> 计算奖励 -> 画图"""
        self.step_count += 1
        actions = np.array(actions) 
        
        # === 1. 物理运动学更新 ===
        for i in range(self.agent_num):
            self.agents_v[i] += actions[i] * 1.0 
            speed = np.linalg.norm(self.agents_v[i])
            if speed > self.max_speed:
                self.agents_v[i] = (self.agents_v[i] / speed) * self.max_speed
            self.agents_pos[i] += self.agents_v[i] * 0.1 
            self.agents_pos[i] = np.clip(self.agents_pos[i], 0, self.map_size)
            
        # 记录每一步的轨迹
        self.history_agents.append(self.agents_pos.copy())
        self.history_users.append(self.users_pos.copy())

        # === 2. 多目标奖励函数计算 ===
        rewards =[]
        for i in range(self.agent_num):
            reward_i = 0.0
            
            # 通信奖励
            comm_rate = 0.0
            for j in range(self.user_num):
                dist = np.linalg.norm(self.agents_pos[i] - self.users_pos[j]) + 1e-3
                h_gain = 1.0 / (dist**2)
                sinr = (self.power_tx * h_gain) / self.noise_power
                comm_rate += math.log2(1 + sinr)
            reward_i += 0.5 * comm_rate 
                
            # 感知奖励
            sensing_score = 0.0
            for j in range(self.user_num):
                dist = np.linalg.norm(self.agents_pos[i] - self.users_pos[j]) + 1e-3
                sensing_snr = self.power_tx / (dist**4) 
                sensing_score += sensing_snr
            reward_i += 100.0 * sensing_score 

            # 碰撞惩罚
            for k in range(self.agent_num):
                if i != k:
                    dist_uav = np.linalg.norm(self.agents_pos[i] - self.agents_pos[k])
                    if dist_uav < self.safe_dist:
                        reward_i -= 15.0 

            rewards.append([reward_i]) 

        # === 3. 自动绘制轨迹图 (每 200 回合画一张) ===
        if self.step_count >= self.max_steps:
            self.episode_counter += 1
            if self.episode_counter % 200 == 0:
                self.render_trajectory()

        obs = self.get_obs()
        dones =[True if self.step_count >= self.max_steps else False for _ in range(self.agent_num)]
        infos =[{"comm_rate": 0, "sensing_score": 0} for _ in range(self.agent_num)]
        
        return obs, rewards, dones, infos

    def render_trajectory(self):
        """生成高清轨迹图"""
        plt.figure(figsize=(8, 8))
        hist_a = np.array(self.history_agents) # (steps, agents, 2)
        hist_u = np.array(self.history_users)  # (steps, users, 2)
        
        # 配色方案 (学术常用配色)
        colors =['#d62728', '#1f77b4', '#2ca02c'] 
        
        # 1. 画无人机飞行轨迹
        for i in range(self.agent_num):
            # 画线 (带点透明度)
            plt.plot(hist_a[:, i, 0], hist_a[:, i, 1], color=colors[i], alpha=0.6, linestyle='--')
            # 终点画个显眼的三角形代表UAV
            plt.scatter(hist_a[-1, i, 0], hist_a[-1, i, 1], marker='^', color=colors[i], s=200, label=f'UAV {i+1}')
            # 起点画个小圆点
            plt.scatter(hist_a[0, i, 0], hist_a[0, i, 1], marker='o', color=colors[i], s=50)

        # 2. 画地面用户 (假设静止)
        for j in range(self.user_num):
            plt.scatter(hist_u[0, j, 0], hist_u[0, j, 1], marker='*', color='#ff7f0e', s=300, edgecolor='black', label=f'User {j+1}' if j==0 else "Users")
            # 画一个浅色感知覆盖圈
            circle = plt.Circle((hist_u[0, j, 0], hist_u[0, j, 1]), 15.0, color='#ff7f0e', alpha=0.1)
            plt.gca().add_patch(circle)

        # 3. 设置图表格式
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.xlabel("X Coordinate (m)", fontsize=14)
        plt.ylabel("Y Coordinate (m)", fontsize=14)
        plt.title(f"ISAC UAV Swarm Trajectories (Episode {self.episode_counter})", fontsize=16)
        plt.legend(fontsize=12, loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # 4. 防止多线程写入冲突，用进程号区分文件名
        pid = os.getpid()
        save_path = os.path.join(self.save_img_dir, f"traj_ep{self.episode_counter}_pid{pid}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_obs(self):
        """生成局部观测向量"""
        obs =[]
        for i in range(self.agent_num):
            ob_i = list(self.agents_pos[i]) + list(self.agents_v[i])
            for k in range(self.agent_num):
                if i != k:
                    ob_i += list(self.agents_pos[k])
            for j in range(self.user_num):
                ob_i += list(self.users_pos[j])
            obs.append(np.array(ob_i))
        return np.array(obs)