import numpy as np
import math

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size  # 最终采样数256
        self.n_actions = n_actions
        self.Npre_batch = 1000        # 预采样数1000（论文参数）
        # PER超参数
        self.alpha = 0.6              # 优先级系数
        self.beta = 0.4               # 重要性采样系数
        # REL超参数（论文公式29）
        self.sigma1 = 100
        self.sigma2 = 2
        self.sigma3 = 5

        # 原有存储数组
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        # 新增：TD-error存储（PER用）、相关性指标fr存储（REL用）
        self.td_error_memory = np.ones((self.mem_size,))  # 初始TD-error为1
        self.fr_memory = np.zeros((self.mem_size,))        # 存储每个经验的fr值

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    # 新增：计算相关性指标fr（论文公式29）
    def calculate_fr(self, actor_states):
        # actor_states：[n_agents, obs_dims]，提取UAV和目标的位置（从观测中解析，与sim_env一致）
        p_list = []
        for i in range(self.n_agents):
            # 观测前4维：x/length, y/length, vx/vmax, vy/vmax（归一化）
            x = actor_states[i][0] * 2  # 还原到实际场景尺寸（length=2）
            y = actor_states[i][1] * 2
            p_list.append(np.array([x, y]))
        p0, p1, p2, pe = p_list  # p0-p2为围捕UAV，pe为目标

        # 计算论文中SumS=S1+S2+S3，S4为UAV三角形面积
        def cal_triangle_S(p1, p2, p3):
            S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
            return 0.0 if math.isclose(S, 0.0, abs_tol=1e-9) else S
        S1 = cal_triangle_S(p0, p1, pe)
        S2 = cal_triangle_S(p1, p2, pe)
        S3 = cal_triangle_S(p2, p0, pe)
        SumS = S1 + S2 + S3
        S4 = cal_triangle_S(p0, p1, p2)

        # 计算包围网重心O，及UAV到O的距离和Sum_dOU
        O = np.mean([p0, p1, p2], axis=0)  # 网重心
        d0 = np.linalg.norm(p0 - O)
        d1 = np.linalg.norm(p1 - O)
        d2 = np.linalg.norm(p2 - O)
        Sum_dOU = d0 + d1 + d2

        # 论文公式29：fr = σ1*(SumS-S4) + σ2*SumS + σ3*Sum_dOU
        fr = self.sigma1 * (SumS - S4) + self.sigma2 * SumS + self.sigma3 * Sum_dOU
        return fr

    def store_transition(self, raw_obs, state, action, reward, 
                               raw_obs_, state_, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        #if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
        
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        # 新增：计算并存储当前经验的fr值
        self.fr_memory[index] = self.calculate_fr(raw_obs)
        self.mem_cntr += 1

    # 新增：更新TD-error（从maddpg的learn方法中调用）
    def update_td_error(self, indices, td_errors):
        self.td_error_memory[indices] = np.abs(td_errors.cpu().detach().numpy()) + 1e-6  # 避免0优先级

    # 核心修改：双层采样（PER+REL）替换原随机采样
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        # 第一步：PER优先采样1000个样本（基于TD-error的优先级）
        # 计算优先级：p = |td_error|^α + 1e-6
        priorities = self.td_error_memory[:max_mem] ** self.alpha
        prob = priorities / np.sum(priorities)  # 优先级概率分布
        # 采样1000个，允许重复但论文中未禁止，与原代码replace=False保持一致（若内存不足则采全部）
        pre_batch_size = min(self.Npre_batch, max_mem)
        pre_batch_indices = np.random.choice(max_mem, pre_batch_size, replace=False, p=prob)
        # 计算重要性采样权重
        is_weights = (max_mem * prob[pre_batch_indices]) ** (-self.beta)
        is_weights = is_weights / np.max(is_weights)  # 归一化

        # 第二步：REL相关性采样256个样本（从1000个中选fr最接近最新经验的）
        latest_fr = self.fr_memory[min(self.mem_cntr-1, max_mem-1)]  # 最新经验的fr值
        pre_fr = self.fr_memory[pre_batch_indices]
        # 计算fr差值的绝对值，选最小的256个
        fr_diff = np.abs(pre_fr - latest_fr)
        final_idx_in_pre = np.argsort(fr_diff)[:self.batch_size]
        final_batch_indices = pre_batch_indices[final_idx_in_pre]
        final_is_weights = is_weights[final_idx_in_pre]  # 对应最终样本的权重

        # 按最终采样索引提取数据，与原返回格式一致（新增is_weights用于损失加权）
        states = self.state_memory[final_batch_indices]
        rewards = self.reward_memory[final_batch_indices]
        states_ = self.new_state_memory[final_batch_indices]
        terminal = self.terminal_memory[final_batch_indices]
        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][final_batch_indices])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][final_batch_indices])
            actions.append(self.actor_action_memory[agent_idx][final_batch_indices])
        # 返回：原有数据 + 采样索引 + 重要性权重（用于maddpg的learn方法）
        return actor_states, states, actions, rewards, actor_new_states, states_, terminal, final_batch_indices, final_is_weights

    def ready(self):
        # 调整：确保预采样数足够（至少1000个经验才开始学习）
        if self.mem_cntr >= self.Npre_batch:
            return True
