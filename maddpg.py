import os
import torch as T
import torch.nn.functional as F
from agent import Agent
# from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.00001, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        # 拼接基础场景路径，并保存为实例变量【关键修改1】
        self.base_chkpt_dir = chkpt_dir + scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(self.base_chkpt_dir, 'logs'))
        # 初始化智能体，仍用基础路径（不影响训练，仅保存时动态改路径）
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=self.base_chkpt_dir))

    # 恢复episode参数，适配main中传轮次的调用【关键修改2】
    def save_checkpoint(self, episode):
        print('... saving checkpoint ...')
        episode_str = str(episode)
        # 动态构建「场景名_轮次」的新保存文件夹，路径和原有风格一致
        new_save_dir = f"{self.base_chkpt_dir}_{episode_str}"
        # 确保新文件夹存在，不存在则创建
        os.makedirs(new_save_dir, exist_ok=True)

        for agent in self.agents:
            # 提取原有所有模型的文件名（保持文件名完全不变）
            actor_file = os.path.basename(agent.actor.chkpt_file)
            critic_file = os.path.basename(agent.critic.chkpt_file)
            target_actor_file = os.path.basename(agent.target_actor.chkpt_file)
            target_critic_file = os.path.basename(agent.target_critic.chkpt_file)

            # 关键3：统一更新【所有模型】的保存路径到新轮次文件夹
            agent.actor.chkpt_file = os.path.join(new_save_dir, actor_file)
            agent.critic.chkpt_file = os.path.join(new_save_dir, critic_file)
            agent.target_actor.chkpt_file = os.path.join(new_save_dir, target_actor_file)
            agent.target_critic.chkpt_file = os.path.join(new_save_dir, target_critic_file)

            # 原有保存逻辑完全不变，调用agent.save_models()会保存所有4类模型到新路径
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx],time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        if not memory.ready():
            return
        # 新增：接收采样索引、重要性采样权重
        actor_states, states, actions, rewards, actor_new_states, states_, dones, batch_indices, is_weights = memory.sample_buffer()
        device = self.agents[0].actor.device
        # 转换为tensor（IS权重也转tensor）
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        is_weights = T.tensor(is_weights, dtype=T.float).to(device)  # [256,]

        all_agents_new_actions = []
        old_agents_actions = []
    
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        # 存储所有agent的TD-error，用于更新buffer
        all_td_errors = []
        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()
            # 新增：计算TD-error并存储
            td_error = target - critic_value
            all_td_errors.append(td_error)
            # 新增：IS权重加权MSE损失（PER核心）
            critic_loss = F.mse_loss(target * is_weights, critic_value * is_weights)

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten() * is_weights)  # IS权重加权策略损失
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

        # 新增：更新buffer中的TD-error（取所有agent的TD-error均值）
        avg_td_error = T.mean(T.stack(all_td_errors), dim=0)
        memory.update_td_error(batch_indices, avg_td_error)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        for agent in self.agents:    
            agent.update_network_parameters()
