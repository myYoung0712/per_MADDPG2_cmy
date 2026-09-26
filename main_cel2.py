import numpy as np
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import logging  # 新增：导入logging模块

warnings.filterwarnings('ignore')


# 新增：配置日志系统（仅写入文件，不干扰控制台print）
def setup_logger():
    """配置日志，仅将内容写入log文件，控制台输出保留原print"""
    # 定义固定的日志文件路径
    log_file = 'maddpg_uav_training.log'

    # 创建logger实例
    logger = logging.getLogger('UAV_Training')
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    logger.handlers.clear()  # 清除重复的处理器，避免重复写入

    # 定义日志格式：时间戳 - 日志级别 - 消息内容（便于后续分析）
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 仅添加文件处理器：日志只写入文件，控制台输出由原print负责
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 初始化logger
logger = setup_logger()


def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state


def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency

    image.save(filename)


if __name__ == '__main__':

    env = UAVEnv()
    # print(env.info)
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 2
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=0.00001, beta=0.02, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=256)

    PRINT_INTERVAL = 1
    SAVE_INTERVAL = 100
    N_GAMES = 30000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False
    best_score = -30

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False] * n_agents
        episode_step = 0
        while not any(dones):
            if evaluate:
                # env.render()
                env_render = env.render()
                if episode_step % 10 == 0:
                    # Save the image every 10 episode steps
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if it doesn't exist
                    save_image(env_render, filename)
                # time.sleep(0.01)
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
            obs_, rewards, dones, success = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents

            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory, total_steps)

            obs = obs_
            # print('rewards[1]:', rewards[1])
            score += sum(rewards[0:3])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1
            # print('score:', score)

        print('score:', score)
        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i % SAVE_INTERVAL == 0 and i > 0 and avg_score > best_score:
                # 保留print（控制台输出） + 新增logger.info（写入文件）
                print('New best score', avg_score, '>', best_score, 'saving models...')
                logger.info(f'New best score {avg_score} > {best_score} saving models...')
                maddpg_agents.save_checkpoint(i)
                best_score = avg_score
            if i % 1000 == 0 and i > 10000:
                print('1w轮之后每1000轮保存一次：', 'saving models...')
                maddpg_agents.save_checkpoint(i)

        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),
                  '; average target score {:.1f}'.format(avg_target_score),'status:', success)
            logger.info(f'episode {i} average score {avg_score:.1f} ; average target score {avg_target_score:.1f} ; catch status:{success}')

    # save data
    file_name = 'score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)