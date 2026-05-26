import gymnasium as gym
from QLearning import QLearning

from dataclasses import dataclass

@dataclass
class Cfg:
    policy_lr: float = 0.001 
    gamma: float = 0.99 
    train_eps = 5000

cfg = Cfg()


env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
#env = CliffWalkingWapper(env)
agent = QLearning(
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    learning_rate=cfg.policy_lr,
    gamma=cfg.gamma,)
rewards = []  
ma_rewards = [] # moving average reward
for i_ep in range(cfg.train_eps): # train_eps: ÑµÁ·µÄ×î´óepisodesÊý
    ep_reward = 0  # ¼ÇÂ¼Ã¿¸öepisodeµÄreward
    state = env.reset()  # ÖØÖÃ»·¾³, ÖØÐÂ¿ªÒ»¾Ö£¨¼´¿ªÊ¼ÐÂµÄÒ»¸öepisode£©
    while True:
        if isinstance(state, tuple):
            state, _ = state
        action = agent.choose_action(state)  # ¸ù¾ÝËã·¨Ñ¡ÔñÒ»¸ö¶¯×÷
        next_state, reward, done, info, _ = env.step(action)  # Óë»·¾³½øÐÐÒ»´Î¶¯×÷½»»¥
        agent.update(state, action, reward, next_state, done)  # Q-learningËã·¨¸üÐÂ
        state = next_state  # ´æ´¢ÉÏÒ»¸ö¹Û²ìÖµ
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
    else:
        ma_rewards.append(ep_reward)
    print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))
