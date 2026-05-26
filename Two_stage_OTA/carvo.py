import gymnasium as gym
from DQN import Agent

class Config():
    def __init__(self):
        self.max_episodes = 10000
        self.max_steps=1000
cfg = Config()

agent = Agent()
env = gym.make('CartPole-v1') 
n_states = env.observation_space.shape[0] 
n_actions = env.action_space.n 

rewards = [] 
moving_average_rewards = [] 
ep_steps = []
for i_episode in range(1, cfg.max_episodes+1):
    observation, _ = env.reset() 
    state = observation
    ep_reward = 0
    for i_step in range(1, cfg.max_steps+1):
        action = agent.select_action(state) 
        next_state, reward, done, info, _ = env.step(action)
        ep_reward += reward
        agent.memory_push(state, action, reward, next_state, done) 
        state = next_state 
        agent.update()
        if done:
            break
    #if i_episode % cfg.target_update == 0: 
    #    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print('Episode:', i_episode, ' Reward: %i' %
          int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
    ep_steps.append(i_step)
    rewards.append(ep_reward)
    if i_episode == 1:
        moving_average_rewards.append(ep_reward)
    else:
        moving_average_rewards.append(
            0.9*moving_average_rewards[-1]+0.1*ep_reward)
