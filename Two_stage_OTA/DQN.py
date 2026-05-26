import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random

episodes = 10000
trajectory = 100
gamma = 0.99
epsilon = 0.1
tau = 0.005

env = gym.make('CartPole-v1') 
n_states = env.observation_space.shape[0] # »ñÈ¡×ÜµÄ×´Ì¬Êý
n_actions = env.action_space.n # »ñÈ¡×ÜµÄ¶¯×÷Êý
class D1_net(nn.Module):
    def __init__(self):
        super(D1_net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,n_actions)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=2e-4)
    def forward(self,x):
        x = self.net(x)
        return x

class D2_net(nn.Module):
    def __init__(self):
        super(D2_net,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,n_actions)
        )
    def forward(self,x):
        x = self.net(x)
        return x


class Replay_Buffer():
    def __init__(self):
        self.replay_buffer = []
        self.length = 0
    
    def add(self,state, action, reward, next_state, done):
        self.replay_buffer.append({
            'state':state,
            'action':action,
            'reward':reward,
            'next_state':next_state,
            'done':done
        })
        self.length += 1

    def sample(self,batch_size):
        transisions = random.sample(self.replay_buffer, batch_size)
        s = torch.FloatTensor(np.array([t['state'] for t in transisions])) 
        a = torch.LongTensor([t['action'] for t in transisions])
        r = torch.FloatTensor([t['reward'] for t in transisions])
        s_ = torch.FloatTensor(np.array([t['next_state'] for t in transisions]))
        done = torch.BoolTensor(([t['done'] for t in transisions]))
        return s, a, r, s_, done

    def size(self):
        return self.length

class Agent(object):
    def __init__(self):
        self.d1 = D1_net()
        self.d2 = D2_net()
        self.replay_buffer = Replay_Buffer()
        self.epsilon = 0.05
        self.step = 0

    def select_action(self, state):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_value = self.d1(torch.tensor(state,dtype=torch.float))
                action = np.argmax(q_value.numpy())
        return action
    
    def memory_push(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if self.replay_buffer.length < 10:
            return
        self.step += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(10)
        st_d1 = self.d1(state)
        q_now = st_d1.gather(1, action.unsqueeze(1))
        with torch.no_grad():
            #Q(st,at)
            #state_tensor = torch.FloatTensor(state).unsqueeze(0)
            #next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            mask = ~done
            #argmaxa Q(st+1,a)
            st_1_d1 = self.d1(next_state)
            #st_1_d1 = self.d2(next_state)# test DQN
            argmax_a = st_1_d1.argmax(dim=1, keepdim=True)
            st_1_d2 = self.d2(next_state)
            q_st_1_max_a = st_1_d2.gather(1, argmax_a)
            #q_target = rt + gamma*Q'(st+1,argman_a)
            q_target = reward.unsqueeze(1) + gamma*q_st_1_max_a*mask.unsqueeze(1).float()
            print(q_target)
            #TD error
            #TD_error = q_st_at - q_target
            #TD_error = TD_error.detach().numpy()

        loss = F.mse_loss(q_now,q_target)

        self.d1.optim.zero_grad()
        loss.backward()
        self.d1.optim.step()
        for target_param, param in zip(self.d2.parameters(), self.d1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


if __name__ == '__main__':
    agent = Agent()
    for i_episode in range(episodes):
        state, _ = env.reset()  # GymÐÂ°æ±¾reset·µ»ØµÄÊÇÒ»¸öÔª×é (obs, info)
        ep_reward = 0

        for i_step in range(trajectory):
            # Ñ¡Ôñ¶¯×÷
            action = agent.select_action(state)

            # Ö´ÐÐ¶¯×÷
            next_state, reward, done, truncated, info = env.step(action)

            # ¼ÇÂ¼¾­Ñé»Ø·Å
            agent.memory_push(state, action, reward, next_state, done)

            # ¸üÐÂÍøÂç
            agent.update()

            # ×´Ì¬×ªÒÆ
            state = next_state
            ep_reward += reward

            # Èç¹û»ØºÏÌáÇ°½áÊø
            if done or truncated:
                break

        # ´òÓ¡ÑµÁ·ÐÅÏ¢
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode}, Reward: {ep_reward}, Buffer Size: {agent.replay_buffer.size()}")

    env.close()

