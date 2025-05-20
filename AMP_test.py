import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.transforms import NormalizeFeatures
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
import numpy as np
from datetime import datetime

from deap import tools,creator,base
import os
import selection

import csv

from gymnasium.envs.registration import register

date = datetime.today().strftime('%Y-%m-%d')
file_path = 'reward.csv'
PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

env_id = 'sky130AMP_NMCF-v0'
env_dict = gym.envs.registration.registry.copy()

for env in env_dict:
    if env_id in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry[env]

print("Register the environment")
register(
        id = env_id,
        entry_point = 'AMP_NMCF:AMPNMCFEnv',
        max_episode_steps = None,
        )

#env = gym.make(env_id)

EP_MAX = 10000 #max episode
STEPS = 64  #one episode contains STEPS steps
LR_v = 1e-4    #value network learning rate
LR_pi = 3e-4   #policy network learning rate
LR_r = 5e-5    #reward network learning rate
K_epoch = 7    #one sample reused times
GAMMA = 0.99   #discount rate
LAMBDA = 0.95  #generalized advantage estimation
CLIP = 0.2     #clip
observation_param = 12
design_param = 24
state_space = 12
population_size = 16#20

#ANN
class Pi_net(nn.Module):
    def __init__(self):
        super(Pi_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(design_param, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, 3) for _ in range(design_param)
        ])
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_pi)
 
    def forward(self, x):
        x = self.net(x)
        return torch.stack([head(x) for head in self.heads], dim=1)

class V_net(nn.Module):
    def __init__(self):
        super(V_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(design_param, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)
 
    def forward(self, x):
        x = self.net(x)
        return x
    
def init_weights(m):
    if isinstance(m, GCNConv):
        nn.init.xavier_uniform_(m.lin.weight, gain=5.0)
#GCN
class Pi_GCNnet(nn.Module):
    def __init__(self):
        super(Pi_GCNnet, self).__init__()
        self.conv1 = GCNConv(state_space, 64)
        self.conv2 = GCNConv(64, 64)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        #self.conv3 = GCNConv(64, 32)
        #self.conv4 = GCNConv(32, 32)
        self.heads = nn.ModuleList([
            nn.Linear(64, 3) for _ in range(design_param)
        ])
        self.pool = global_mean_pool

        self.optim = torch.optim.Adam(self.parameters(), lr =LR_pi)

        #nn.init.xavier_uniform_(self.conv1.weight, gain=5.0)
        #nn.init.xavier_uniform_(self.conv2.weight, gain=5.0)
        self.apply(init_weights)
 
    def forward(self, x, edge_index):
        x = F.tanh(self.norm1(self.conv1(x, edge_index)))
        x = F.tanh(self.norm2(self.conv2(x, edge_index)))
        #x = self.conv2(x, edge_index)
        #x = self.conv3(x, edge_index)
        #x = self.conv4(x, edge_index)
        x = self.pool(x, batch=None)
        logits = torch.stack([head(x) for head in self.heads], dim=1)

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        return logits
    


class V_GCNnet(nn.Module):
    def __init__(self):
        super(V_GCNnet, self).__init__()
        self.conv1 = GCNConv(state_space, 64)
        self.conv2 = GCNConv(64, 64)
        #self.conv3 = GCNConv(64, 32)
        #self.conv4 = GCNConv(32, 32)
        
        self.pool = global_mean_pool
        
        self.q_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32,1)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)

    def forward(self, x, edge_index):
        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))
        #x = self.conv3(x, edge_index)
        #x = self.conv4(x, edge_index)
        
        x = self.pool(x, batch=None)
        
        return self.q_head(x)
    
class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 11)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=LR_r)

    def forward(self, x):
        x = self.net(x)
        x = torch.tanh(x)
        return (x - 1)/2

class Agent(object):
    def __init__(self,reward_model=None):
        self.v = V_GCNnet()        
        self.pi = Pi_GCNnet()
        self.old_pi = Pi_GCNnet()
        self.old_v = V_GCNnet()
        self.reward_model = reward_model
                                
        #self.load()
        self.data = []
        self.step = 0
        
        self.edge_index = torch.tensor([
           [1,0],  [2,0],  [3,0], [4,0],  [7,0],  [24,0],  [25,0], 
           [2,1],  [3,1],  [4,1], [7,1],  [25,1],  [12,1],  [24,1],  [17,1],  [18,1],  [19,1], [20,1], 
           [3,2],  [4,2],  [7,2],  [24,2],  [13,2],  [25,2],  
           [4,3],  [7,3],  [24,3],  [25,3],  [12,3],  [13,3],  [14,3],  [15,3],  [16,3], 
           [7,4],  [24,4],  [8,4],  [9,4], [25,4],
           [6,5],  [15,5],  [25,5], 
           [15,6],  [16,6],  [10,6],  [11,6], [25,6], [27,6], 
           [24,7],  [25,7],  [22,7],  [23,7],  [28,7],
           [9,8],  [15,8],  [19,8], 
           [16,9],  [20,9], 
           [11,10],  [16,10],  [21,10],  [22,10], [25,10], [27,10],
           [25,11],  [16,11],  [23,11],  [27,11], [28,11], 
           [13,12],  [14,12],  [15,12],  [16,12], [17,12],  [18,12],  [19,12], [20,12], 
           [14,13],  [15,13],  [16,13],  [18,13],
           [15,14],  [16,14],  [26,14],  
           [16,15],  [19,15], 
           [20,16],  [27,16], 
           [18,17],  [19,17],  [20,17],  [26,17], 
           [19,18],  [20,18],  [26,18],
           [20,19],  [26,19], 
           [26,20], 
           [22,21],  [26,21], 
           [23,22], [26,22],  [28,22],
           [26,23], [27,23],  [28,23], 
           [26,24], 
            ], dtype=torch.long).t().contiguous()
 
    def choose_action(self, s):
        with torch.no_grad():   
            logits = self.old_pi(s, self.edge_index)
            logits = logits.view(-1, design_param, 3)
            probs = F.softmax(logits, dim=-1)
            #print("logits:",logits)
            dist = Independent(Categorical(probs),1)
            entropy = dist.entropy().mean()
            #print("Probs and Policy Entropy:",probs , entropy.item())
            with open("entropy.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([entropy.item()])
            action = dist.sample()
            old_log_probs = dist.log_prob(action).detach()
        return action, old_log_probs
 
    def push_data(self, transitions):
        self.data.append(transitions)
 
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done, l_old_prob, l_parameter, l_info_score  = [], [], [], [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done, old_log_prob, parameter, info_score = item
            
            l_s.append(torch.tensor(s, dtype=torch.float))
            l_a.append(torch.tensor(a, dtype=torch.float))
            l_r.append(torch.tensor(r, dtype=torch.float).unsqueeze(0))
            l_s_.append(torch.tensor(s_, dtype=torch.float))
            l_done.append(torch.tensor(done, dtype=torch.float).unsqueeze(0))
            l_old_prob.append(old_log_prob)
            l_parameter.append(torch.tensor(parameter, dtype=torch.float))
            l_info_score.append(torch.tensor(info_score, dtype=torch.float))
        s = torch.stack(l_s, dim=0)
        a = torch.stack(l_a, dim=0)
        r = torch.stack(l_r, dim=0).squeeze(0)
        s_ = torch.stack(l_s_, dim=0)
        done = torch.stack(l_done, dim=0).squeeze(0)
        old_log_prob = torch.stack(l_old_prob, dim=0)
        parameter = torch.stack(l_parameter, dim=0)
        info_score = torch.stack(l_info_score, dim=0)
        self.data = []

        return s, a, r, s_, done, old_log_prob, parameter, info_score
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
            advantages.insert(0, last_advantage)
        return torch.tensor(advantages).float()
 
    def updata(self):
        self.step += 1
        s, a, r, s_, done, old_log_probs, parameter, info_score = self.sample()

        #training reward model
        expected_score = self.reward_model(parameter).squeeze()
        loss = F.mse_loss(expected_score, info_score)
        self.reward_model.optim.zero_grad()
        loss.backward()
        self.reward_model.optim.step()

        with torch.no_grad():
            old_values = self.old_v(s, self.edge_index).squeeze()
            old_next_values = self.old_v(s_, self.edge_index).squeeze()
        advantages = self.compute_advantages(r, old_values, old_next_values, done)
        print("old_values old_next_values advantage:",old_values,old_next_values, advantages)
        returns = advantages + old_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(K_epoch):
            current_logits = self.pi(s, self.edge_index)
            current_logits = current_logits.view(-1, design_param, 3)
            probs = F.softmax(current_logits, dim=-1)
            current_dist = Independent(Categorical(probs),1)
            a = a.squeeze(1)
            log_probs = current_dist.log_prob(a)
            print("log_probs:",current_dist, log_probs,old_log_probs.squeeze(),a)

            ratios = torch.exp(log_probs - old_log_probs.squeeze())


            entropy = current_dist.entropy().mean()
            entropy_weight = 0.01 * (0.995 ** self.step)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-CLIP, 1+CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - entropy * entropy_weight
        
            current_values = self.v(s, self.edge_index).squeeze()
            value_loss = F.mse_loss(current_values, returns)
            print("Value Loss:", value_loss.item(),current_values,returns)
            print("Policy Loss:", policy_loss.item(), ratios, advantages)

            self.pi.optim.zero_grad()
            self.v.optim.zero_grad()

            (policy_loss + value_loss).backward()

            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), 0.5)

            self.pi.optim.step()
            self.v.optim.step()
        self.old_pi.load_state_dict(self.pi.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())
 
    def save(self):
        torch.save(self.pi.state_dict(), 'pi.pth')
        torch.save(self.v.state_dict(), 'v.pth')
        print('...save model...')
 
    def load(self):
        try:
            self.pi.load_state_dict(torch.load('pi.pth'))
            self.v.load_state_dict(torch.load('v.pth'))
            print('...load...')
        except:
            pass


"""init_parameters = torch.Tensor([5 ,5, 15,          
                  4, 1, 11,
                  3, 3, 12,
                  2, 3, 336,
                  1, 0.5, 14,
                  6, 0.5, 4,
                  0.5, 4, 44,
                  6e-06, 
                  10,
                  10])"""
def main():
    reward_model = RewardModel(design_param)
    env = gym.make(env_id, reward_model=reward_model)#options={"reward_model": reward_model})  
    agent = Agent(reward_model)
    #agent.load()
    max_rewards = -1000000
    reset_sign = False
    creator.create("FitnessMin", base.Fitness, weights=((-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    population = [creator.Individual(selection.generate_individual()) for _ in range(population_size)]
    for j in range(EP_MAX):
        reset_index = j % population_size
        #selected_individuals = selection.nsga_selection(population, iteration_times, population_size/2)
        #population = selection.crossover_variation(selected_individuals)
        s, _ = env.reset(episodes=reset_index)
        rewards = 0
        for i in range(STEPS):
            print("steps:", i)
            a, old_log_prob = agent.choose_action(torch.tensor(s, dtype=torch.float))
            s_, r, done, parameters, info = env.step(a.squeeze())
            #print("check s a:",s,a)
            info_score = [info['Power'], info['dcgain'], info['GBW'], info['phase_margin (deg)'], info['TC'], info['vos'], info['cmrrdc'], info['PSRP'], info['PSRN'], info['sr'], info['setting_time']]
            agent.push_data((s, a, r, s_, done, old_log_prob, parameters, info_score))
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([r])
            rewards += r
            #if done:
            #    break
            s = s_
        agent.updata()
        if max_rewards < rewards:
            max_rewards = rewards
            agent.save()
 
if __name__ == '__main__':
    main()
