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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

env_id = 'sky130AMP_NMCF-v0'
env_dict = gym.envs.registration.registry.copy()

for env in env_dict:
    if env_id in env:
        #print("Remove {} from registry".format(env))
        del gym.envs.registration.registry[env]

#print("Register the environment")
register(
        id = env_id,
        entry_point = 'AMP_NMCF:AMPNMCFEnv',
        max_episode_steps = None,
        )

#env = gym.make(env_id)

EP_MAX = 10000 #max episode
STEPS = 16  #one episode contains STEPS steps
LR_v = 1e-4    #value network learning rate
LR_pi = 3e-4   #policy network learning rate
LR_r = 5e-5    #reward network learning rate
K_epoch = 4    #one sample reused times 8-4-2
GAMMA = 0.99   #discount rate
LAMBDA = 0.95  #generalized advantage estimation
CLIP = 0.2     #clip
design_param = 14
state_space = 12

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
        # x ½øÀ´µÄÐÎ×´¿ÉÄÜÊÇ (Batch, 22, 12)
        B, N, F = x.shape
        
        # 1. Õ¹Æ½ Node Î¬¶È: (Batch * 22, 12)
        x_flat = x.view(-1, F)
        
        # 2. ¶¯Ì¬¹¹½¨ Batched edge_index (·Ç³£¹Ø¼ü£¡)
        # Ô­Ê¼ edge_index ÐÎ×´ (2, E)£¬ÐèÒª¸øÃ¿¸öÍ¼µÄ½ÚµãË÷Òý¼ÓÉÏÆ«ÒÆÁ¿
        E = edge_index.shape[1]
        offset = (torch.arange(B, device=x.device) * N).view(B, 1)
        # Æ«ÒÆºóµÄ batched_edge_index ÐÎ×´ (2, B * E)
        batched_edge_index = (edge_index.unsqueeze(0) + offset.unsqueeze(1)).transpose(0, 1).reshape(2, -1)
        
        # 3. ¹ýÍ¼¾í»ý²ã (×¢Òâ F.tanh ÒÑÆúÓÃ£¬¸ÄÓÃ torch.tanh)
        x_out = torch.tanh(self.norm1(self.conv1(x_flat, batched_edge_index)))
        x_out = torch.tanh(self.norm2(self.conv2(x_out, batched_edge_index)))
        
        # 4. È«¾Ö³Ø»¯ (±ØÐë´«Èë batch Ë÷ÒýÏòÁ¿¸æËß PyG ÄÄÐ©½ÚµãÊôÓÚÄÄ¸öÍ¼)
        # batch_idx ¿´ÆðÀ´Ïñ: [0,0.., 1,1.., ..., B-1,B-1..]
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        x_pooled = self.pool(x_out, batch=batch_idx) # Êä³öÐÎ×´»Ö¸´Îª (Batch, 64)
        
        # 5. Êä³ö Action Logits
        logits = torch.stack([head(x_pooled) for head in self.heads], dim=1) # (Batch, 14, 3)
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
        B, N, F_in = x.shape
        x_flat = x.reshape(-1, F_in)
        
        E = edge_index.shape[1]
        offset = (torch.arange(B, device=x.device) * N).view(B, 1)
        batched_edge_index = (edge_index.unsqueeze(0) + offset.unsqueeze(1)).transpose(0, 1).reshape(2, -1)
        
        x_out = torch.tanh(self.conv1(x_flat, batched_edge_index))
        x_out = torch.tanh(self.conv2(x_out, batched_edge_index))
        
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        x_pooled = self.pool(x_out, batch=batch_idx)
        
        return self.q_head(x_pooled) # Êä³ö (Batch, 1)
    
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
            nn.Linear(32, 12)
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
            # --- Signal Path ---
            # Diff Pair (0,1) -> Tail (4)
            [4, 0], [0, 4], [4, 1], [1, 4], [0, 1], [1, 0],
            # Active Load (0->2, 1->3, 2-3 mirror)
            [0, 2], [2, 0], [2, 3], [3, 2], [1, 3], [3, 1],
            # Stage 1 to 2 (3->5, 3->17)
            [3, 5], [5, 3], [3, 17], [17, 3], 
            # Output Node (5, 6, 19)
            [5, 6], [6, 5], [5, 19], [19, 5], [6, 19], [19, 6],
            # Compensation (17-19)
            [17, 19], [19, 17],

            # --- Bias & Mirrors ---
            # Ib Loop (18 connects to Gates: 7, 4, 6, 8, 14)
            [18, 7], [7, 18], [18, 4], [4, 18], [18, 6], [6, 18], 
            [18, 8], [8, 18], [18, 14], [14, 18],

            # NMOS Bias Gen
            [8, 9], [9, 8],     # M_feeder -> M14
            [9, 10], [10, 9],   # M14 -> M16
            
            # M13/M15/M8 Connections
            [9, 12], [12, 9],   # Mirror Gate M14->M13
            [10, 13], [13, 10], # Mirror Gate M16->M15
            [12, 13], [13, 12], # Stack M13-M15
            [11, 12], [12, 11], # M8 Drain -> M13 Drain

            # Rc Control
            [14, 15], [15, 14], # M10 -> M11
            [15, 16], [16, 15], # M11 -> M12
            [15, 17], [17, 15], # M11 Gate -> M9 Gate

            # --- Power Connections (To 20 and 21) ---
            # PMOS Sources -> VDD (20)
            [7, 20], [4, 20], [6, 20], [8, 20], [11, 20], [14, 20], [0, 20], [1, 20],
            # NMOS Sources -> GND (21)
            [2, 21], [3, 21], [5, 21], [10, 21], [13, 21], [16, 21]
        ], dtype=torch.long).t().contiguous()
 
    def choose_action(self, s):
        with torch.no_grad():   
            logits = self.old_pi(s, self.edge_index)
            logits = logits.view(-1, design_param, 3)
            probs = F.softmax(logits, dim=-1)
            ##print("logits:",logits)
            dist = Independent(Categorical(probs),1)
            entropy = dist.entropy().mean()
            ##print("Probs and Policy Entropy:",probs , entropy.item())
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
            l_r.append(torch.tensor(r, dtype=torch.float))
            l_s_.append(torch.tensor(s_, dtype=torch.float))
            l_done.append(torch.tensor(done, dtype=torch.float))
            l_old_prob.append(old_log_prob)
            l_parameter.append(torch.tensor(parameter, dtype=torch.float))
            l_info_score.append(torch.tensor(info_score, dtype=torch.float))
        s = torch.stack(l_s, dim=0)
        a = torch.stack(l_a, dim=0)
        r = torch.stack(l_r, dim=0)
        s_ = torch.stack(l_s_, dim=0)
        done = torch.stack(l_done, dim=0)
        old_log_prob = torch.stack(l_old_prob, dim=0)
        parameter = torch.stack(l_parameter, dim=0)
        info_score = torch.stack(l_info_score, dim=0)
        self.data = []

        return s, a, r, s_, done, old_log_prob, parameter, info_score
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            ##print(f"[DEBUG] t={t}, reward={rewards[t]}, value={values[t]}, next_value={next_values[t]}, done={dones[t]}")
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
            advantages.insert(0, last_advantage)
        return torch.tensor(advantages).float()
 
    def update(self):
        self.step += 1
        s, a, r, s_, done, old_log_probs, parameter, info_score = self.sample()

        #training reward model
        if self.reward_model:
            expected_score = self.reward_model(parameter).squeeze()
            loss = F.mse_loss(expected_score, info_score)
            self.reward_model.optim.zero_grad()
            loss.backward()
            self.reward_model.optim.step()
        advantages_all = []
        returns_all = []

        # »ñÈ¡ÖÖÈº´óÐ¡
        pop_size = r.shape[1] 

        # °´¸öÌå±éÀú£¨ÌáÈ¡µ¥¸ö¸öÌåÍêÕûµÄ STEPS Ê±¼äÐòÁÐ£©
        for pop_index in range(pop_size):
            traj_r = r[:, pop_index]
            traj_done = done[:, pop_index]
            traj_s = s[:, pop_index, :]
            traj_s_ = s_[:, pop_index, :]

            with torch.no_grad():
                traj_old_values = self.old_v(traj_s, self.edge_index).squeeze().cpu().numpy()
                traj_old_next_values = self.old_v(traj_s_, self.edge_index).squeeze().cpu().numpy()

            traj_advantages = self.compute_advantages(
                rewards=traj_r,
                values=traj_old_values,
                next_values=traj_old_next_values,
                dones=traj_done
            )
            traj_returns = traj_advantages + torch.tensor(traj_old_values).float()

            # È¥µôÕâÀïµÄ¸öÌå¶ÀÁ¢±ê×¼»¯£¬ÒÆµ½È«¾Ö×ö
            # traj_advantages = (traj_advantages - traj_advantages.mean()) / (traj_advantages.std() + 1e-8)

            traj_advantages_tensor = traj_advantages.clone().detach().float()
            traj_returns_tensor = traj_returns.clone().detach().float()
            
            advantages_all.append(traj_advantages_tensor)
            returns_all.append(traj_returns_tensor)
            
        # Ê¹ÓÃ dim=1 ½«Î¬¶È×ª»Ø (STEPS, pop_size)
        advantages = torch.stack(advantages_all, dim=1)  
        returns = torch.stack(returns_all, dim=1)
        
        # È«¾ÖÍ³Ò»±ê×¼»¯ Advantage£¬±£ÁôÓÅÁÓ²îÒì
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        #print(np.array(advantages).shape, np.array((returns)).shape)

        #with torch.no_grad():
        #    old_values = self.old_v(s, self.edge_index).squeeze()
        #    old_next_values = self.old_v(s_, self.edge_index).squeeze()
        #advantages = self.compute_advantages(r, old_values, old_next_values, done)
        ##print("old_values old_next_values advantage:",old_values,old_next_values, advantages)
        #returns = advantages + old_values

        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        STEPS, pop_size, N, F_in = s.shape

        for _ in range(K_epoch):
            s_flat_steps = s.view(-1, N, F_in) # (STEPS * pop_size, 22, 12)
            current_logits = self.pi(s_flat_steps, self.edge_index) 
            
            # 2. ½«Êä³ö½á¹û²ð½â»Ø (STEPS, pop_size, 14, 3)
            current_logits = current_logits.view(STEPS, pop_size, design_param, 3)
            
            # ÓÉÓÚµ±Ç° logits ÊÇ (STEPS, pop_size, 14, 3)
            # PyTorch µÄ Categorical ÆÚÍû×îºóÔÚÒ»Î¬ÊÇ¸ÅÂÊ£¬Î¬¶ÈÒÑ¾­ÕýÈ·£¬²»ÐèÒª permute
            probs = F.softmax(current_logits, dim=-1)
            current_dist = Independent(Categorical(probs), 1)
            log_probs = current_dist.log_prob(a)
            #print("log_probs:",current_dist, log_probs,old_log_probs.squeeze(),a)

            ratios = torch.exp(log_probs - old_log_probs.squeeze())


            entropy = current_dist.entropy().mean()
            entropy_weight = max(0.01 * (0.995 ** self.step), 0.002)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-CLIP, 1+CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - entropy * entropy_weight
        
            current_values = self.v(s_flat_steps, self.edge_index).squeeze()
            current_values = current_values.view(STEPS, pop_size)
            value_loss = F.mse_loss(current_values, returns)
            #print("Value Loss:", value_loss.item(),current_values,returns)
            #print("Policy Loss:", policy_loss.item(), ratios, advantages)

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
        #print('...save model...')
 
    def load(self):
        try:
            self.pi.load_state_dict(torch.load('pi_15_pop.pth'))
            self.v.load_state_dict(torch.load('v_15_pop.pth'))
            #print('...load...')
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
    reward_model = None#RewardModel(design_param)
    env = gym.make(env_id, disable_env_checker=True)
    #env._init_random_sim(100)
    agent = Agent(reward_model)
    agent.load()
    max_rewards = -1000000
    #creator.create("FitnessMin", base.Fitness, weights=((-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0)))
    #creator.create("Individual", list, fitness=creator.FitnessMin)
    for j in range(EP_MAX):
        s, _ = env.reset()
        rewards = []
        for i in range(STEPS):
            #print("steps:", i)
            a, old_log_prob = agent.choose_action(torch.tensor(s, dtype=torch.float))
            s_, r, done, parameters, _, info_batch = env.step(a.numpy())
            info = info_batch['info_batch']
            single_info = []
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for i in range(len(info)):
                    single_info.append([info[i]['Power'][1],
                                        info[i]['dcgain'][1],
                                        info[i]['GBW'][1],
                                        info[i]['phase_margin (deg)'][1],
                                        info[i]['TC'][1],
                                        info[i]['vos'][1],
                                        info[i]['cmrrdc'][1],
                                        info[i]['PSRP'][1],
                                        info[i]['PSRN'][1],
                                        info[i]['sr'][1],
                                        info[i]['settlingTime'][1],
                                        info[i]['reward']])
                    writer.writerow([info[i]['Power'][0],
                                        info[i]['dcgain'][0],
                                        info[i]['GBW'][0],
                                        info[i]['phase_margin (deg)'][0],
                                        info[i]['TC'][0],
                                        info[i]['vos'][0],
                                        info[i]['cmrrdc'][0],
                                        info[i]['PSRP'][0],
                                        info[i]['PSRN'][0],
                                        info[i]['sr'][0],
                                        info[i]['settlingTime'][0],
                                        info[i]['reward']])
            agent.push_data((s, a, r, s_, done, old_log_prob, parameters, single_info))
            rewards += r
            #if done:
            #    break
            s = s_
        agent.update()
        episode_reward_sum = sum(rewards)
        if max_rewards < episode_reward_sum:
            max_rewards = episode_reward_sum
            #agent.save()
 
if __name__ == '__main__':
    main()
