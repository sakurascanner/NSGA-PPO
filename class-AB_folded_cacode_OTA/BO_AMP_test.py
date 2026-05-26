import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.transforms import NormalizeFeatures
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
import numpy as np
from datetime import datetime
from bo_vanguard import run_bo_vanguard
from deap import tools,creator,base
import os

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
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry[env]

print("Register the environment")
register(
        id = env_id,
        entry_point = 'AMP_NMCF:BoPpoAMPNMCFEnv',
        max_episode_steps = None,
        )

#env = gym.make(env_id)

EP_MAX = 10000 #max episode
STEPS = 64  #one episode contains STEPS steps
LR_v = 1e-4    #value network learning rate
LR_pi = 3e-4   #policy network learning rate
LR_r = 5e-5    #reward network learning rate
K_epoch = 4    #one sample reused times
GAMMA = 0.99   #discount rate
LAMBDA = 0.95  #generalized advantage estimation
CLIP = 0.2     #clip
observation_param = 12
design_param = 24
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
        
    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = torch.tanh(self.norm1(self.conv1(x, edge_index)))
        x = torch.tanh(self.norm2(self.conv2(x, edge_index)))

        x = self.pool(x, batch)

        logits = torch.stack(
            [head(x) for head in self.heads],
            dim=1
        )

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

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = F.tanh(self.conv1(x, edge_index))
        x = F.tanh(self.conv2(x, edge_index))

        x = self.pool(x, batch)

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
        
    def build_batch_graph(self, s):
        """
        s shape:
            single graph: [29, 12]
            batch graph : [B, 29, 12]
        """

        # Single environment case
        if s.dim() == 2:
            data = Data(
                x=s,
                edge_index=self.edge_index
            )
            return Batch.from_data_list([data])

        # Batch environment case
        data_list = []

        for i in range(s.shape[0]):
            data = Data(
                x=s[i],
                edge_index=self.edge_index
            )
            data_list.append(data)

        batch_graph = Batch.from_data_list(data_list)

        return batch_graph
 
    def choose_action(self, s):
        with torch.no_grad():   
            graph_batch = self.build_batch_graph(s)
            logits = self.old_pi(graph_batch)
            logits = logits.view(-1, design_param, 3)
            probs = F.softmax(logits, dim=-1)
            print("logits:",logits)
            dist = Independent(Categorical(probs),1)
            entropy = dist.entropy().mean()
            #print("Probs and Policy Entropy:",probs , entropy.item())
            with open("entropy.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([entropy.item()])
            action = dist.sample()
            old_log_probs = (dist.log_prob(action)).detach()
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
            
            # --- FIX: Wrap old_log_prob in a torch.tensor ---
            l_old_prob.append(torch.tensor(old_log_prob, dtype=torch.float))
            # ------------------------------------------------
            
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
            #print(f"[DEBUG] t={t}, reward={rewards[t]}, value={values[t]}, next_value={next_values[t]}, done={dones[t]}")
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
            advantages.insert(0, last_advantage)
        return torch.tensor(advantages).float()
 
    def updata(self):
        self.step += 1
        s, a, r, s_, done, old_log_probs, parameter, info_score = self.sample()

        #training reward model
        # if self.reward_model:
        #     expected_score = self.reward_model(parameter).squeeze()
        #     loss = F.mse_loss(expected_score, info_score)
        #     self.reward_model.optim.zero_grad()
        #     loss.backward()
        #     self.reward_model.optim.step()

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
                traj_graph = self.build_batch_graph(traj_s)
                traj_old_values = self.old_v(traj_graph).squeeze().cpu().numpy()
                traj_next_graph = self.build_batch_graph(traj_s_)
                traj_old_next_values = self.old_v(traj_next_graph).squeeze().cpu().numpy()

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
        
        print(np.array(advantages).shape, np.array((returns)).shape)

        #with torch.no_grad():
        #    old_values = self.old_v(s, self.edge_index).squeeze()
        #    old_next_values = self.old_v(s_, self.edge_index).squeeze()
        #advantages = self.compute_advantages(r, old_values, old_next_values, done)
        #print("old_values old_next_values advantage:",old_values,old_next_values, advantages)
        #returns = advantages + old_values

        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = STEPS * pop_size
        
        s_flat = s.reshape(batch_size, s.shape[2], s.shape[3]) 
        a_flat = a.reshape(batch_size, -1) 
        old_log_probs_flat = old_log_probs.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)
        # ----------------------------

        mini_batch_size = 128

        dataset_size = batch_size

        for _ in range(K_epoch):

            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, mini_batch_size):

                end = start + mini_batch_size
                mb_idx = indices[start:end]

                s_mb = s_flat[mb_idx]
                a_mb = a_flat[mb_idx].long()
                old_log_probs_mb = old_log_probs_flat[mb_idx]
                advantages_mb = advantages_flat[mb_idx]
                returns_mb = returns_flat[mb_idx]

                graph_batch = self.build_batch_graph(s_mb)

                current_logits = self.pi(graph_batch)

                current_logits = current_logits.view(-1, design_param, 3)

                probs = F.softmax(current_logits, dim=-1)

                current_dist = Independent(Categorical(probs), 1)

                log_probs = (
                    current_dist.log_prob(a_mb)
                )

                ratios = torch.exp(log_probs - old_log_probs_mb)

                entropy = (
                    current_dist.entropy()
                ).mean()

                entropy_weight = max(
                    0.002,
                    0.01 * (0.9995 ** self.step)
                )

                surr1 = ratios * advantages_mb

                surr2 = torch.clamp(
                    ratios,
                    1 - CLIP,
                    1 + CLIP
                ) * advantages_mb

                policy_loss = -torch.min(
                    surr1,
                    surr2
                ).mean()

                policy_loss -= entropy * entropy_weight

                current_values = self.v(graph_batch).squeeze()

                value_loss = F.mse_loss(
                    current_values,
                    returns_mb
                )

                self.pi.optim.zero_grad()
                self.v.optim.zero_grad()

                (policy_loss + value_loss).backward()

                torch.nn.utils.clip_grad_norm_(
                    self.pi.parameters(),
                    0.5
                )

                torch.nn.utils.clip_grad_norm_(
                    self.v.parameters(),
                    0.5
                )

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
            self.pi.load_state_dict(torch.load('pi_15_pop.pth'))
            self.v.load_state_dict(torch.load('v_15_pop.pth'))
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
    reward_model = None#RewardModel(design_param)
    env = gym.make(env_id)
    #env._init_random_sim(100)
    agent = Agent(reward_model)
    agent.load()
    max_rewards = -1000000
    s, _ = env.reset()
    #creator.create("FitnessMin", base.Fitness, weights=((-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0)))
    #creator.create("Individual", list, fitness=creator.FitnessMin)
    for j in range(EP_MAX):
        rewards = []
        for i in range(STEPS):
            print("steps:", i)
            a, old_log_prob = agent.choose_action(torch.tensor(s, dtype=torch.float))
            s_, r, done, parameters, info_batch = env.step(a.squeeze())
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
                    writer.writerow([r[i],info[i]['Power'][0],
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
        agent.updata()
        episode_reward_sum = sum(rewards)
        if max_rewards < episode_reward_sum:
            max_rewards = episode_reward_sum
            #agent.save()
 
def make_env(optimal_x0):
    """Utility function to create envs for the VectorEnv"""
    def thunk():
        env = gym.make(env_id, optimal_x0=optimal_x0) # Pass x0 to your new environment
        return env
    return thunk

def xmain():
    # 1. Run the BO Vanguard first to find the optimal starting parameters
    print("Running BO Vanguard to find optimized x0...")
    
    env_bounds_low = np.array([ 0.5, 0.5, 1,    # M0(W_low,L_low,M_low)  (M is index 2)
                                0.5, 0.5, 1,    # M8                     (M is index 5)
                                0.5, 0.5, 1,    # M10                    (M is index 8)
                                0.5, 0.5, 100,  # M11                    (M is index 11)
                                0.5, 0.5, 1,    # M17                    (M is index 14)
                                0.5, 0.5, 1,    # M21                    (M is index 17)
                                0.5, 0.5, 1,    # M23                    (M is index 20)
                                3e-6,           # Ib                     
                                1,              # C0                     (Index 22)
                                1])             # C1                     (Index 23)
                                
    env_bounds_high = np.array([10, 4, 50,  
                                10, 4, 50,      
                                10, 4, 50,     
                                10, 4, 500,    
                                10, 4, 50,     
                                10, 4, 50,    
                                10, 5, 50,    
                                20e-6,        
                                50,    
                                50])   
    
    # Define the exact indices that MUST be integers
    idx_integer_params = [2, 5, 8, 11, 14, 17, 20, 22, 23]
    
    optimal_x0 = run_bo_vanguard(
        env_bounds_low, 
        env_bounds_high, 
        max_simulations=50, 
        int_indices=idx_integer_params
    )
    
    print(f"BO Complete. Optimal x0: {optimal_x0}")

    # 2. Initialize Parallel Environments using Gym's AsyncVectorEnv
    num_envs = 16 
    envs = gym.vector.AsyncVectorEnv([make_env(optimal_x0) for _ in range(num_envs)])

    agent = Agent()
    # agent.load()
    max_rewards = -1000000
    s, _ = envs.reset() 

    for j in range(EP_MAX):
        # s shape will be (num_envs, obs_shape) natively!
        rewards = []
        
        for i in range(STEPS):
            print(f"Episode {j}, Step {i}")
            # Ensure s is cast to float32 for PyTorch
            a, old_log_prob = agent.choose_action(torch.tensor(s, dtype=torch.float32))
            
            # Step all parallel environments at once
            s_, r, done, truncated, info = envs.step(a.detach().numpy())
            true_next_states = s_.copy()
            # ==============================================================
            # NEW LOGGING BLOCK: Adjusted for Gym's AsyncVectorEnv dict stacking
            # ==============================================================
            single_info = []
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                for env_idx in range(num_envs):
                    actual_info = info
                    
                    # 2. Extract true terminal states if the env just finished
                    if 'final_observation' in info and info['final_observation'][env_idx] is not None:
                        true_next_states[env_idx] = info['final_observation'][env_idx]
                        
                    if 'final_info' in info and info['final_info'][env_idx] is not None:
                        actual_info = info['final_info'][env_idx]
                    else:
                        actual_info = {k: v[env_idx] for k, v in info.items() if hasattr(v, '__getitem__')}
                    
                    # 1. Append Scores to single_info
                    single_info.append([
                        actual_info['Power'][1],
                        actual_info['dcgain'][1],
                        actual_info['GBW'][1],
                        actual_info['phase_margin (deg)'][1],
                        actual_info['TC'][1],
                        actual_info['vos'][1],
                        actual_info['cmrrdc'][1],
                        actual_info['PSRP'][1],
                        actual_info['PSRN'][1],
                        actual_info['sr'][1],
                        actual_info['settlingTime'][1],
                        actual_info['reward']
                    ])
                    
                    # 2. Write Values to CSV
                    writer.writerow([
                        r[env_idx],
                        actual_info['Power'][0],
                        actual_info['dcgain'][0],
                        actual_info['GBW'][0],
                        actual_info['phase_margin (deg)'][0],
                        actual_info['TC'][0],
                        actual_info['vos'][0],
                        actual_info['cmrrdc'][0],
                        actual_info['PSRP'][0],
                        actual_info['PSRN'][0],
                        actual_info['sr'][0],
                        actual_info['settlingTime'][0],
                        actual_info['reward']
                    ])
            # ==============================================================

            # Push to agent (passing a.numpy() as the 'parameters' argument in case your RewardModel needs it later)
            agent.push_data((s, a.numpy(), r, true_next_states, truncated, old_log_prob.numpy(), a.numpy(), single_info))
            
            rewards.append(r)
            s = s_
            
        agent.updata()
        
        # Calculate mean episode reward across all parallel envs
        episode_reward_sum = np.sum(rewards, axis=0).mean() 
        print(f"Mean Episode Reward: {episode_reward_sum}")
        
        if max_rewards < episode_reward_sum:
            max_rewards = episode_reward_sum
            agent.save()

if __name__ == '__main__':
    xmain()