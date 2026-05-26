import matplotlib.pyplot as plt
import csv
import ast

def load_rewards_from_csv(file_path):
    rewards = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            reward_list = ast.literal_eval(row[0])
            rewards.append(reward_list)
    
    return rewards

def main():
    try:
        # ¶ÁÈ¡CSVÎÄ¼þ£¬¼ÙÉèÃ¿ÐÐÊÇÒ»¸öÊýÖµ
        #with open('reward_nsga_success.csv', 'r') as f:
        with open('rewardc.csv', 'r') as f:
            data = [float(line.strip()) for line in f]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Ã¿30¸öÔªËØ·ÖÒ»×é£¬²»×ã30¸öÒ²×÷ÎªÒ»×é
    groups = [data[i:i+64] for i in range(0, len(data), 64)]

    # ¼ÆËãÃ¿×éµÄÆ½¾ùÖµ
    averages = [sum(g)/len(g) for g in groups]

    # »æÖÆÍ¼±í
    plt.figure(figsize=(15, 6))
    plt.plot(averages, marker='', linestyle='-')
    plt.xlabel('Training Batch (Every 64 Episodes)')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.xticks(range(len(averages)), [f"{i*300}" for i in range(len(averages))])
    plt.tight_layout()
    plt.show()

def group_average(data, group_size):
    groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]
    averages = [sum(g)/len(g) for g in groups]
    return averages

def listmain():

    reward_by_env = []

    try:
        # ¶ÁÈ¡CSVÎÄ¼þ£¬¼ÙÉèÃ¿ÐÐÊÇÒ»¸öÊýÖµ
        #with open('reward_nsga_success.csv', 'r') as f:
        with open('reward.csv', 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                reward_list = ast.literal_eval(line[0])

                if not reward_by_env:
                    reward_by_env = [[] for _ in range(len(reward_list))]

                for i, r in enumerate(reward_list):
                    reward_by_env[i].append(r)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    group_size = 64
    averaged_rewards_by_env = [group_average(env_rewards, group_size) for env_rewards in reward_by_env]

    plt.figure(figsize=(15, 6))

    num_envs = len(reward_by_env)
    for i in range(num_envs):
        plt.plot(averaged_rewards_by_env[i], label=f'Env {i+1}', alpha=0.7)
    plt.xlabel(f'Training Batch (Every {group_size} Episodes)')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.xticks(range(len(averages)), [f"{i*300}" for i in range(len(averages))])
    plt.tight_layout()
    plt.show()

def xmain():
    try:
        # ¶ÁÈ¡CSVÎÄ¼þ£¬¼ÙÉèÃ¿ÐÐÊÇÒ»¸öÊýÖµ
        with open('entropy_64step_15pop.csv', 'r') as f:
            data = [float(line.strip()) for line in f]
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    groups = [data[i:i+64] for i in range(0, len(data), 64)]

    # ¼ÆËãÃ¿×éµÄÆ½¾ùÖµ
    averages = [sum(g)/len(g) for g in groups]

    # »æÖÆÍ¼±í
    plt.figure(figsize=(15, 6))
    plt.plot(averages, marker='', linestyle='-')
    plt.xlabel('Training Batch (Every 64 Episodes)')
    plt.ylabel('Average Entropy')
    plt.title('Training Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    #plt.xticks(range(len(averages)), [f"{i*300}" for i in range(len(averages))])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    listmain()
    #main()