"""
A simple market game where at each time step there are two random prices at the market: 1) Ask price and 2) Bid price.
All prices are randomly generated in the range of [0:10].
The agent wants to earn profit through market arbitrage, which means buying at a lower price and then selling at a higher price. 
The agent has 100 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.
A reinforcement learning model is developed to train the agent to perform effective arbitrage.
The agent's performance is then backtested and compared against the optimal trading decisions (found via mathematical optimisation) and a heuristic trading policy.

- This 'train_and_save' script trains and saves a RL agent
  To test the trained RL agent and analyse its actions, run the evaluation script 'run_and_analyse'
  To visualise the agent's policy, run 'visualise_policy'

Note: this version of the code allows impossible actions, e.g., selling out of the empty inventory, but penalises them.

Andrey Churkin https://andreychurkin.ru/

"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from more_realistic_market_game_2_env import RandomPriceMarketEnv
import numpy as np
import time
import matplotlib.pyplot as plt

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Track start time
start_time = time.time()



def train_agent(total_timesteps=200_000):
    # Create the environment
    env = RandomPriceMarketEnv()

    # Wrap the environment to allow logging of training data
    env = Monitor(env)
    
    # Create the PPO agent
    rl_model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    rl_model.learn(total_timesteps=total_timesteps)
    
    return rl_model, env



def evaluate_agent(rl_model, env, episodes=10):
    all_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_rewards = []
        done = False
        
        while not done:
            action, _ = rl_model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
        
        cumulative_episode_rewards = np.cumsum(episode_rewards)
        all_rewards.append(cumulative_episode_rewards)
        
        print(f"Episode {episode + 1}: Total Reward = {sum(episode_rewards):.2f}")

    # Plot results
    plt.figure(figsize=(10, 5))

    for i, rewards_history in enumerate(all_rewards):
        plt.plot(rewards_history, label=f'Episode {i+1}')
    
    plt.title('Evaluation: cumulative rewards over time per episode')
    plt.xlabel('Step')
    plt.ylabel('Сumulative reward')
    plt.legend(fontsize=8, loc='upper left')
    plt.grid(True, color='gray', alpha=0.2)  # Major grid
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

    plt.tight_layout()

    plt.savefig("plt_10_evaluation_episodes.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("plt_10_evaluation_episodes.png", format="png", bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    # Train the agent
    print("Training the agent...")
    rl_model, env = train_agent()

    # After training, access the logged data
    monitor_data_episode_rewards = env.get_episode_rewards()
    monitor_data_episode_lengths = env.get_episode_lengths()

    print("\nThe model has been successfully trained. Total number of episodes = ",len(monitor_data_episode_lengths))

    # Plot rewards per episode
    plt.rcParams['font.family'] = 'Courier New'
    plt.plot([i for i in range(0,len(monitor_data_episode_rewards))], monitor_data_episode_rewards)
    plt.xlabel('Training episode')
    plt.ylabel('Reward')
    plt.title('Total training reward vs episode')
    plt.grid(True, color='gray', alpha=0.2)  # Major grid
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid
    plt.show()
    
    # Evaluate the agent
    print("\nEvaluating the agent...")
    evaluate_agent(rl_model, env)
    
    # Save the RL model
    rl_model.save("ppo_market_game_2_v7")
    print("\nThe trained model has been saved")


"""
To test the trained RL agent and analyse its actions, run the evaluation script 'run_and_analyse'
To visualise the agent's policy, run 'visualise_policy'
"""