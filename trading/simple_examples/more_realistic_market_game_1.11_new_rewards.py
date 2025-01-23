"""
A more realistic market game where at each time step, there are two random prices at the market: 1) sell price, 2) buy price. 
The agent wants to earn profit due to market arbitrage. That is, to buy at a lower price and then sell at a higher price. 
The agent has 1000 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.

! Note: this version of the code allows impossible actions, e.g., selling out of the empty inventory, but penalises them !

Andrey Churkin https://andreychurkin.ru/

"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time
import matplotlib.pyplot as plt


# Track start time
start_time = time.time()


class MoreRealisticMarketGameEnv(gym.Env):
    def __init__(self):
        super(MoreRealisticMarketGameEnv, self).__init__()
        
        # Observation space: [sell_price, buy_price, inventory, step/max_step ratio]
        self.observation_space = Box(low=np.array([0, 0, 0, 0]),
                                            high=np.array([10, 10, 100, 1]),
                                            dtype=np.float32)

        # Action space: 0 = Do Nothing, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)

        # Initialize environment variables
        self.sell_price = None
        self.buy_price = None
        self.inventory = None
        self.current_step = None
        self.max_steps = 100
        self.total_profit = None

    """ The same seed is used to reset the episode (for testing purposes) """
    select_seed = 42 # <-- my deterministic test
    # select_seed = None # <-- random episodes

    def reset(self, seed=select_seed, options=None): 
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Randomize initial prices and reset inventory and step counter
        self.sell_price = self.np_random.uniform(0, 10)
        self.buy_price = self.np_random.uniform(0, 10)
        self.inventory = 0
        self.current_step = 0
        self.total_profit = 0
        
        return np.array([self.sell_price, self.buy_price, self.inventory, self.current_step/self.max_steps], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        # Update sell and buy prices randomly at each step
        self.sell_price = self.np_random.uniform(0, 10)
        self.buy_price = self.np_random.uniform(0, 10)

        reward = 0
        done = False

        if action == 0:  # Do nothing
            reward = 0
            # reward = -0.1 # add penalty


        elif action == 1:  # Buy
            # reward = -self.buy_price
            # reward = 0 # testing free buying
            # reward = 10 - self.buy_price # incentive to buy low
            reward = (10 - self.buy_price)/10 # incentive to buy low, but selling is more important


            self.inventory += 1

        elif action == 2:  # Sell
            if self.inventory > 0:
                reward = self.sell_price # incentive to sell high

                self.inventory -= 1
            else:  # Invalid action (selling with no inventory)
                """
                Haozhe: I think penalization is not the best to enforce the rule as it alters the optimal solution to the dynamic programming problem, thus needs careful analysis.
                Andrey: However, without a penalty, the agent learns never to buy. It only sells all the time.
                """
                # reward = -10 # add penalty
                # reward = -1 # add penalty
                # reward = -20 # add an even larger penalty
                reward = 0 # do nothing - simply a useless action

        # Update total profit
        self.total_profit += reward

        # Construct the new state
        next_state = np.array([self.sell_price, self.buy_price, self.inventory, self.current_step/self.max_steps], dtype=np.float32)

        # Penalise inventory at the last step
        if self.current_step == self.max_steps:
            reward -= 10*self.inventory

        # Terminate the episode if max steps are reached
        if self.current_step >= self.max_steps:
            done = True

        return next_state, reward, done, False, {}

    def render(self):
        print(f"\nStep #{self.current_step}; Sell Price: {self.sell_price:.2f}; Buy Price: {self.buy_price:.2f}; Inventory: {self.inventory}; Total Profit: {self.total_profit:.2f}")



# Instantiate the environment
env = MoreRealisticMarketGameEnv()

# Wrap the environment to allow logging of training data
env = Monitor(env)


# Train a RF model
model = PPO("MlpPolicy", env, verbose=1)
# model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100_000)


# After training, access the logged data
monitor_data_episode_rewards = env.get_episode_rewards()
monitor_data_episode_lengths = env.get_episode_lengths()

# print("monitor_data_episode_rewards = ",monitor_data_episode_rewards)
# print("monitor_data_episode_lengths = ",monitor_data_episode_lengths)


# Save the trained model
# model.save("ppo_morerealisticmarketgame_1")
# model.save("ppo_morerealisticmarketgame_2")
model.save("ppo_morerealisticmarketgame_15_new_rewards_test")


# model.save("dqn_morerealisticmarketgame_1")



print("\nThe model has been successfully trained. Total number of episodes = ",len(monitor_data_episode_lengths))


# Calculate total time taken
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken for training: {elapsed_time:.2f} seconds")



# Plot rewards per episode
plt.plot([i for i in range(0,len(monitor_data_episode_rewards))], monitor_data_episode_rewards)
plt.xlabel('Training episode')
plt.ylabel('Reward')
plt.title('Total training reward vs episode')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid
plt.show()


# # Evaluate the agent
n_eval_episodes = 10 # episodes for testing the trained agent
print(f"\nEvaluating the agent with {n_eval_episodes} episodes ...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes)
print(f"Evaluation mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
