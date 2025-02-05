"""
A simple market game where at each time step there are two random prices at the market: 1) Ask price and 2) Bid price.
All prices are randomly generated in the range of [0:10].
The agent wants to earn profit through market arbitrage, which means buying at a lower price and then selling at a higher price. 
The agent has 100 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.
A reinforcement learning model is developed to train the agent to perform effective arbitrage.
The agent's performance is then backtested and compared against the optimal trading decisions (found via mathematical optimisation) and a heuristic trading policy.

- This 'env' script only formulates the environment. To train and save a RL agent for this environment, run 'train_and_save'

Note: this version of the code allows impossible actions, e.g., selling out of the empty inventory, but penalises them.

Andrey Churkin https://andreychurkin.ru/

"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import time

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Track start time
start_time = time.time()


""" Select how the Bid/Ask prices will be randomly generated """
# price_generation_mode = "all_random" # <--- Bid and Ask prices are completely random and independent, within the [0:10] range (less realistic, but explores different price combinations)
# price_generation_mode = "ask_always_slightly_higher" # <--- Ask price is always a bit higher, but still within [0:10] range (this is a more realistic market representation)
price_generation_mode = "ask_always_higher" # <--- Ask price can be much higher, but within [0:10] range (this is a more realistic market representation)


# # ---------- Create the environment: ----------
class RandomPriceMarketEnv(gym.Env):
    """
    A simple trading environment where an agent can buy and sell at different prices.
    The goal is to maximize profit through arbitrage.
    
    State space: [ask price, bid price, inventory]
    Action space: 0 (do nothing), 1 (buy), 2 (sell)
    """

    def __init__(self):
        super(RandomPriceMarketEnv, self).__init__()
        
        # Observation space: [ask price, bid price, inventory]
        self.observation_space = Box(low=np.array([0, 0, 0]),
                                            high=np.array([10, 10, 100]),
                                            dtype=np.float32)

        # Action space: 0 = Do Nothing, 1 = Buy, 2 = Sell
        self.action_space = Discrete(3)

        # Initialize environment variables
        self.ask_price = None
        self.bid_price = None
        self.inventory = None
        self.current_step = None
        self.max_steps = 100
        self.total_profit = None


    """ Define a seed to generate the same prices in all episodes (for testing purposes) """
    select_seed = None # <-- random episodes
    # select_seed = 42 # <-- a deterministic test
    

    def reset(self, seed=select_seed, options=None): 
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Randomize initial prices, reset the inventory and step counter:
        if price_generation_mode == "all_random":
            self.ask_price = self.np_random.uniform(0, 10)
            self.bid_price = self.np_random.uniform(0, 10)
        elif price_generation_mode == "ask_always_slightly_higher":
            self.bid_price = np.random.uniform(0, 10)
            self.ask_price = min(self.bid_price + np.random.uniform(0, 1), 10)  # Ask price is slightly higher, but not exceeding 10
        elif price_generation_mode == "ask_always_higher":
            self.bid_price = np.random.uniform(0, 10)
            self.ask_price = self.bid_price + np.random.uniform(0, 10 - self.bid_price)  # Ask price can be much higher, but within [0:10] range

        self.inventory = 0
        self.current_step = 0
        self.total_profit = 0
        
        observation = np.array([
            self.ask_price, 
            self.bid_price, 
            self.inventory
            ], dtype=np.float32)
        return observation, {}

    def step(self, action):
        self.current_step += 1
        reward = 0
        done = False

        if action == 0:  # <-- Do nothing
            reward = 0
            # reward = -0.1 # add penalty

        elif action == 1:  # <-- Buy
            reward = -self.ask_price
            self.inventory += 1

        elif action == 2:  # <-- Sell
            if self.inventory > 0:
                reward = self.bid_price
                self.inventory -= 1
            else:  # <-- Invalid action (selling with no inventory)
                # reward = 0 # do nothing - simply a useless action
                # reward = -1 # add penalty
                reward = -10 # add penalty
                # reward = -20 # add an even larger penalty

        # Update total profit:
        self.total_profit += reward

        # Generate new prices for next step:
        if price_generation_mode == "all_random":
            self.ask_price = self.np_random.uniform(0, 10)
            self.bid_price = self.np_random.uniform(0, 10)
        elif price_generation_mode == "ask_always_slightly_higher":
            self.bid_price = np.random.uniform(0, 10)
            self.ask_price = min(self.bid_price + np.random.uniform(0, 1), 10)  # Ask price is slightly higher, but not exceeding 10
        elif price_generation_mode == "ask_always_higher":
            self.bid_price = np.random.uniform(0, 10)
            self.ask_price = self.bid_price + np.random.uniform(0, 10 - self.bid_price)  # Ask price can be much higher, but within [0:10] range

        # Construct the new state
        next_state = np.array([
            self.ask_price, 
            self.bid_price, 
            self.inventory
            ], dtype=np.float32)

        # Terminate the episode if max steps are reached
        if self.current_step >= self.max_steps:
            done = True

        return next_state, reward, done, False, {}

    def render(self):
        print(f"Step #{self.current_step} --> Ask Price: {self.ask_price:.2f}; Bid Price: {self.bid_price:.2f}; Inventory: {self.inventory}; Total Profit: {self.total_profit:.2f}")



# # ---------- Test the environment: ----------
if __name__ == "__main__":
    env = RandomPriceMarketEnv()
    
    # Reset environment
    obs, _ = env.reset()
    print("\nInitial State:")
    env.render()
    
    # Run a few random steps
    print("\nTaking some random actions...")
    for i in range(5):
        # Take random action
        action = env.action_space.sample()
        print(f"\nStep #{i+1} --> Action = {action}")
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print state
        env.render()
        print(f"Reward: {reward}")


"""
This script only formulates the environment
To train and save a RL agent for this environment, run 'train_and_save'
"""