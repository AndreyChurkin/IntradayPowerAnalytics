"""
A simple market game where at each time step there are two random prices at the market: 1) sell price and 2) buy price.
All prices are randomly generated in the range of [0:10].
The agent wants to earn profit through market arbitrage, which means buying at a lower price and then selling at a higher price. 
The agent has 100 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.
A reinforcement learning model is developed to train the agent to perform effective arbitrage.
The agent's performance is then backtested and compared against the optimal trading decisions (found via mathematical optimisation) and a heuristic trading policy.

Note: this version of the code allows impossible actions, e.g., selling out of the empty inventory, but penalises them.

Andrey Churkin https://andreychurkin.ru/

"""

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from gurobipy import Model, GRB, quicksum
import time
import matplotlib.pyplot as plt

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Track start time
start_time = time.time()


class RandomPriceMarketEnv(gym.Env):
    def __init__(self):
        super(RandomPriceMarketEnv, self).__init__()
        
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

    """ Define a seed to generate the same prices in all episodes (for testing purposes) """
    select_seed = None # <-- random episodes
    # select_seed = 42 # <-- a deterministic test
    

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
            # reward = 0
            reward = -0.1 # add penalty


        elif action == 1:  # Buy
            reward = -self.buy_price
            # reward = 0 # testing free buying
            # reward = 10 - self.buy_price # incentive to buy low
            # reward = (10 - self.buy_price)/10 # incentive to buy low, but selling is more important

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
                reward = -1 # add penalty
                # reward = -20 # add an even larger penalty
                # reward = 0 # do nothing - simply a useless action

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
env = RandomPriceMarketEnv()

# Load the trained RL model:
model = PPO.load("ppo_random_price_market_test1", env)



# # ---------- Test the trained agent, get its actions for further visualisation: ----------

# Initialize lists to store variables
buy_prices = []
sell_prices = []
actions = []
rewards = []
inventory = []

# Run a single test episode
state, info = env.reset()
done = False

while not done:
    # Get the action from the trained agent
    action, _states = model.predict(state, deterministic=True)

    # Take a step in the environment
    next_state, reward, done, truncated, info = env.step(action)

    # Log variables
    sell_prices.append(state[0]) 
    buy_prices.append(state[1]) 
    inventory.append(state[2])
    actions.append(action)   # The agent's chosen action
    rewards.append(reward)   # The reward for this step

    # Update state
    state = next_state

cumulative_rewards = np.cumsum(rewards)

test_steps = 100

print(f"\n-----------------------------------------------------------------")
print(f"The agent's trading policy has been tested for {test_steps} steps")
print(f"Total agent's profit: {sum(rewards[:test_steps])}")
print(f"The agent's strategy includes: {actions[:test_steps].count(0)} 'Do nothing' actions, {actions[:test_steps].count(1)} 'Buy' actions, {actions[:test_steps].count(2)} 'Sell' actions")
print(f"Maximum inventory reached during trading: {max(inventory)}")
print(f"Minimum inventory reached during trading: {min(inventory)}")
print(f"Inventory at the end of the episode: {inventory[-1]}")
print(f"-----------------------------------------------------------------")



# # ---------- Create a simple heuristic trading policy as a benchmark: ----------

# h_max_steps = 100
h_max_steps = test_steps
def heuristic_trading_policy(max_steps=h_max_steps):
    """
    A simple heuristic trading policy for the RandomPriceMarketEnv.
    - Buy when the buy price < 5.
    - Sell when the sell price > 5 (if inventory > 0).
    - Do nothing otherwise.
    """

    heuristic_rewards = []
    heuristic_actions = []
    h_inventory = 0
    h_inventory_peak = 0

    for h_step in range(max_steps):
        h_sell_price = sell_prices[h_step]
        h_buy_price = buy_prices[h_step]
        
        # Apply the heuristic policy
        if h_sell_price > 5 and h_inventory > 0:  # Sell action
            h_action = 2
            h_inventory -= 1
            h_reward = h_sell_price
        elif h_buy_price < 5:  # Buy action
            h_action = 1
            h_inventory += 1
            h_reward = -h_buy_price
        else:  # Do nothing
            h_action = 0
            h_reward = 0

        # track the maximum inventory peak during trading
        if h_inventory > h_inventory_peak:
            h_inventory_peak = h_inventory

        # Update total profit and record the action
        heuristic_rewards.append(h_reward)
        heuristic_actions.append(h_action)

    return heuristic_rewards, heuristic_actions, h_inventory_peak


heuristic_rewards, heuristic_actions, heuristic_inventory_peak = heuristic_trading_policy()
cumulative_heuristic_rewards = np.cumsum(heuristic_rewards)

print(f"\nThe simple heuristic trading policy has been tested for {h_max_steps} steps")
print(f"Total heuristic profit: {sum(heuristic_rewards)}")
print(f"The heuristic strategy includes: {heuristic_actions.count(1)} 'Buy' actions, {heuristic_actions.count(2)} 'Sell' actions")
print(f"Maximum inventory reached during trading: {heuristic_inventory_peak}")



# # ---------- Finding the optimal trading solution using Gurobi: ----------

# Number of steps to consider in the optimisation problem
# opti_steps_max = 100
opti_steps_max = test_steps

# Define the initial inventory level
x_initial_inventory = 0
x_max_inventory = 100  # Maximum allowable inventory

# Create the Gurobi model
model = Model("TradingGameOptimization")

# Decision variables
x_buy = model.addVars(opti_steps_max, vtype=GRB.BINARY, name="Buy")      # 1 if buying, 0 otherwise
x_sell = model.addVars(opti_steps_max, vtype=GRB.BINARY, name="Sell")    # 1 if selling, 0 otherwise
x_inventory = model.addVars(opti_steps_max, vtype=GRB.INTEGER, name="Inventory", lb=0, ub=x_max_inventory)

# Objective: Maximize total profit
model.setObjective(
    quicksum(x_sell[t] * sell_prices[t] - x_buy[t] * buy_prices[t] for t in range(opti_steps_max)),
    GRB.MAXIMIZE
)

# Constraints
for t in range(opti_steps_max):
    if t == 0:
        # Inventory at the first timestep depends on initial inventory and action taken
        model.addConstr(x_inventory[t] == x_initial_inventory + x_buy[t] - x_sell[t], name=f"Inventory_Constraint_t{t}")
    else:
        # Inventory evolves over time
        model.addConstr(x_inventory[t] == x_inventory[t-1] + x_buy[t] - x_sell[t], name=f"Inventory_Constraint_t{t}")

    # Ensure that the agent cannot buy and sell at the same time
    model.addConstr(x_buy[t] + x_sell[t] <= 1, name=f"Mutually_Exclusive_Actions_t{t}")

# Solve the model
model.optimize()

# Check and print the results
if model.status == GRB.OPTIMAL:
    print(f"\nThe optimal solution for a session with {opti_steps_max} steps is found by Gurobi")
    print(f"The optimal total profit: {model.objVal}")
    x_buy_solution = [x_buy[t].X for t in range(opti_steps_max)]
    x_sell_solution = [x_sell[t].X for t in range(opti_steps_max)]
    x_inventory_solution = [x_inventory[t].X for t in range(opti_steps_max)]
    print(f"The optimal strategy includes: {sum(x_buy_solution)} 'Buy' actions, {sum(x_sell_solution)} 'Sell' actions")
    print(f"Maximum inventory reached during trading: {max(x_inventory_solution)}")
    optimal_rewards = []
    optimal_actions = []
    # compute the rewards according to the optimal actions
    for t in range(opti_steps_max):
        optimal_rewards.append(-x_buy_solution[t]*buy_prices[t] + x_sell_solution[t]*sell_prices[t])
        if x_buy_solution[t] == 1:
            optimal_actions.append(1)
        elif x_sell_solution[t] == 1:
            optimal_actions.append(2)
        else:
            optimal_actions.append(0)
    cumulative_optimal_rewards = np.cumsum(optimal_rewards)





# # ---------- Plotting the analysis: ----------

# plot_max = 100 # number of steps (trades) to visualise
plot_max = test_steps

# Plot the results
plt.figure(figsize=(12, 8))

# Plot prices and costs
plt.subplot(4, 1, 1)
plt.plot(buy_prices[:plot_max], label="Buy price")
plt.plot(sell_prices[:plot_max], label="Sell price")
plt.ylabel("Buy/sell price")
plt.legend(fontsize=7,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

# Plot actions
plt.subplot(4, 1, 2)
plt.plot(optimal_actions[:plot_max], label="Optimal actions (Gurobi) (0 = Hold, 1 = Buy, 2 = Sell)", drawstyle="steps-post",color="red")
plt.plot(heuristic_actions[:plot_max], label="Heuristic policy actions (0 = Hold, 1 = Buy, 2 = Sell)", drawstyle="steps-post",color="pink")
plt.plot(actions[:plot_max], label="Agent actions (0 = Hold, 1 = Buy, 2 = Sell)", drawstyle="steps-post",color=(0.3, 0.3, 0.3))
plt.ylabel("Actions")
plt.ylim([-0.5, 2.5])  # Since actions are binary
plt.legend(fontsize=7,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

# Plot rewards
plt.subplot(4, 1, 3)
plt.plot(optimal_rewards[:plot_max], label="Rewards per step - optimal (Gurobi)",color="red")
plt.plot(heuristic_rewards[:plot_max], label="Rewards per step - heuristic policy",color="pink")
plt.plot(rewards[:plot_max], label="Rewards per step - agent", color="green")
plt.ylabel("Reward per step")
plt.legend(fontsize=7,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

# Plot cumulative rewards
plt.subplot(4, 1, 4)
plt.plot(cumulative_optimal_rewards[:plot_max], label="Cumulative reward - optimal (Gurobi)",color="red")
plt.plot(cumulative_heuristic_rewards[:plot_max], label="Cumulative reward - heuristic policy",color="pink")
plt.plot(cumulative_rewards[:plot_max], label="Cumulative reward - agent", color="green")
plt.ylabel("Cumulative reward")
plt.xlabel("Evaluation time step - first "+str(plot_max)+" steps displayed")
plt.legend(fontsize=7,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

plt.tight_layout()
plt.show()


