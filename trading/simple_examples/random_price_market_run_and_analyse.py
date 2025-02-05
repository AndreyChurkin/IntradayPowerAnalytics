"""
A simple market game where at each time step there are two random prices at the market: 1) Ask price and 2) Bid price.
All prices are randomly generated in the range of [0:10].
The agent wants to earn profit through market arbitrage, which means buying at a lower price and then selling at a higher price. 
The agent has 100 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.
A reinforcement learning model is developed to train the agent to perform effective arbitrage.
The agent's performance is then backtested and compared against the optimal trading decisions (found via mathematical optimisation) and a heuristic trading policy.

- This 'run_and_analyse' script evaluates and visualises the RL agent's actions

Note: this version of the code allows impossible actions, e.g., selling out of the empty inventory, but penalises them.

Andrey Churkin https://andreychurkin.ru/

"""

from more_realistic_market_game_2_env import RandomPriceMarketEnv
from more_realistic_market_game_2_train_and_save import evaluate_agent
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import numpy as np
import matplotlib.pyplot as plt

from gurobipy import Model, GRB, quicksum

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)



# Instantiate the environment
env = RandomPriceMarketEnv()

# # Load the trained RL model:
# model = PPO.load("ppo_market_game_2", env)
# model = PPO.load("ppo_market_game_2_v2", env)
# model = PPO.load("ppo_market_game_2_v4", env)
# model = PPO.load("ppo_market_game_2_v5", env)
# model = PPO.load("ppo_market_game_2_v6", env)
model = PPO.load("ppo_market_game_2_v7", env)




# # ---------- Test the trained agent, get its actions for further visualisation ----------

# Initialize lists to store variables
bid_prices = []
ask_prices = []
actions = []
rewards = []
inventory = []
rewards_out_of_env = [] # We will manually check the rewards due to the actions out of the environment 

# Run a single test episode
state, info = env.reset()
done = False

test_steps = 0 # Number of steps used at the evaluation stage

while not done:
    test_steps += 1
    # Get the action from the trained agent
    action, _states = model.predict(state, deterministic=True)

    # Take a step in the environment
    next_state, reward, done, truncated, info = env.step(action)

    # Log variables
    ask_prices.append(state[0]) 
    bid_prices.append(state[1]) 
    inventory.append(state[2])
    actions.append(action)   # The agent's chosen action
    rewards.append(reward)   # The reward for this step

    # Calculate the rewards manually for double-checking
    if action == 0:
        rewards_out_of_env.append(0)
    elif action == 1:
        rewards_out_of_env.append(-state[0])
    elif action == 2:
        if state[2] > 0:
            rewards_out_of_env.append(state[1])
        else:
            rewards_out_of_env.append(0)

    # Update state
    state = next_state

cumulative_rewards = np.cumsum(rewards)
cumulative_rewards_out_of_env = np.cumsum(rewards_out_of_env)

print(f"\n-----------------------------------------------------------------")
print(f"The agent's trading policy has been tested for {test_steps} steps")
print(f"Total agent's rewards: {sum(rewards[:test_steps])}")
print(f"Total agent's profit calculated outside the environment: {sum(rewards_out_of_env[:test_steps])}")
print(f"The agent's strategy includes: {actions[:test_steps].count(0)} 'Do nothing' actions, {actions[:test_steps].count(1)} 'Buy' actions, {actions[:test_steps].count(2)} 'Sell' actions")
print(f"Maximum inventory reached during trading: {max(inventory)}")
print(f"Inventory at the end of the episode: {inventory[-1]}")
print(f"-----------------------------------------------------------------")



# # ---------- Create a simple heuristic trading policy as a benchmark: ----------

# h_max_steps = 100
h_max_steps = test_steps
def heuristic_trading_policy(max_steps=h_max_steps):
    """
    A simple heuristic trading policy for the environment:
    - Buy when the ask price < 5.
    - Sell when the bid price > 5 (if inventory > 0).
    - Do nothing otherwise.
    """

    heuristic_rewards = []
    heuristic_actions = []
    h_inventory = 0
    h_inventory_peak = 0

    for h_step in range(max_steps):
        h_ask_price = ask_prices[h_step]
        h_bid_price = bid_prices[h_step]
        
        # Apply the heuristic policy
        if h_bid_price > 5 and h_inventory > 0:  # Sell action
            h_action = 2
            h_inventory -= 1
            h_reward = h_bid_price
        elif h_ask_price < 5:  # Buy action
            h_action = 1
            h_inventory += 1
            h_reward = -h_ask_price
        else:  # Do nothing
            h_action = 0
            h_reward = 0

        # track the maximum inventory peak during trading
        if h_inventory > h_inventory_peak:
            h_inventory_peak = h_inventory

        # Update total profit and record the action
        heuristic_rewards.append(h_reward)
        heuristic_actions.append(h_action)

    return heuristic_rewards, heuristic_actions, h_inventory_peak, h_inventory

heuristic_rewards, heuristic_actions, heuristic_inventory_peak, h_inventory = heuristic_trading_policy()
cumulative_heuristic_rewards = np.cumsum(heuristic_rewards)

print(f"\n-----------------------------------------------------------------")
print(f"The simple heuristic trading policy has been tested for {h_max_steps} steps")
print(f"Total heuristic profit: {sum(heuristic_rewards)}")
print(f"The heuristic strategy includes: {heuristic_actions.count(1)} 'Buy' actions, {heuristic_actions.count(2)} 'Sell' actions")
print(f"Maximum inventory reached during trading: {heuristic_inventory_peak}")
print(f"Inventory at the end of the episode: {h_inventory}")
print(f"-----------------------------------------------------------------")



# # ---------- Finding the optimal trading solution using Gurobi: ----------

# Number of steps to consider in the optimisation problem:
# opti_steps_max = 100
opti_steps_max = test_steps

# Define the initial inventory level:
x_initial_inventory = 0
x_max_inventory = 100  # Maximum allowable inventory

# Create the Gurobi model:
opt_model = Model("TradingGameOptimization")
opt_model.setParam("OutputFlag", 0)

# Decision variables:
x_buy = opt_model.addVars(opti_steps_max, vtype=GRB.BINARY, name="Buy")      # 1 if buying, 0 otherwise
x_sell = opt_model.addVars(opti_steps_max, vtype=GRB.BINARY, name="Sell")    # 1 if selling, 0 otherwise
x_inventory = opt_model.addVars(opti_steps_max, vtype=GRB.INTEGER, name="Inventory", lb=0, ub=x_max_inventory)

# Objective: Maximize total profit
opt_model.setObjective(
    quicksum(x_sell[t] * bid_prices[t] - x_buy[t] * ask_prices[t] for t in range(opti_steps_max)),
    GRB.MAXIMIZE
)

# Constraints
for t in range(opti_steps_max):
    if t == 0:
        # Inventory at the first timestep depends on initial inventory and action taken
        opt_model.addConstr(x_inventory[t] == x_initial_inventory + x_buy[t] - x_sell[t], name=f"Inventory_Constraint_t{t}")
    else:
        # Inventory evolves over time
        opt_model.addConstr(x_inventory[t] == x_inventory[t-1] + x_buy[t] - x_sell[t], name=f"Inventory_Constraint_t{t}")

    # Ensure that the agent cannot buy and sell at the same time
    opt_model.addConstr(x_buy[t] + x_sell[t] <= 1, name=f"Mutually_Exclusive_Actions_t{t}")

# Solve the opt_model
opt_model.optimize()

# Check and print the results
if opt_model.status == GRB.OPTIMAL:
    print(f"\n-----------------------------------------------------------------")
    print(f"The optimal solution for a session with {opti_steps_max} steps is found by Gurobi")
    print(f"The optimal total profit: {opt_model.objVal}")
    x_buy_solution = [x_buy[t].X for t in range(opti_steps_max)]
    x_sell_solution = [x_sell[t].X for t in range(opti_steps_max)]
    x_inventory_solution = [x_inventory[t].X for t in range(opti_steps_max)]
    print(f"The optimal strategy includes: {sum(x_buy_solution)} 'Buy' actions, {sum(x_sell_solution)} 'Sell' actions")
    print(f"Maximum inventory reached during trading: {max(x_inventory_solution)}")
    print(f"Inventory at the end of the episode: {x_inventory_solution[-1]}")
    print(f"-----------------------------------------------------------------")

    optimal_rewards = []
    optimal_actions = []

    # Compute the rewards according to the optimal actions:
    for t in range(opti_steps_max):
        optimal_rewards.append(-x_buy_solution[t]*ask_prices[t] + x_sell_solution[t]*bid_prices[t])
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

plt.rcParams['font.family'] = 'Courier New'

plt.figure(figsize=(12, 8))

# Plot Bid/Ask prices
plt.subplot(4, 1, 1)
plt.plot(bid_prices[:plot_max], label="Bid price")
plt.plot(ask_prices[:plot_max], label="Ask price")
plt.ylabel("Bid/Ask price")
plt.legend(fontsize=8,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

# Plot actions
plt.subplot(4, 1, 2)
plt.plot(optimal_actions[:plot_max], label="Optimal actions (Gurobi)", drawstyle="steps-post",color="red")
plt.plot(heuristic_actions[:plot_max], label="Heuristic policy actions", drawstyle="steps-post",color="pink")
plt.plot(actions[:plot_max], label="Agent actions", drawstyle="steps-post",color=(0.3, 0.3, 0.3))
plt.ylabel("Actions:\nHold(0),Buy(1),Sell(2)")
plt.ylim([-0.5, 2.5])  # Since actions are binary
plt.legend(fontsize=8,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

# Plot rewards
plt.subplot(4, 1, 3)
plt.plot(optimal_rewards[:plot_max], label="Optimal (Gurobi)",color="red")
plt.plot(heuristic_rewards[:plot_max], label="Heuristic policy",color="pink")
# plt.plot(rewards[:plot_max], label="Rewards per step - agent (within the environment)", color="yellowgreen")
plt.plot(rewards_out_of_env[:plot_max], label="Agent", color="green")
plt.ylabel("Reward per step")
plt.legend(fontsize=8,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

# Plot rewards
plt.subplot(4, 1, 4)
plt.plot(cumulative_optimal_rewards[:plot_max], label="Optimal (Gurobi)",color="red")
plt.plot(cumulative_heuristic_rewards[:plot_max], label="Heuristic policy",color="pink")
# plt.plot(cumulative_rewards[:plot_max], label="Cumulative reward - agent (within the environment)", color="yellowgreen")
plt.plot(cumulative_rewards_out_of_env[:plot_max], label="Agent", color="green")
plt.ylabel("Cumulative reward")
plt.xlabel("Evaluation time step")
plt.legend(fontsize=8,loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

plt.tight_layout()

plt.savefig("plt_action_analysis.pdf", format="pdf", bbox_inches="tight")
plt.savefig("plt_action_analysis.png", format="png", bbox_inches="tight")
# plt.savefig("plt_action_analysis.svg", format="svg", bbox_inches="tight")

plt.show()



# # Evaluate the agent for 10 episodes
print("\nEvaluating the agent for 10 episodes:")
evaluate_agent(model, env)
    

