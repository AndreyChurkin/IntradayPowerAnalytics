"""
A simple market game where at each time step there are two random prices at the market: 1) Ask price and 2) Bid price.
All prices are randomly generated in the range of [0:10].
The agent wants to earn profit through market arbitrage, which means buying at a lower price and then selling at a higher price. 
The agent has 100 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.
A reinforcement learning model is developed to train the agent to perform effective arbitrage.
The agent's performance is then backtested and compared against the optimal trading decisions (found via mathematical optimisation) and a heuristic trading policy.

- this 'performance_evaluation' script analyses the RL agent's performance across multiple runs and compares it with the optimal trading solutions (Gurobi)

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



""" Set the number of runs to evaluate the agent's performance """
N_runs = 100

all_cumulative_agent_rewards = []
all_cumulative_optimal_rewards = []

# # Run the agent and the optimisation model in a loop:
for run in range(N_runs):
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

    # Check the results
    if opt_model.status == GRB.OPTIMAL:
        x_buy_solution = [x_buy[t].X for t in range(opti_steps_max)]
        x_sell_solution = [x_sell[t].X for t in range(opti_steps_max)]
        x_inventory_solution = [x_inventory[t].X for t in range(opti_steps_max)]

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
    
    # Write down cumulative rewards for this run:
    all_cumulative_agent_rewards.append(cumulative_rewards_out_of_env)
    all_cumulative_optimal_rewards.append(cumulative_optimal_rewards)



# # Calculate the minimum, maximum, and average rewards per step:
plot_agent_rewards_max = []
plot_agent_rewards_min = []
plot_agent_rewards_average = []
plot_optimal_rewards_max = []
plot_optimal_rewards_min = []
plot_optimal_rewards_average = []

for t in range(test_steps):
    agent_t_values = [arr[t] for arr in all_cumulative_agent_rewards]
    plot_agent_rewards_max.append(max(agent_t_values))
    plot_agent_rewards_min.append(min(agent_t_values))
    plot_agent_rewards_average.append(sum(agent_t_values)/len(agent_t_values))

    optimal_t_values = [arr[t] for arr in all_cumulative_optimal_rewards]
    plot_optimal_rewards_max.append(max(optimal_t_values))
    plot_optimal_rewards_min.append(min(optimal_t_values))
    plot_optimal_rewards_average.append(sum(optimal_t_values)/len(optimal_t_values)) 



# # ---------- Plotting the analysis: ----------

# plot_max = 100 # number of steps (trades) to visualise
plot_max = test_steps

plt.rcParams['font.family'] = 'Courier New'

plt.figure(figsize=(10, 5))

time_steps = np.arange(0, test_steps)

plt.fill_between(time_steps, plot_optimal_rewards_min, plot_optimal_rewards_max, color="red", alpha=0.3)
plt.fill_between(time_steps, plot_agent_rewards_min, plot_agent_rewards_max, color="green", alpha=0.3)
plt.plot(time_steps, plot_optimal_rewards_average, color="red", label="Optimal trading (Gurobi)")
plt.plot(time_steps, plot_agent_rewards_average, color="green", label="RL agent")

plt.title(f'Performance evaluation across {N_runs} runs')
plt.xlabel('Step')
plt.ylabel('Сumulative reward')
plt.legend(fontsize=8, loc='upper left')
plt.grid(True, color='gray', alpha=0.2)  # Major grid
plt.minorticks_on()  # Enable minor ticks
plt.grid(True, which='minor', linestyle='-', color='gray', alpha=0.05)  # Minor grid

plt.tight_layout()

plt.savefig("plt_performance_evaluation.pdf", format="pdf", bbox_inches="tight")
plt.savefig("plt_performance_evaluation.png", format="png", bbox_inches="tight")

plt.show()

