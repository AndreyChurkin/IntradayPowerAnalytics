"""
A simple market game where at each time step there are two random prices at the market: 1) Ask price and 2) Bid price.
All prices are randomly generated in the range of [0:10].
The agent wants to earn profit through market arbitrage, which means buying at a lower price and then selling at a higher price. 
The agent has 100 steps per episode to perform such an arbitrage and get profit. The profit is equal to the prices of all sales minus the prices of purchases. 
At each time step, the agent has three available actions {do nothing, buy, sell}.
A reinforcement learning model is developed to train the agent to perform effective arbitrage.
The agent's performance is then backtested and compared against the optimal trading decisions (found via mathematical optimisation) and a heuristic trading policy.

- This 'visualise_policy' script visualises the RL agent's policy as optimal actions in the Bid-Ask price space

Note: this version of the code allows impossible actions, e.g., selling out of the empty inventory, but penalises them.

Andrey Churkin https://andreychurkin.ru/

"""

from more_realistic_market_game_2_env import RandomPriceMarketEnv
from stable_baselines3 import PPO, DQN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import font_manager

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)



# # Instantiate the environment
env = RandomPriceMarketEnv()

# # Load the trained RL model:
# model = PPO.load("ppo_market_game_2", env)
# model = PPO.load("ppo_market_game_2_v2", env) # no penalty
# model = PPO.load("ppo_market_game_2_v3", env) # penalty -1
# model = PPO.load("ppo_market_game_2_v4", env) # penalty -10
# model = PPO.load("ppo_market_game_2_v5", env) # penalty -10, buying at 'Ask price'
# model = PPO.load("ppo_market_game_2_v6", env) # penalty -10, buying at 'Ask price', all random prices
model = PPO.load("ppo_market_game_2_v7", env) # penalty -10, Ask price can be much higher





""" 
Select the policy visualisation method: a scatter plot or a heatmap (via imshow function) 
Note that 'scatter' plot is less efficient for large numbers of points. It is also less beautiful and consistent than 'imshow'
"""
# visualisation_method = "scatter"
visualisation_method = "imshow"



"""
Define the number of subplots corresponding to different values of the inventory
E.g., [0, 2] = plot two policies (for the empty inventory and the inventory of 2)
The dimensions of the np.array() will define the number of subplots and their location
"""
# inventory_subplots = np.array([0, 1])
# inventory_subplots = np.array([0, 1, 2])
# inventory_subplots = np.array([[0, 1], [2, 3]])
inventory_subplots = np.array([[0, 1, 2], [3, 4, 5]])
# inventory_subplots = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


if inventory_subplots.ndim == 1:
    vis_columns = inventory_subplots.shape[0]
    vis_rows = 1
elif inventory_subplots.ndim > 1:
    vis_columns = inventory_subplots.shape[1]
    vis_rows = inventory_subplots.shape[0]

inventory_subplots_flatten = inventory_subplots.flatten() # a vector of all inventory parameters to iterate over



""" Set the number of grid steps per axis to visualise the policy in 2D """
policy_vis_grid_steps = 200



# # ---------- Get agent's output for different inputs to visualise the policy ----------

vis_data_ask_prices_all_subplots = []
vis_data_bid_prices_all_subplots = []
vis_data_actions_all_subplots = []
vis_data_actions_meshgrid_all_subplots = [] # Z values for plotting a meshed grid (array of arrays)

for inventory_subplot in inventory_subplots_flatten: # iterate over inventory values (subplots)
    vis_data_ask_prices = []
    vis_data_bid_prices = []
    vis_data_actions = []
    vis_data_actions_meshgrid = [] # Z values for plotting a meshed grid (array of arrays)

    for grid_step_y in np.linspace(0, 10, policy_vis_grid_steps): # iterate over Ask price
        actions_array = [] # to collect actions over one loop

        for grid_step_x in np.linspace(0, 10, policy_vis_grid_steps): # iterate over Bid price
            state = [grid_step_y, grid_step_x, inventory_subplot] # [ask price, bid price, inventory]
            
            # Get the action from the trained agent
            action, _states = model.predict(state, deterministic=True)

            vis_data_bid_prices.append(grid_step_x)
            vis_data_ask_prices.append(grid_step_y)
            vis_data_actions.append(action.item())
            actions_array.append(action.item())
        vis_data_actions_meshgrid.append(actions_array) # collect arrays of action arrays

    vis_data_bid_prices = np.array(vis_data_bid_prices)
    vis_data_ask_prices = np.array(vis_data_ask_prices)
    vis_data_actions = np.array(vis_data_actions)

    vis_data_ask_prices_all_subplots.append([vis_data_ask_prices])
    vis_data_bid_prices_all_subplots.append([vis_data_bid_prices])
    vis_data_actions_all_subplots.append([vis_data_actions])
    vis_data_actions_meshgrid_all_subplots.append(vis_data_actions_meshgrid)
        
print(f"\n-----------------------------------------------------------------")
print(f"The the policy has been evaluated for {len(inventory_subplots_flatten)} values of inventory, {policy_vis_grid_steps**2} points per subplot")
print(f"Visualising the policy as '{visualisation_method}' plot")
print(f"-----------------------------------------------------------------")



# # Define colors and labels for discrete action values [0, 1, 2]:

# colors = ["gray", "green", "red"]
# colors = ["lightgray", "lightskyblue", "lightcoral"]
# colors = ["#E0E0E0", "#AEEEEE", "#FF8C69"]
# colors = ["lightgray", "lightskyblue", "#fab387"]
# colors = ["lightgray", "#80F0F0", "lightcoral"]
# colors = ["#B0B0B0", "lightskyblue", "lightcoral"]
colors = ["#B0B0B0", "#99c2ff", "#ff9999"]

actions_cmap = mcolors.ListedColormap(colors)

# Normalize the actions to ensure colors are assigned correctly
norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

labels = {0: 'Do nothing', 1: 'Buy', 2: 'Sell'}


# # Select the font family:
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'Courier New'
# plt.rcParams['font.family'] = 'Arial'


""" Note that scatter() plot is less efficient for large numbers of points. It is also less beautiful and consistent than imshow() """
if visualisation_method == "scatter":
    
    fig, axes = plt.subplots(vis_rows, vis_columns, figsize=(vis_columns*4*0.9, vis_rows*4)) # figure size (width, height)

    # Loop through the subplots
    subplot_count = 0
    for i in range(vis_rows):  # Row index
        for j in range(vis_columns):  # Column index
            subplot_count += 1

            ax = axes[i,j]  # Access each subplot

            ax.scatter(vis_data_bid_prices_all_subplots[subplot_count-1],
                       vis_data_ask_prices_all_subplots[subplot_count-1],
                       c = vis_data_actions_meshgrid_all_subplots[subplot_count-1], 
                    cmap=actions_cmap,
                    s = 3, # <--- Set a reasonable scatter marker size here 
                    alpha = 1,
                    edgecolor = 'none',
                    marker = "s",
                    norm = norm
                   )
            
            # Add a diagonal line where Ask price = Bid price
            min_price = min(min(vis_data_bid_prices), min(vis_data_ask_prices))
            max_price = max(max(vis_data_bid_prices), max(vis_data_ask_prices))
            ax.plot([min_price, max_price], [min_price, max_price], 
                    'k--', alpha=0.5)
            
            ax.set_xlabel('Bid price')
            ax.set_ylabel('Ask price')
            ax.set_title(f"Inventory = {inventory_subplots_flatten[subplot_count-1]}",fontweight='bold')

            ax.set_xlim(0, 10) # <--- price plotting limits [0:10], adjust if using different price range
            ax.set_ylim(0, 10)

    plt.gca().set_aspect('equal')
    plt.subplots_adjust(wspace=0.4, hspace=0.9)

    # Create legend markers
    patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}")
        for i in range(3)
    ]
    
    fig.legend(handles=patches, loc="upper center", ncol=3,
                # fontsize=14, 
                # bbox_to_anchor=(0.5, 1.05),
                title="Actions:",
                title_fontproperties = font_manager.FontProperties(weight='bold')
                )

    # plt.tight_layout()  # Fix spacing
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # left/bottom/right/top extensions of the plot

    # Save as PDF, PNG, or SVG
    plt.savefig("plt_policy_visualisation.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("plt_policy_visualisation.png", format="png", bbox_inches="tight")
    # plt.savefig("plt_policy_visualisation.svg", format="svg", bbox_inches="tight")

    plt.show()

elif visualisation_method == "imshow":
    fig, axes = plt.subplots(vis_rows, vis_columns, figsize=(vis_columns*4*0.9, vis_rows*4)) # figure size (width, height)

    # Loop through the subplots
    subplot_count = 0
    for i in range(vis_rows):  # Row index
        for j in range(vis_columns):  # Column index
            subplot_count += 1

            ax = axes[i,j]  # Access each subplot

            ax.imshow(vis_data_actions_meshgrid_all_subplots[subplot_count-1], 
                   cmap=actions_cmap, 
                   origin='lower', 
                   extent=[0, 10, 0, 10], # <--- price plotting limit labels [0:10], adjust if using different price range
                   norm = norm
                   )
            
            # Add a diagonal line where Ask price = Bid price
            min_price = min(min(vis_data_bid_prices), min(vis_data_ask_prices))
            max_price = max(max(vis_data_bid_prices), max(vis_data_ask_prices))
            ax.plot([min_price, max_price], [min_price, max_price], 
                    'k--', alpha=0.5)
            
            ax.set_xlabel('Bid price')
            ax.set_ylabel('Ask price')
            ax.set_title(f"Inventory = {inventory_subplots_flatten[subplot_count-1]}",fontweight='bold')
    
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(wspace=0.4, hspace=0.9)

    # Create legend markers
    patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}")
        for i in range(3)
    ]
    
    fig.legend(handles=patches, loc="upper center", ncol=3,
                # fontsize=14, 
                # bbox_to_anchor=(0.5, 1.05),
                title="Actions:",
                title_fontproperties = font_manager.FontProperties(weight='bold')
                )

    # plt.tight_layout()  # Fix spacing
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # left/bottom/right/top extensions of the plot

    # Save as PDF, PNG, or SVG
    plt.savefig("plt_policy_visualisation.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("plt_policy_visualisation.png", format="png", bbox_inches="tight")
    # plt.savefig("plt_policy_visualisation.svg", format="svg", bbox_inches="tight")

    plt.show()

else:
    print("Warning: please select the correct visualisation method")

