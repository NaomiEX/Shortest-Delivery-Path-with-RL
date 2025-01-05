import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from itertools import product
from mpl_toolkits.axes_grid1 import AxesGrid

from modules.constants import *
from .agent_movement import *
from .utils import softmax
from .env import EnvironmentElement

class Visualization:
    """
    Class containing all visualization methods and 
    helper methods for processing data to prepare for visualizing.
    """

    @staticmethod
    def plot_policy(env, learning, history, episode=-1, agent_state_mode="absolute_positions",
                    agent_type=None):
        """
        Plot the policy as the direction of the action giving the highest value in the given position
        on the grids
        
        Args:
            env: Environment object
            learning: Learning object representing the agent policy
            history: History object for each episode
            episode: Selected episode from history
            agent_state_mode: State representation of agent in grid world
            agent_type: Type of agent
        """
        def possible_movements(pos):
            surrounding_cells = env.get_cells_around_pos(pos)
            possible_movements = []
            for movement in Movement.__subclasses__():
                if movement.is_possible(pos, surrounding_cells):
                    possible_movements.append(movement)
            return possible_movements
        
        # Get point A and point B
        a_loc = history.get_env_history(episode)["loc_a"]
        b_loc = env.loc_b
        
        # Initialize plots
        grid_size = env.grid_size
        _, ax = plt.subplots(nrows=1, ncols=2)
        
        # Initialize variables based on state_mode
        if agent_state_mode == "absolute_positions":
            target_1, target_1_label, has_item_1 = (a_loc, "A", 0) if agent_type == EnvironmentElement.AGENT_TYPE_1 \
                                                    else (b_loc, "B", 1)
            target_2 = list(np.random.randint(0, grid_size[1], size=(2)))
            has_item_2 = 1 if agent_type == EnvironmentElement.AGENT_TYPE_1 else 0
            target_2_label = str(target_2)
        else:
            target_1, target_1_label = (a_loc, "A")
            target_2, target_2_label = (b_loc, "B")
        
        ax[0].imshow(np.zeros(grid_size), cmap=plt.get_cmap("binary"))
        ax[0].set_title(f"Table 1 (Target: {target_1_label})")
        ax[0].text(target_1[0], target_1[1], 'o',
                   horizontalalignment='center',
                   verticalalignment='center')
        
        ax[1].imshow(np.zeros(grid_size), cmap=plt.get_cmap("binary"))
        ax[1].set_title(f"Table 2 (Target: {target_2_label})")
        ax[1].text(target_2[0], target_2[1], 'o',
                   horizontalalignment='center',
                   verticalalignment='center')
        

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                actions = possible_movements((x, y))
                
                # Target 1
                if agent_state_mode == "absolute_positions":
                    state = [x, y, target_1[0], target_1[1], has_item_1]
                else:
                    state = [a - b for a, b in zip((x, y), target_1)]

                # if state != [0, 0]:
                plot_cond = [x,y] != target_1 if agent_state_mode == "absolute_positions" else state != [0,0]
                if plot_cond:
                    best_action = learning.table.max_action(state, actions)
                    ax[0].text(x, y, best_action.direction(),
                                horizontalalignment='center',
                                verticalalignment='center')
                
                # Table B
                if agent_state_mode == "absolute_positions":
                    state = [x, y, target_2[0], target_2[1], has_item_2]
                else:
                    state = [a - b for a, b in zip((x, y), target_2)]
                plot_cond = [x,y] != target_2 if agent_state_mode == "absolute_positions" else state != [0,0]
                if plot_cond:
                    best_action = learning.table.max_action(state, actions)
                    ax[1].text(x, y, best_action.direction(),
                               horizontalalignment='center',
                               verticalalignment='center')
        
        plt.suptitle(f"Policy{' for agent ' + str(agent_type.value)}")
        plt.show()

    @staticmethod
    def plot_agent_path(agent_history, env_history):
        """Plot agent path in an interactive HTML animation window

        Args:
            agent_history (dict): agent history containing details about agent configuration like position, state, etc.
            env_history (dict): environment history containing details about env configuration like occupancy, location of A
        """
        plt.ioff()
        MARKERSIZE=10

        grid_w, grid_h = env_history['occupancy'][0][0].shape
        offsets = np.linspace(-0.3, 0.3, num=4)
        
        # centers position to appear on the center of a grid instead of on grid lines
        def center_grid_pos(pos, offset=0):
            x,y=pos
            return [x + 0.5 + offset, grid_h - y - 0.5 + offset]
        
        # Collect relevant information about agent
        agent_pos = {agent_idx: val['pos'] for agent_idx, val in agent_history.items()}
        agent_info = {agent_idx: {"agent_type":val['agent_type'], "target_reached":val['target_reached']} \
            for agent_idx, val in agent_history.items() }
        repeated_agent_pos = {agent_idx: [agent_pos[agent_idx][0]] \
            for agent_idx in agent_pos.keys()}
        max_move_num = max([len(v) for v in agent_pos.values()])
        done = []
        
        # repeat positions of agent when it's another agent's turn, 
        # simulates the fact that agents perform their moves one by one,
        # thus when it's another agent's turn other agents just stay still.
        for i in range(1, max_move_num):
            for agent_idx in range(4):
                if i >= len(agent_pos[agent_idx]) :
                        continue
                agent_type, target_reached = agent_info[agent_idx].values()
                next_pos = agent_pos[agent_idx][i]
                repeated_agent_pos[agent_idx].append(next_pos)

                if (agent_type == EnvironmentElement.AGENT_TYPE_1 and len(target_reached)==2 and target_reached[-1] == i):
                    other_agent = agent_history[agent_idx]['other_agent']
                    if (repeated_agent_pos[other_agent][-1] == next_pos):
                        done.append(agent_idx)
                elif (agent_type == EnvironmentElement.AGENT_TYPE_2 and len(target_reached) and target_reached[0]==i):
                    other_agent = agent_history[agent_idx]['other_agent']
                    if (repeated_agent_pos[other_agent][-1] == next_pos):
                        done.append(other_agent)
                
                for repeat_agent_idx in range(4):
                    if repeat_agent_idx == agent_idx or \
                        i >= len(agent_pos[repeat_agent_idx]) or repeat_agent_idx in done:
                        continue
                    repeated_agent_pos[repeat_agent_idx].append(repeated_agent_pos[repeat_agent_idx][-1])
        
        # places each agent on different offsets so they do not overlap
        agent_pos = {agent_idx: [center_grid_pos(p, offset=offsets[agent_idx]) for p in val] for agent_idx, val in repeated_agent_pos.items()}
        
        # add positions of items
        item_pos = dict()

        for agent_idx in agent_pos.keys():
            agent_type = agent_info[agent_idx]['agent_type']
            if agent_type == EnvironmentElement.AGENT_TYPE_2:
                continue
            picked_up_item = False
            item_pos_agent = []
            for (agent_x, agent_y) in agent_pos[agent_idx]:
                if picked_up_item == False:
                    if (0.0 <= agent_x < 1.0 and grid_h-1 <= agent_y < grid_h):
                        picked_up_item = True

                if picked_up_item == True:
                    item_pos_agent.append([agent_x + 0.08, agent_y+0.05])
                else:
                    item_pos_agent.append([None, None])

            item_pos[agent_idx] = item_pos_agent

        # check when the item is passed between type I and II agents
        for agent_idx in item_pos.keys():
            other_agent = agent_history[agent_idx]['other_agent']
            if (other_agent is None):
                continue
            start = len(item_pos[agent_idx])
            for (other_x, other_y) in agent_pos[other_agent][start:]:
                item_pos[agent_idx].append([other_x + 0.08, other_y + 0.08])
        
        # get agent x and y positions throughout the episode
        for agent_idx in agent_pos.keys():
            agent_idx_pos_x, agent_idx_pos_y = zip(*agent_pos[agent_idx])
            agent_pos[agent_idx] = dict(agent_pos_x = list(agent_idx_pos_x),
                                        agent_pos_y = list(agent_idx_pos_y))
            
        for item_idx in item_pos.keys():
            item_idx_pos_x, item_idx_pos_y = zip(*item_pos[item_idx])
            item_pos[item_idx] = dict(item_pos_x = list(item_idx_pos_x),
                                        item_pos_y = list(item_idx_pos_y))
            
        max_move_num = max([len(v['agent_pos_x']) for v in agent_pos.values()])
        loc_a = center_grid_pos(env_history['loc_a'])
        loc_b = env_history['loc_b']
        loc_b = center_grid_pos(loc_b)
        
        fig = plt.figure(clear=True)
        ax = fig.add_subplot()
        ax.set_xlim((0, grid_w))
        ax.set_ylim((0, grid_h))
        ax.set_xticks(range(grid_w))
        ax.set_yticks(range(grid_h))
        ax.grid()
        
        paths = []
        heads = []
        items=  []
        colors = ["b", "g", "r", "c"]
        for i in range(len(agent_pos)):
            paths += ax.plot([], [], f"{colors[i]}-", lw=2)
            heads += ax.plot([], [], f"{colors[i]}", marker=f"${ '2' if (i > 1) else '1'}$", 
                            markersize=MARKERSIZE)
            
        for i in range(len(item_pos)):
            items += ax.plot([], [], f"{colors[i]}", marker="*", 
                            markersize=5)
        # since location of A doesn't change can directly plot 
        loc_a, = ax.plot(loc_a[0], loc_a[1], marker="$A$", markersize=MARKERSIZE)
        # since location of B doesn't change can directly plot
        loc_b, = ax.plot(loc_b[0], loc_b[1], marker="$B$", markersize=MARKERSIZE)

        # plot figure for frame i showing agent path up to position i
        def animate(i):
            for agent_idx in range(len(agent_pos)):
                agent_pos_x, agent_pos_y = agent_pos[agent_idx]['agent_pos_x'], agent_pos[agent_idx]['agent_pos_y']
                x = agent_pos_x[:i]
                y = agent_pos_y[:i]
                paths[agent_idx].set_data(x, y)
                point_i = max(i-1, 0)
                # plot a circle to represent current agent position
                heads[agent_idx].set_data(agent_pos_x[point_i:point_i+1], agent_pos_y[point_i:point_i+1])

                # plot a * to represent an item held by an agent
                if (agent_idx in item_pos):
                    if (point_i < len(item_pos[agent_idx]['item_pos_x']) and item_pos[agent_idx]['item_pos_x'][point_i] is not None):
                        try:
                            items[agent_idx].set_data(item_pos[agent_idx]['item_pos_x'][point_i:point_i+1],
                                                item_pos[agent_idx]['item_pos_y'][point_i:point_i+1])
                        except:
                            print(point_i)
                    else:
                        items[agent_idx].set_data([], [])
            if len(items) > 0:
                return *paths, *heads, *items
            else:
                return *paths, *heads
        
        # create animation
        anim = animation.FuncAnimation(fig, animate,
                                    frames=max_move_num+1, interval=100, 
                                    blit=True)
        
        # display as HTML to be interactive
        display(HTML(anim.to_jshtml()))
        # clean up
        plt.close()
        plt.ion()


    @staticmethod
    def get_random_episode(history, metric_hist, episode, flip=False):
        """Get a random episode from *evaluation* history. 
        Used to select a run to plot agent path

        Args:
            history (History): history object containing eval agent, env, metrics histories
            metric_hist (list of dict): metrics for eval
            episode (int): episode number
            flip (bool, optional): whether to use negative rewards or not. Defaults to False.
        """
        ep_idx_offset = None
        ep_metric_hist = []
        ep_rewards= []
        # gets all the evaluation metrics for a particular episode
        # gets all the rewards for a particular episode
        for idx, d in enumerate(metric_hist):
            if d["episode"] != episode:
                continue
            if ep_idx_offset is None:
                ep_idx_offset = idx
            ep_metric_hist.append(d)
            ep_rewards.append(d['reward'])
            
        ep_rewards = np.array(ep_rewards)
        if (ep_rewards.ndim == 2):
            ep_rewards = np.average(ep_rewards, axis=-1)

        assert ep_rewards.ndim == 1, f"expected number of dims of ep_rewards to be 1 but instead: {ep_rewards.ndim}"

        if flip:
            # clip to set reward lower bounds to 0
            ep_rewards = np.clip(ep_rewards, 0, None)
        else:
            ep_rewards = np.abs(ep_rewards)

        # get probability of choosing each run
        probs = softmax(ep_rewards)
        # randomly choose a run
        random_ep = ep_idx_offset + np.random.choice(a=range(len(probs)), p=probs)
        random_agent_hist = history.get_agent_history(episode=random_ep)
        random_env_hist = history.get_env_history(episode=random_ep)
        return random_agent_hist, random_env_hist
    
    @staticmethod
    def plot_last_path_eval(history):
        """Plots a random run from the last evaluation episode

        Args:
            history (History): history object containing eval agent, env, metrics histories
        """
        metric_hist = history.get_metric_history()
        last_ep = metric_hist[-1]['episode']
        # get agent and env history from a random run from the last episode
        l_random_agent_hist, l_random_env_hist = Visualization.get_random_episode(history, metric_hist, last_ep, flip=True)
        display(HTML(f"<h2>Random Agent Path at Episode {last_ep} (following greedy policy)</h2>"))
        # Plot agent path on the selected run
        Visualization.plot_agent_path(l_random_agent_hist, l_random_env_hist)
        

    @staticmethod
    def plot_first_and_custom_path_eval(history, best=False):
        """Plot random run from first and either the last or the best evaluation episode

        Args:
            history (History): history object containing eval agent, env, metrics histories
            best (bool, Optional): True to plot a random path from the best eval episode, False otherwise. Defaults to False.
        """
        metric_hist = history.get_metric_history()

        first_ep = metric_hist[0]['episode']
        # get agent and env history from a random run from the first episode
        f_random_agent_hist, f_random_env_hist = Visualization.get_random_episode(history, metric_hist, first_ep)
        display(HTML("<h2>Random Agent Path at Episode 1 (following greedy policy)</h2>"))
        # Plot agent path on the selected run
        Visualization.plot_agent_path(f_random_agent_hist, f_random_env_hist)

        if best:
            avg_losses_per_ep = Visualization.get_avg_losses_per_ep(history)
            avg_losses = [np.average(e['loss']) for e in avg_losses_per_ep]
            best_ep = avg_losses_per_ep[np.argmin(avg_losses)]['episode']
            b_random_agent_hist, b_random_env_hist = Visualization.get_random_episode(history, metric_hist, best_ep)
            display(HTML(f"<h2>Random Agent Path at Episode {best_ep} (following greedy policy)</h2>"))
            Visualization.plot_agent_path(b_random_agent_hist, b_random_env_hist)

        else:
            # Plot agent path for a random run on the last episode
            Visualization.plot_last_path_eval(history)


    @staticmethod
    def get_avg_losses_per_ep(hist):
        """Helper method to get average losses over the multiple runs for each EVAL episode

        Args:
            hist (History): history object containing eval agent, env, metrics histories

        Returns:
            list of dict: average losses per evaluation episode
        """
        eps_nums = set([d['episode'] for d in hist.get_metric_history()])
        # get all metrics for each run for each episode
        ep_history = {episode: [d for d in hist.get_metric_history() if d['episode']==episode] for episode in eps_nums}
        ep_avg = []

        for ep, ep_hist in ep_history.items():
            avg_losses = []
            loss_types = ep_hist[0]['loss_type']
            for idx, loss_name in enumerate(loss_types):

                # get average a and b losses for the episode
                avg_losses.append(tuple(sum((map(lambda l: l if l is not None else MAX_EPISODE_EXCEED_LOSS[loss_name], x))) / len(x) \
                    for x in zip(*[d["loss"][idx] for d in ep_hist])))
            ep_avg.append(dict(episode=ep, loss_type=loss_types, loss=avg_losses))

        # sort in ascending order of episode
        ep_avg.sort(key=lambda d: d['episode'])
        return ep_avg

    @staticmethod
    def plot_losses(history, batch_size = 100, train=True, title=None, ymax=None):
        """
        Plot the losses for history of episodes. 
        
        Args:
            hist (History): history object containing eval agent, env, metrics histories
            batch_size (int, Optional): Number of episodes in a batch. Defaults to 100
            train (boolean, Optional): True for plotting training losses, False for plotting evaluation loss. Defaults to True
            title (str, Optional): Main title for the graph plotted. Defaults to None
            ymax (List[int], Optional): list of custom values to set as the maximum value in the y-axis. Defaults to None
        """
        # set up figure to plot L2 and L1 loss
        fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(12, 12), linewidth=0.1)
        if title is None:
            title = f"{'Training' if train else 'Evaluation'} Losses" + \
                    (f"(over batches of size {batch_size})" if train else "")
        fig.suptitle(title)

        for ax in axs:
            ax.remove()

        # 2 axes as we have 2 losses, L1 and L2
        row_axs = [None]*2 

        # add subfigure per subplot
        gridspec = axs[0].get_subplotspec().get_gridspec()
        subfigs = [fig.add_subfigure(gs) for gs in gridspec]

        for row, subfig in enumerate(subfigs):
            subfig.suptitle(f'L{1+row} Loss')

            # create 1x2 subplots per subfig
            axs = subfig.subplots(nrows=1, ncols=2)
            row_axs[row] = axs

            for col, ax in enumerate(axs):
                ax.plot()
                ax.set_title(f'Path to {chr(65+col)}')

        if train:
            for model, hist in history.items():
                # NOTE: ASSUMES THAT ALL MODELS ARE TRAINED WITH SAME NUMBER OF EPISODES/ITERATIONS
                metric_hist_train = hist.get_metric_history()
                loss_types = metric_hist_train[0]['loss_type']
                
                for idx, loss_name in enumerate(loss_types):
                    # get a and b losses for loss_name
                    loss_a, loss_1, loss_2, loss_b = [
                        list(map(lambda l: l if l is not None else MAX_EPISODE_EXCEED_LOSS[loss_name], x))\
                        for x in zip(*[d["loss"][idx] for d in metric_hist_train])]

                    # get average a and b losses over batch_size batches
                    avg_loss_a = np.average(np.array(loss_a).reshape(-1, batch_size), axis=1)
                    avg_loss_1 = np.average(np.array(loss_1).reshape(-1, batch_size), axis=1)
                    avg_loss_2 = np.average(np.array(loss_2).reshape(-1, batch_size), axis=1)
                    avg_loss_b = np.average(np.array(loss_b).reshape(-1, batch_size), axis=1)
                    
                    # x axis values
                    batch_nums = [x+1 for x in range(len(avg_loss_a))]

                    # plot loss graphs
                    for colnum, (avg_loss1, avg_loss2) in zip([0,1], [(avg_loss_a, avg_loss_1), (avg_loss_2, avg_loss_b)]):
                        labels = [f"{model} (agent type I to A)", f"{model} (agent type I to handover location)"] if colnum == 0\
                            else [f"{model} (agent type II to handover location)", f"{model} (agent type II to B)"] 

                        row_axs[idx][colnum].plot(batch_nums, avg_loss1, marker="o", label=labels[0])
                        row_axs[idx][colnum].plot(batch_nums, avg_loss2, marker="o", label=labels[1])
                        row_axs[idx][colnum].set_xticks(batch_nums)
        else: # plot losses for eval
            for model, hist in history.items():
                # gets the average losses over runs per episode
                hist_avg = Visualization.get_avg_losses_per_ep(hist)
                eps = [d['episode'] for d in hist_avg]
                
                loss_types = hist_avg[0]['loss_type']

                for idx, loss_name in enumerate(loss_types):
                    # get a and b losses for that episode
                    loss_a, loss_1, loss_2, loss_b = zip(*[d["loss"][idx] for d in hist_avg])
                    
                    # plot loss graphs
                    for colnum, (l1, l2) in zip([0,1], [(loss_a, loss_1), (loss_2, loss_b)]):
                        labels = [f"{model} (agent type I to A)", f"{model} (agent type I to handover location)"] if colnum == 0\
                            else [f"{model} (agent type II to handover location)", f"{model} (agent type II to B)"] 

                        row_axs[idx][colnum].plot(eps, l1, marker="o", label=labels[0])
                        row_axs[idx][colnum].plot(eps, l2, marker="o", label=labels[1])

        # sets subplot titles and x,y labels
        for idx, loss_name in enumerate(loss_types):
            for j in range(2):
                ax =row_axs[idx][j] 
                ymax_g = ymax[idx] if ymax is not None else (120 if idx == 0 else 12000)
                ax.set_ylim([0, ymax_g])
                # Shrink current axis by 20%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper right', fontsize='7')
                
                ax.title.set_text(f"{'Average ' if train else ''}{loss_name} for Agent Type {'I' if j == 0 else 'II'}")
                ax.set_xlabel("Episode" if not train else "Batch Number")
                ax.set_ylabel(f'Loss')

    @staticmethod
    def plot_training_loss(engine):
        """
        Plot the training loss of the agents
        """
        training_losses = engine.get_training_loss()
        ys = [[x for x in loss if x != None] for loss in training_losses]
        fig, ax = plt.subplots(nrows=1, ncols=len(ys))
        
        fig.set_size_inches(10, 5)
        for idx, y in enumerate(ys):
            ax[idx].plot([i+1 for i in range(len(y))], y)
            ax[idx].title.set_text(f"Training Loss for Agent Type {'I' if idx==0 else 'II'}")
            ax[idx].set_xlabel('Batch #')
            ax[idx].set_ylabel('Loss')
            
        plt.show()

    @staticmethod
    def plot_q_values(learning, target=None, has_item=None, proj="3D", mode="absolute_positions"):
        """
        Plot the summation of Q-values of the actions (from adjacent cells) that 
        leads to cells in a heat map.
        
        Args:
            learning (Learning): The learning algorithm used by the agent
            target (List[int]): target location
            has_item (boolean): whether agent has an item
            proj (string): Projection to create 2D or 3D heatmap
            mode (string): state representation of agent
        """
        mode = mode.lower()
        table = learning.table
        x, y = GRID_SIZE
        
        if mode == "absolute_positions":
            grid_plot = np.zeros((x, y))
            
            # Generate all possible combinations of states
            if target is not None:
                target_x = [target[0]]
                target_y = [target[1]]
            else:
                target_x, target_y = range(x), range(y)
            
            if has_item is None:
                has_item_ = range(2)
            elif has_item:
                has_item_ = [1]
            else:
                has_item_ = [0]
            
            states = filter(lambda x: x[0] != x[2] or x[1] != x[3], product(range(x), range(y), target_x, target_y, has_item_))
            
            # Check the boundary of the states
            left_bound, right_bound, up_bound, down_bound = -1, y, -1, x
            
            # Prepare for 2D or 3D heatmap
            if proj == "3D":
                xs, ys = np.meshgrid(np.arange(x), np.arange(y))
            elif proj == '2D':
                xs, ys = np.arange(x), np.arange(y)
                label_x, label_y = xs, ys
            else:
                raise ValueError(f"No such projection, {proj}, supported")
        elif mode == "single_relative_target":
            grid_plot = np.zeros((2*x - 1, 2*y - 1))
            
            # Generate all possible combinations of states
            states = product(range(-x+1, x), range(-y+1, y))
            
            # Get the boundaries of states
            left_bound, right_bound, up_bound, down_bound = -y, y, -x, x
            
            # Preparation for projection in 2D or 3D heatmap
            if proj == "3D":
                xs, ys = np.meshgrid(np.arange(2*x - 1) - x, np.arange(2*y - 1) - y)
            elif proj == "2D":
                xs, ys = np.arange(2*x - 1), np.arange(2*y - 1)
                label_x, label_y = np.arange(-x+1, x), np.arange(-y+1, y)
            else:
                raise ValueError(f"No such projection, {proj}, supported")
        else:
            raise ValueError(f"No such mode, {mode}, supported")

        for state in states:
            # Get the q-values from the DQN
            q_values = softmax(table(state).squeeze().data.numpy())
            grid_x, grid_y, *_ = state
            
            if mode == "single_relative_target":
                grid_x += x - 1
                grid_y += y - 1
            
            # Calculate the values for neighbouring cells/states
            if grid_x - 1 > left_bound:
                grid_plot[grid_y, grid_x - 1] += q_values[Movement.cls_to_idx(MoveLeft)]
            
            if grid_x + 1 < right_bound:
                grid_plot[grid_y, grid_x + 1] += q_values[Movement.cls_to_idx(MoveRight)]
            
            if grid_y - 1 > up_bound:
                grid_plot[grid_y - 1, grid_x] += q_values[Movement.cls_to_idx(MoveUp)]
            
            if grid_y + 1 < down_bound:
                grid_plot[grid_y + 1, grid_x] += q_values[Movement.cls_to_idx(MoveDown)]

        for grid_plot_x in range(right_bound):
            for grid_plot_y in range(down_bound):
                nneighbors = int(grid_plot_x - 1 > left_bound and [grid_plot_x-1, grid_plot_y] != target) + \
                    int(grid_plot_x + 1 < right_bound and [grid_plot_x+1, grid_plot_y] != target) + \
                    int(grid_plot_y - 1 > up_bound and [grid_plot_x, grid_plot_y-1] != target) + \
                        int(grid_plot_y + 1 < down_bound and [grid_plot_x, grid_plot_y+1] != target)

                grid_plot[grid_plot_y, grid_plot_x] /= nneighbors

        
        # Plot 3D heatmap
        if proj == "3D":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xs, ys, grid_plot, cmap='viridis')
            
            ax.set_xlabel('Row')
            ax.set_ylabel('Column')
            fig.colorbar(surf, ax=ax, shrink = 0.5, aspect=10, label='Q-Value Summation')
            plt.title( "3-D Heat Map: Summation of Q-Values of Actions which Leads to The Position" )
        # Plot 2D heatmap
        elif proj == '2D':
            plt.imshow( grid_plot )
            plt.xticks(xs, label_x)
            plt.yticks(ys, label_y)
            plt.xlabel('Row')
            plt.ylabel('Column')
            plt.title( "2-D Heat Map: Summation of Q-Values of Actions which Leads to the Position" )
            plt.colorbar(label='Q-Values Summation')
        
        plt.show()
        
    @staticmethod
    def plot_move_qvals_per_state(network, env, agent_mode="absolute_positions", 
                                  target=None, has_item=False, agent_type=EnvironmentElement.AGENT_TYPE_1):
        """
        Plots heatmap of Q-values for each move for each possible state

        Args:
            network (QNetwork): trained QNetwork object to obtain predictions from
            env (Environment): environment in which the QNetwork was trained in
        """
        max_x = env.grid_size[0]-1
        max_y = env.grid_size[1]-1

        if agent_mode == "absolute_positions":
            range_x = range(0, max_x + 1)
            range_y = range(0, max_y + 1)
        elif agent_mode == "single_relative_target":
            range_x = range(-max_x, max_x + 1)
            range_y = range(-max_y, max_y+1)
        
        # table to store q-values for all states for all movements
        movement_q = [np.zeros([len(range_y), len(range_x)]) for _ in range(len(Movement.__subclasses__()))]

        if target is None:
            target = list(np.random.randint(0, 4, size=(2)))
            while target in [[0,0], [max_x, max_y]]:
                target = list(np.random.randint(0, 4, size=(2)))
        else:
            assert isinstance(target, (list, tuple)) and len(target) == 2
            target = list(target)
        # run (x,y) to model and obtain predicted Q-values for all movements
        for x,y in product(range_x, range_y):
            with torch.no_grad():
                if agent_mode == "absolute_positions":
                    inp = torch.FloatTensor([[x, y, target[0], target[1], int(has_item)]])
                elif agent_mode == "single_relative_target":
                    inp = torch.FloatTensor([[x, y]])
                model_res = network(inp, target=False).squeeze().numpy()
            # since model predictions are in range (-inf, inf), softmax is applied to normalize values to [0,1]
            
            preds = softmax(model_res)
            for i in range(len(preds)):
                offset_x, offset_y = (max_x, max_y) if agent_mode == "single_relative_target" else (0,0)
                grid_x, grid_y = x + offset_x, y + offset_y
                movement_q[i][grid_y,grid_x] = preds[i]

        # set up figure
        fig = plt.figure(figsize=(10,10))
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(2, 2),
                        axes_pad=(0.5, 0.5),
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        )
        num_moves = len(Movement.__subclasses__())
        max_grid_x = max_x if agent_mode == "absolute_positions" else max_x*2
        max_grid_y = max_y if agent_mode == "absolute_positions" else max_y*2
        norm = plt.Normalize(np.min(movement_q), np.max(movement_q))
        
        # set up subplots with axes, labels, heatmap, etc.
        for idx, ax in zip(range(num_moves), grid):
            move_name = Movement.idx_to_cls(idx).__name__
            im = ax.imshow(movement_q[idx], norm=norm)
            ax.set_title(f'Move {move_name.strip("Move")}')
            ax.set_xticks(range(0, max_grid_x+1))
            ax.set_yticks(range(0, max_grid_y+1))
            ax.set_xticklabels(list(range_x))
            ax.set_yticklabels(list(range_y))
            ax.set_xticks(np.arange(.5, max_grid_x+1, 1), minor=True)
            ax.set_yticks(np.arange(.5, max_grid_y+1, 1), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
            ax.set_xlabel(f'{"" if agent_mode == "absolute_positions" else "state"} x')
            ax.set_ylabel(f'{"" if agent_mode == "absolute_positions" else "state"} y')
            ax.label_outer()
            
        # use one colorbar because everything is on the same scale
        grid.cbar_axes[0].colorbar(im)
        fig.suptitle(f"Predicted Q-values (normalized) for{' agent ' + str(agent_type.value) if agent_mode == 'absolute_positions' else ''} all movements in all positions (target=[{target[0]},{target[1]}])", fontsize=16, y=.9)
        plt.show()
