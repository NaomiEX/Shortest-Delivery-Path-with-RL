from .env import EnvironmentElement
from .constants import *
from .viz import Visualization
from .engine import init_training

import numpy as np
import random
import torch
import pickle
from IPython.display import HTML, display

def test_grids(save_to=None, load_from=None, lr=0.01, verbose=1, **kwargs):
    """Test the model on several grid sizes

    Args:
        save_to (str, optional): file name to save grid test results. Defaults to None.
        load_from (str, optional): file name to load grid test results. Defaults to None.
        lr (float, optional): learning rate. Defaults to 0.1.
        gamma (float, optional): discount factor. Defaults to 0.9.
        policy_cls (Policy, optional): Behaviour policy to follow. Defaults to EpsilonGreedy.

    Returns:
        list of dict: each dict stores the test results for each grid size
    """
    grid_sizes = [(3,3), (5,5), (6,6)]
    update_intervals = [(100, 150), (175, 325), (225, 350)]
    results_grid = []

    # load hyperparam results from .pkl file
    if load_from is not None:
        try:
            with open(load_from, "rb") as f:
                results_grid = pickle.load(f)
        except:
            print("Cannot load from given file, proceeding to perform grid testing")
    
    if len(results_grid) == 0:
        # Set common hyperparameters
        mode = "absolute_positions"
        loc_a = [0, 0]
        gamma = 0.9
        tau = 0.5
        double = True
        
        # train model for every grid size
        for grid_size, update_interval in zip(grid_sizes, update_intervals):
            print(f"testing with grid size: {grid_size}")
            
            # Reset seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            
            # Initialize environments for training
            loc_b = [x-1 for x in grid_size]
            env, *_, eval_history, engine = init_training(loc_a=loc_a, loc_b=loc_b, mode=mode, lr=lr, gamma=gamma, tau=tau, double=double, update_intervals=update_interval, grid_size=grid_size)
            
            # Run the training and record the results
            best_algos = engine.run(verbose=verbose)
            result = dict(grid_size=grid_size, env=env, learning=best_algos, eval_history=eval_history)
            results_grid.append(result)
            
        # if provided, save results to .pkl file
        if save_to is not None:
            if save_to.split(".")[-1] != "pkl":
                save_to = f"{save_to}.pkl"
            with open(f"{save_to}", "wb") as f:
                pickle.dump(results_grid, f)

    # process to the format expected for visualization
    result_for_viz = {d['grid_size']: d['eval_history'] for d in results_grid}
    # plot losses for each grid size
    Visualization.plot_losses(result_for_viz, train=False, **kwargs)
    return results_grid

def test_grids_viz(grid_results):
    """Visualize agent paths from first and last episode 
    and visualize policy for each grid size

    Args:
        grid_results (list of dict): each dict stores the test results for each grid size
    """
    for grid_result in grid_results:
        # Get the policy for Type I and Type II agents
        l1, l2 = grid_result['learning']
        
        # Plot the animation for agents movement in selected episode
        display(HTML(f"<h1>Grid size: {grid_result['grid_size']}</h1>"))
        Visualization.plot_first_and_custom_path_eval(grid_result["eval_history"], best=True)
        
        # Plot the policy of Type I agent
        display(HTML(f"<h2>Action which gives the highest value for each grid position in Agent Type I</h2>"))
        Visualization.plot_policy(grid_result["env"], l1, grid_result["eval_history"], agent_type=EnvironmentElement.AGENT_TYPE_1)
        
        # Plot the policy of Type II agent
        display(HTML(f"<h2>Action which gives the highest value for each grid position in Agent Type II</h2>"))
        Visualization.plot_policy(grid_result["env"], l2, grid_result["eval_history"], agent_type=EnvironmentElement.AGENT_TYPE_2)

