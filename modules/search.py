import numpy as np

from itertools import product
import pickle

from .viz import Visualization
from .engine import init_training
from .constants import *



def get_best_model(hyperparam_res):
    """
    Function for obtaining the top-5 best models in terms of convergence rate
    and the minimum loss values
    
    Args:
        hyperparam_res: List of dictionary storing the results of each model tested in hyperparameter testing
    
    Return:
        List of top-5 selected models
    """
    models = []
    
    for r in hyperparam_res:
        avg_losses_per_ep = Visualization.get_avg_losses_per_ep(r['eval_history'])
        
        # average between a and b losses
        combined_ab_loss = [np.average(d['loss'][0]) for d in avg_losses_per_ep]
        convergence_idx = np.argmin(combined_ab_loss)
        min_loss = combined_ab_loss[convergence_idx]
        
        models.append((r, min_loss, convergence_idx))

    # Select the top 5 models based on minimum loss and convergence rate
    return sorted(models, key=lambda x: (x[1], x[2]))[:5]

def get_best_model_per_policy(hyperparam_res, policies):
    """Gets model with the best hyperparameters for each policy.

    Args:
        hyperparam_res (list of dict): list of dictionaries where each dict contains the hyperparameters and evaluation results
        policies (list of Policy): list of policy classes

    Returns:
        dict: keys are policy names and values are each policy's best hyperparameters and evaluation results
    """
    best_per_policy = dict()
    for policy_cls in policies:
        # gets all hyperparameter test results for that policy
        result_pol = [r for r in hyperparam_res if r['hyperparams']['policy'] == policy_cls]

        best_idx = None
        best_convergence_idx = None
        best_loss = float("inf")

        for idx, r in enumerate(result_pol): # iterate through all hyperparameter combos for this policy
            # get the average loss per evaluation episode 
            avg_losses_per_ep = Visualization.get_avg_losses_per_ep(r['eval_history'])
            # average between a and b losses
            combined_ab_loss = [np.average(d['loss'][0]) for d in avg_losses_per_ep]
            convergence_idx = np.argmin(combined_ab_loss)
            min_loss = combined_ab_loss[convergence_idx]
            # set as best if it has lower loss or converges faster
            if min_loss < best_loss or (min_loss == best_loss and convergence_idx < best_convergence_idx):
                best_idx = idx
                best_loss = min_loss
                best_convergence_idx = convergence_idx
        best_per_policy[policy_cls.__name__] = result_pol[best_idx]
    return best_per_policy

def hyperparam_search(save_to=None, load_from=None, **kwargs):
    """Train models on several hyperparameters to obtain the best model
    
    Args:
        save_to (str, optional): file name to save hyperparameter search results. Defaults to None.
        load_from (str, optional): file name to load hyperparameter search results. Defaults to None.
    """
    # Set of hyperparameters to be tested
    learning_rates = [0.01, 0.001]
    gammas = [0.1, 0.9]
    taus = [0, 0.5, 0.9]
    doubles = [True, False]
    update_intervals = [(150, 250), (175, 325), (200, 375)]

    results = []

    # load hyperparam results from .pkl file
    if load_from is not None:
        try:
            with open(load_from, "rb") as f:
                while True:
                    try:
                        results.append(pickle.load(f))
                    except EOFError:
                        break
        except:
            print("Cannot load from given file, proceeding to perform hyperparameter testing")

    if len(results) == 0:
        # Initialize common parameters
        mode = "absolute_positions"
        loc_a = [0, 0]
        loc_b = [x-1 for x in GRID_SIZE]
        
        # train model for every lr, gamma, policy combination
        for lr, gamma, tau, double, update_interval in product(learning_rates, gammas, taus, doubles, update_intervals):
            print(f"testing with lr: {lr}, gamma: {gamma}, tau: {tau}, double: {double}, update_interval: {update_interval}")
            
            *_, eval_history, engine = init_training(loc_a=loc_a, loc_b=loc_b, mode=mode, lr=lr, gamma=gamma, tau=tau, double=double, update_intervals=update_interval, grid_size=GRID_SIZE)
            engine.run(verbose=1)
            result = dict(hyperparams=dict(lr=lr, gamma=gamma, tau=tau, double=double, update_interval=update_interval), eval_history=eval_history)
            results.append(result)

        # if provided, save results to .pkl file
        if save_to is not None:
            if save_to.split(".")[-1] != "pkl":
                save_to = f"{save_to}.pkl"
            with open(f"{save_to}", "wb") as f:
                for res in results:
                    pickle.dump(res, f)
    
    # Only select the top-5 models
    best_models = get_best_model(results)
    
    # process to the format expected for visualization
    best_per_policy_hist = {f"Top {idx+1}" : val["eval_history"] for idx, (val, _, _) in enumerate(best_models)}
    
    # plot losses
    Visualization.plot_losses(best_per_policy_hist, train=False, 
                              title="Evaluation Losses for Models with Best Hyperparameters per Policy", 
                              **kwargs)
