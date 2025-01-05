import random
import math
from itertools import product
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from .history import History
from .env import Environment, EnvironmentElement
from .constants import *
from .agent_movement import *
from .reward import Reward
from .qlearning import DeepQLearning
from .losses import L1_Loss, L2_Loss
from .model import Baseline, Dueling, QNetwork
from .losses import Loss

# Engine for running the learning algorithm
class Engine:
    def __init__(self, env, reward, l_algos,
                 history, eval_history, losses, iters=500000, 
                 eval_interval = 100):
        """
        Args:
            env (Environment) : Environment in which Agent is located
            reward : Reward function
            agent : Agent object
            history : History object to access the past episodes
            eval_history: History object to access details about evaluation runs
            losses: list of Loss objects which calculate loss of an episode
            iters: number of iterations to train the model
            eval_interval: interval between performing evaluations
        """
        
        assert 0 < eval_interval < iters, \
            f"eval_interval should be between 0 and number of iterations: {iters}"
        assert isinstance(losses, list), f"Losses must be a list"
        assert all([l in Loss.__subclasses__() for l in losses]), "One of the given losses is invalid"
        
        # Learning params
        self.count = 0
        
        # Engine params
        self.iters = iters
        self.eval_interval = eval_interval
        
        # Objects
        self.env = env
        self.reward = reward
        self.l_algos = l_algos
        self.history = history
        self.eval_history = eval_history
        self.losses = list(losses)
        self.training_loss = [[] for _ in range(len(self.l_algos))]

        # Evaluation metrics
        self.best_eval_loss = math.inf
        self.best_eval_ep = None
        self.best_eval_l_algos = [None]* len(self.l_algos)
    
    def __iter__(self):
        """
        Return an iterable object where each iteration is 1 episode
        """
        self.count = 0
        self.training_loss = [[] for _ in range(len(self.l_algos))]
        return self
    
    def __next__(self):
        """
        Represent 1 iteration, i.e. 1 episode in the training
        """
        if self.count < self.iters:
            self.count += 1
            
            # training
            self.train()
            train_reward = self.__run_episode()
            train_reward = sum(train_reward) / len(train_reward)
            
            # Batch update for sampling/gradient method
            for idx, learning in enumerate(self.l_algos):
                if not learning.eval_mode:
                    loss = learning.batch_update(self.history)
                    self.training_loss[idx].append(loss)
                
                    # Update the behaviour policy accordingly
                    learning.policy.update(self.history, self.iters)
            
            # Compute training loss
            loss_types, losses = zip(*((loss_func.__name__, 
                        loss_func.evaluate_single(self.history))
                        for loss_func in self.losses))
            self.history.add_metrics(self.count, loss_types, losses, train_reward)
            
            # Compute evaluation loss
            eval_reward = loss_types = losses = None
            if self.count % self.eval_interval == 0:
                eval_reward, loss_types, losses = self.__run_eval()
                eval_reward = sum(eval_reward) / len(eval_reward)
            return train_reward, eval_reward, loss_types, losses
        else:
            raise StopIteration
        
    def train(self):
        """
        Sets the engine and its associated learning algorithm and environment to training mode. 
        """
        for learning in self.l_algos: learning.train()
        self.env.history = self.history

    def eval(self):
        """
        Sets the engine and its associated learning algorithm and environment to evaluation mode.
        """
        for learning in self.l_algos: learning.eval()
        self.env.history = self.eval_history

    def __run_episode(self, **kwargs):
        """
        Runs an episode, where the agent interacts with the environment until a terminal state is reached 
        or a predefined condition is met, such as exceeding the maximum allowed steps.
        
        Args:
            **kwargs: Arguments that might be required by the environment's reset function.
        
        Returns:
            total_reward: Total accumulated reward during the episode.
        """
        current_states = self.env.reset(**kwargs)
        total_rewards = [0] * self.env.num_agents
        step_counter = 0
        
        print("New episode")
        while True:
            step_counter += 1
            
            # Check if step count exceeds 2*grid size
            if step_counter > 2 * np.prod(self.env.grid_size):
                for learning in self.l_algos:
                    if not learning.eval_mode:
                        agent_hists = [agent_hist for agent_hist in self.history.latest_agent_histories.values() if agent_hist['agent_type'] == learning.agent_type]
                        for agent_hist in agent_hists: learning.penalty(agent_hist)
                    else:
                        total_rewards = [reward + MAX_EPISODE_EXCEED_PENALTY for reward in total_rewards]
                break
            
            # Check if all the items are delivered to B
            done = 0
            for agent_hist in self.env.history.latest_agent_histories.values():
                if len(agent_hist['target_reached']) == 2 \
                    and agent_hist["agent_type"] == EnvironmentElement.AGENT_TYPE_2:
                    done +=1
            
            if done == 2:
                print("Finish episode!")
                break
            # Run the episode as usual
            else:
                current_states, current_rewards = self.step(current_states)
                total_rewards = [x + y for (x, y) in zip(total_rewards, current_rewards)]

        return total_rewards

    def __run_eval(self):
        """
        Executes the agent's evaluation on all possible grid positions. 
        
        Return:
            eval_reward: The average reward the agent achieves across all grid positions.
            loss_types: Types of losses that are evaluated.
            losses: The average values for each of the loss types computed across all grid positions. 
        """
        # Get all grid positions
        w, h = self.env.grid_size
        grid_pos = list(product(range(w), range(h)))
        grid_pos = list(product(*(grid_pos for _ in range(self.env.num_agents))))
        # filter out duplicates
        grid_pos = list(filter(lambda x: len(set(x)) == len(x), grid_pos))
        # random sampling without replacement
        grid_pos_sample = random.sample(grid_pos, min(MAX_EVAL_POS_SAMPLES, len(grid_pos)))
        
        # Switch to evaluation mode
        self.eval()
        total_rewards = [0] * self.env.num_agents
        loss_averages = dict()
        idx = 0
        
        for loc_agents in grid_pos_sample:
            idx += 1
            loc_agents = list(map(list, loc_agents))
            
            if idx % 50 == 0 and self.verbose > 1:
                print(f"Evaluating: {idx}/{len(grid_pos_sample)}")
            
            # runs episode with randomly sampled agent positions
            eval_reward = self.__run_episode(loc_agents=loc_agents)
            total_rewards = [x + y for (x, y) in zip(total_rewards, eval_reward)]
            
            # This is for recording history per episode in evaluation
            loss_types, losses = zip(*[(loss_func.__name__, 
                        loss_func.evaluate_single(self.eval_history))
                        for loss_func in self.losses])
            self.eval_history.add_metrics(self.count, loss_types, losses, eval_reward)
            
            # Calculate the average loss values
            for ind, loss_type in enumerate(loss_types):
                loss_val = loss_averages.setdefault(loss_type, [])
                loss_val.append(losses[ind])

        # Calculate the average reward and average loss values across all valid grid positions
        n_pos = len(grid_pos_sample)
        eval_rewards = [total_reward / n_pos for total_reward in total_rewards]
        loss_types = tuple(loss_averages.keys())
        losses= tuple(tuple(sum((map(lambda l: l if l is not None else MAX_EPISODE_EXCEED_LOSS[key], x))) / n_pos \
                                for x in zip(*values)) for key, values in loss_averages.items())
        
        try:
            avg_loss = np.average(losses)
        except:
            print(losses)
            raise Exception()
        
        assert isinstance(avg_loss, float), f"expected average loss to be a float instead got: {avg_loss}"
        
        if avg_loss < self.best_eval_loss:
            saved_idx = 0
            for idx, l_algo in enumerate(self.l_algos):
                if l_algo in self.best_eval_l_algos:
                    continue
                print(f"Found new best model with average loss: {avg_loss:.5f} < previous lowest eval loss: {self.best_eval_loss:.5f}")
                self.best_eval_l_algos[saved_idx] = deepcopy(l_algo)
                saved_idx += 1
                
            self.best_eval_loss = avg_loss
            self.best_eval_ep = self.count
        
        return eval_rewards, loss_types, losses

    def run(self, verbose=1):
        """
        Start the engine to run the training

        Args:
            verbose : 0 for no output, 
                      1 for evaluation results, 
                      2 for training + evaluation progress + evaluation results
        """
        self.verbose = verbose
        for ep, (train_r, eval_r, loss_types, losses) in enumerate(self, 1):
            if verbose > 1:
                print(f"Episode {ep} has total training rewards of {train_r}")
            if eval_r is not None and verbose > 0:
                print(f"=====EVALUATION AT EPISODE {ep}: REWARD={eval_r}=====")
                for loss_type, loss_val in zip(loss_types, losses):
                    print(f"====EVALUATION AT EPISODE {ep}: LOSS={loss_type}, VALUE={loss_val}=====")
        
        # save best models
        for idx, l_algo in enumerate(self.best_eval_l_algos):
            print(f"saving best l_algo {idx} with the lowest eval loss of: {self.best_eval_loss}")
            fname_prefix=f"saved_models/model_{idx}_ep{self.best_eval_ep}_"
            l_algo.save(fname_prefix=fname_prefix)
        
        return self.best_eval_l_algos


    def step(self, states):
        """
        A single step update after agent makes a movement
        
        Args:
            state : The current state that the agent is in
        
        Return:
            next_state : The subsequent state after agent making the movement
            reward : The reward obtained by making the movement
        """
        rewards = [0] * self.env.num_agents
        next_states = []
        
        # All agents make move sequentially
        for (agent, state) in zip(self.env.agents, states):
            # Agent make its movement
            agent.move()
            next_state = agent.state
            
            # calculate reward
            reward = self.reward.get_reward(agent.agent_index)
            agent.history.latest_agent_history(agent.agent_index)["reward"].append(reward)
            
            # Incremental update for tabular method
            if not agent.l_algo.eval_mode:
                movement = agent.history.latest_agent_history(agent.agent_index)['move'][-1]
                agent.l_algo.incremental_update((state, movement), reward, next_state)
                
            # Collect next_state and rewards
            next_states.append(next_state)
            rewards[agent.agent_index] = reward
            
        return next_states, rewards
    
    def get_training_loss(self):
        """
        Return the training loss.
        
        Return:
            self.next_state : The training loss
        """
        return self.training_loss
    
def init_agent_learning(update_interval, agent_type, **kwargs):
    """
    Helper function to help initialize the learning algorithm for specified agent type
    
    Args:
        update_interval (int): Interval for update the weights in target network
        agent_type (Enum): Type of the agent
        **kwargs: Additional keyword arguments for initializing the network
                    - mode: Agent target mode
                    - tau: Hyperparameter for soft update in target network
                    - lr: learning rate of the network
                    - gamma: discount factor of Q-learning
                    - double: Use of double DQN
    
    Returns:
        learning: DeepQLearning Object acting as the behaviour policy of agents
    """
    if kwargs['mode'] == "absolute_positions":
        if kwargs.get('dueling', None):
            model = Dueling(5, [8, 6, 4])
        else:
            model = Baseline(5, [8, 6, 4])
    elif kwargs['mode'] == "single_relative_target":
        if kwargs.get('dueling', None):
            model = Dueling(2)
        else:
            model = Baseline(2)
    network = QNetwork(model, tau=kwargs['tau'])
    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    criterion = nn.SmoothL1Loss()
    learning = DeepQLearning(network, optimizer, criterion, update_interval=update_interval, agent_type=agent_type, gamma=kwargs['gamma'], double=kwargs['double'])
    return learning

def init_training(**kwargs):
    """
    Helper function for initializing all the components to run a training
    
    Args:
        **kwargs: Keyword arguments for initializing all components
                    - grid_size: Size of the grids
                    - loc_a: Location of point A
                    - loc_b: Location of point B
                    - update_intervals (List): Update intervals of DQN for all type of agents
                    - Other keyword arguments passed into init_agent_learning
    
    Returns:
        env: Environment object for running the simulation
        eval_history: History object storing the results of evaluation
        engine: Engine for running the simulation
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    train_history = History()
    eval_history = History()
    env = Environment(kwargs['grid_size'], train_history, kwargs['loc_a'], kwargs['loc_b'])
    update_int_1, update_int_2 = kwargs['update_intervals']
    learning_1 = init_agent_learning(update_int_1, EnvironmentElement.AGENT_TYPE_1, **kwargs)
    learning_2 = init_agent_learning(update_int_2, EnvironmentElement.AGENT_TYPE_2, **kwargs)
    TypeOneAgent(env, learning_1, mode=kwargs['mode'])
    TypeOneAgent(env, learning_1, mode=kwargs['mode'])
    TypeTwoAgent(env, learning_2, mode=kwargs['mode'])
    TypeTwoAgent(env, learning_2, mode=kwargs['mode'])
    engine = Engine(env, Reward(env), [learning_1, learning_2], train_history, eval_history, losses=[L1_Loss, L2_Loss], iters=3000, eval_interval=200)
    return env, train_history, eval_history, engine