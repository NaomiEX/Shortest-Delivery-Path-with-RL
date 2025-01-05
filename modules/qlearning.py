from abc import ABC, abstractmethod
from datetime import datetime
import os
import os.path as osp
import pickle
import torch
import random

from .state_action_table import Table
from .policies import Policy, EpsilonGreedy
from .constants import MAX_EPISODE_EXCEED_PENALTY_TRAIN
from .agent_movement import Movement

# Template for learning algorithm
class Learning(ABC):
    def __init__(self, lr, gamma, agent_type=None):
        """
        Abstract base class for reinforcement learning algorithms.
        
        Args:
            lr: Learning rate for the algorithm. Must be between 0 and 1.
            gamma: Discount factor for future rewards.
        """
        self.__lr = lr
        self.__gamma = gamma
        # A flag indicating if the agent is in evaluation mode.
        self.__eval_mode = False
        # The policy used in evaluation mode.
        self.__target_policy = EpsilonGreedy(0, None, self, 0, 0)
        # Agent type
        self.__agent_type = agent_type
    
    @property
    def agent_type(self):
        """
        Return agent type (Type I or Type II)
        """
        return self.__agent_type
    
    @property
    def lr(self):
        """Returns the learning rate."""
        return self.__lr
    
    @lr.setter
    def lr(self, val):
        """Sets the learning rate, ensuring it is less than 1."""
        assert val <= 1, "Learning rate should be smaller than 1"
        self.__lr = val
    
    @property
    def gamma(self):
        """Returns the discount factor."""
        return self.__gamma

    @property
    def eval_mode(self):
        """Returns the evaluation mode status."""
        return self.__eval_mode
    
    @property
    def target_policy(self):
        """Returns the target policy used during evaluation."""
        return self.__target_policy
    
    def eval(self):
        """Switches to evaluation mode."""
        self.__eval_mode = True
    
    def train(self):
        """Switches to training mode."""
        self.__eval_mode = False
    
    @abstractmethod
    def get_action(self, state, actions):
        """
        Abstract method to retrieve the action for a given state.

        Args:
            state: Represents the current state of the environment.
            actions: A list of possible actions that can be taken from the given state.
        """        
        pass
    
    @abstractmethod
    def incremental_update(self, curr_key, reward, next_key):
        """
        Abstract method to update learning values based on rewards and states.

        Args:
            curr_key: The key representing the current state-action pair.
            reward: The immediate reward received after taking the action.
            next_key: The key representing the next state-action pair after taking the action.
            
        """        
        pass
    
    @abstractmethod
    def batch_update(self, history):
        """
        Abstract method to update learning values based on all states
        and rewards in 1 episode.
        
        Args:
            history: History object containing all the episodes
        """
        pass
    
    @abstractmethod
    def penalty(self, agent_history):
        """
        Abstract method to add more penalty on all state-action
        pair when the agent fails to finish the episode at target
        point.
        """
        pass

class QLearning(Learning):
    def __init__(self, lr=0.1, gamma=0.9, agent_type=None):
        """
        Implements the Q-Learning algorithm using state-action value tables.
        
        Args:
            lr: The learning rate for implementing the Q-learning algorithm.
            gamma: The discount factor.
        """
        super().__init__(lr, gamma, agent_type)
        
        self.__table = Table()
        self.__policy = None
    
    @property
    def table(self):
        """Returns the state-action value table."""
        return self.__table
    
    @property
    def policy(self):
        """Returns the active policy. Returns target_policy if in evaluation mode."""
        return self.__policy if self.eval_mode is False else self.target_policy
    
    @policy.setter
    def policy(self, value):
        """Sets the policy for the agent. Can be set only once."""
        assert isinstance(value, Policy)
        assert self.__policy is None, "Can only update policy once"
        self.__policy = value
    
    # Update based on q-learning equation
    def incremental_update(self, curr_key, reward, next_state):
        """
        Updates the Q-value for a given state-action based on the Q-learning equation.

        Args:
            curr_key: The key representing the current state-action pair.
            reward: The immediate reward received after taking the action.
            next_state: Represents the state that follows the current state after taking the action.
        """
        table = self.table
        table[curr_key] = (1 - self.lr) * table[curr_key].q_value + \
            self.lr * (reward + self.gamma * table.max_value(next_state))
    
    def batch_update(self, history):
        return super().batch_update(history)
    
    # Get action from the policy
    def get_action(self, state, actions):
        """
        Retrieve the action for a given state using the active policy.

        Args:
            state: Represents the current state of the environment.
            actions: A list of possible actions that can be taken from the given state.
            
        Returns:
            action: The selected action based on the policy.
        """       
        assert self.policy is not None, \
            "Policy cannot be None"
        return self.policy.get_action(state, actions)
    
    def penalty(self, agent_history):
        """
        Apply penalty when the agent could not reach the destination within
        steps limit.
        
        Args:
            agent_history: experience of agent in whole episode
        """
        states, moves = agent_history['state'], agent_history['move']
        moves = [Movement.name_to_cls(move) for move in moves]
        # Distribute penalty across all the actions performed in the episode
        penalty = MAX_EPISODE_EXCEED_PENALTY_TRAIN / len(moves)
        
        # Update state-action value based on the penalty
        for t in range(len(states)-2):
            next_state = states[t+1]
            self.incremental_update((states[t], moves[t]), penalty, next_state)
    
    def store_table(self, folder_path=None):
        """Store state-action-value tables

        Args:
            folder_path (str, optional): path to folder where the Q table is stored. 
                Defaults to None in which case a folder "saved/" is created (if doesn't yet exist)
                within which another folder using current date and time is created to store the q_table.
        """
        curr_datetime = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        if folder_path is None:
            folder_path = f"saved/{curr_datetime}"
        try:
            os.makedirs(folder_path)
        except FileExistsError:
            parent_dir = "/".join(folder_path.split("/")[:-1])
            folder_path = f"{parent_dir.strip('/')}/{curr_datetime}"
            os.makedirs(folder_path)

        name = "Q_Table"

        # serialize as pkl files for memory efficiency
        with open(f"{folder_path}/{name}.pkl", "wb") as f:
            pickle.dump(self.table, f)

    def load_table(self, folder_path):
        """
        Loads a stored state-action-value table.
        
        Args:
            folder_path: Path where the Q-table is stored.
        """
        if not os.path.isdir(folder_path):
            raise ValueError(f"given path: {folder_path} is not a valid directory")
        
        # Deserialize the table
        name = "Q_Table"
        with open(f"{folder_path.strip('/')}/{name}.pkl","rb") as f:
            self.__table = pickle.load(f)
        
        # Default to eval_mode
        self.eval()
        
class DeepQLearning(Learning):
    """
    Learning algorithm for Deep Q-Network using stochastic
    gradient descent
    """
    def __init__(self, dqn, optimizer, criterion, batch_size=32, 
                 gamma=0.9, update_interval=20, transform=None,
                 double=False, agent_type=None):
        super().__init__(0, gamma, agent_type)
        
        # Initialize the learning requirements
        self.__network = dqn
        self.__opt = optimizer
        self.__crit = criterion
        self.__batch_size = batch_size
        self.__policy = EpsilonGreedy(1, learning=self, agent_type=self.agent_type)
        self.__update_interval = update_interval
        self.__trans = transform
        self.__step = 0
        self.__hist = []
        self.__double = double
    
    @property
    def table(self):
        """Return the state-action table wrapped as a network model"""
        return self.__network
    
    @property
    def policy(self):
        """Return the policy based on mode of learning"""
        return self.__policy if self.eval_mode is False else self.target_policy
    
    @policy.setter
    def policy(self, value):
        """Sets the policy for the agent. Can be set only once."""
        assert isinstance(value, Policy)
        self.__policy = value
    
    def __batch_sample(self, hist):
        """
        Sampling from the experience based on the batch size
        to construct a batch of (states, actions, rewards, states')
        for update network weights. Additionally, not_done mask is added
        for modifications on the state representation and correctness
        of the learning algorithm.
        
        Args:
            hist: List of episodes where each episode is represented as dictionary
        Return:
            batch of states, actions, rewards and states' from the samples
        """
        state_pairs, rewards, actions, not_done_mask = [], [], [], []
        
        # Get (state, state'), reward, action and not_done_mask
        for h in hist:
            states = h['state']
            state_pairs.extend([(states[_], states[_+1]) for _ in range(len(states)-1)])
            rewards.extend(h['reward'])
            actions.extend(h['move'])
            
            not_done_mask.extend([1] * len(h['move']))
            
            if len(h['target_reached']) == 2:
                not_done_mask[-1] = 0
        
        # Sample from the experience
        samples = random.sample(
            list(zip(state_pairs, rewards, actions, not_done_mask)),
            k=self.__batch_size
        )
        
        # Group into batches
        states, actions, rewards, next_states, not_done_mask = [], [], [], [], []
        for sample in samples:
            states.append(self.__transform(sample[0][0]))
            next_states.append(self.__transform(sample[0][1]))
            rewards.append(sample[1])
            actions.append(Movement.name_to_idx(sample[2]))
            not_done_mask.append(sample[3])
        
        return torch.cat(states), torch.tensor(actions, dtype=torch.int64), \
            torch.tensor(rewards, dtype=torch.float), torch.cat(next_states), \
                torch.tensor(not_done_mask, dtype=torch.bool)
    
    def __transform(self, x):
        """
        Transformation done on the input state if specified.
        Else, turn the state into tensor
        
        Args:
            x (tensor): Input tensor of states
        """
        if self.__trans is None:
            x = torch.tensor(x, dtype=torch.float)
            if x.dim() == 1: x = x.unsqueeze(0)
        else:
            for func in self.__trans: x = func(x)
        return x
    
    def penalty(self, agent_history):
        """
        Apply penalty to agent when it cannot reach the destination
        within step limits.
        
        Args:
            agent_history: A dictionary object used to store the agent-relevant information in that episode
        """
        rewards = agent_history['reward']
        penalty = MAX_EPISODE_EXCEED_PENALTY_TRAIN / len(rewards)
        # Distribute the penalty across all the actions taken in this episode
        agent_history['reward'] = [r + penalty for r in rewards]
    
    def get_action(self, state, actions):
        """
        Return the action given the state and list of allowable actions
        based on the policy and q-values from the network
        
        Args:
            state (tensor): Tensor of state
            actions (list): List of allowable positions
        """
        return self.policy.get_action(self.__transform(state), actions)
    
    def incremental_update(self, curr_key, reward, next_key):
        return super().incremental_update(curr_key, reward, next_key)
    
    def batch_update(self, history):
        """
        Update the network using stochastic gradient descent on
        batches of sampled experience
        
        Args:
            history: History object
        """
        self.__step += 1
        self.__opt.zero_grad()
        
        # Training based on agent type
        self.__hist.extend([agent for agent in history.latest_agent_histories.values() if agent['agent_type'] == self.agent_type])
        
        # Check if experience exceed batch_size
        if sum(map(lambda x: len(x['state']) - 1, self.__hist)) >= self.__batch_size:
            # Make prediction
            states, actions, rewards, next_states, not_done_mask = self.__batch_sample(self.__hist)
            pred = self.table(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            # Calculate target using either normal or double Q-learning
            if self.__double:
                target_hat = self.table(next_states).detach()
                _, a_prime = target_hat.max(1)
                
                target = self.table(next_states, True).gather(1, a_prime.unsqueeze(1)).squeeze()
                target = target * self.gamma * not_done_mask + rewards
            else:
                target = self.table(next_states, True).max(1).values * self.gamma  * not_done_mask + rewards
            
            # Update the weights by backward propagation
            loss = self.__crit(pred, target)
            loss.backward()
            self.__opt.step()
            
            # Update the weights of target network at fixed interval
            self.__hist = []
            if self.__step == self.__update_interval:
                self.__step = 0
                self.table.update_target()
        
            return loss.item()
        
    def save(self, fname=None, fname_prefix=""):
        directory = osp.dirname(fname_prefix)
        if (len(directory) > 0):
            os.makedirs(directory, exist_ok=True)
        if fname is None:
            fname = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        fname = fname_prefix + fname
            
        self.__network.save(fname, fname_prefix="")