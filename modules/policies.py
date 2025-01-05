from abc import ABC, abstractmethod
import numpy as np

from .agent_movement import Movement
from .state_action_table import Table

# Abstract class for defining behavioural policy
# Note : to access action space, can either pass the action list or get straight from Movement
# Note : For simplicity, actions gotten from Movement class is chosen
class Policy(ABC):
    def __init__(self, agent_type=None):
        """
        Initialize the policy to access the action space
        """
        self.__actions = Movement.__subclasses__()
        self.__agent_type = agent_type
    
    @property
    def agent_type(self):
        """
        Get the agent type
        """
        return self.__agent_type
    
    @property
    def actions(self):
        """
        Get the actions in the action space
        """
        return self.__actions
    
    @abstractmethod
    def update(self, history, max_iters):
        """
        Update the policy based on history
        
        Args:
            history (History) : history object for accessing the past episodes
        """
        pass
    
    @abstractmethod
    def get_action(self, state, actions):
        """
        Get the action based on the policy given the state
        
        Args:
            state : The current state that the agent is in
            actions : Possible movements that the agent can make
        """
        pass
    
    @abstractmethod
    def get_action_prob(self, state, action):
        """
        Get the probability of the action given the state
        
        Args:
            state : The state that the agent is currently in
            action : The action that the agent might make
        """
        pass

# Epsilon Greedy policy -> balance between exploitation and exploration using epsilon
class EpsilonGreedy(Policy):
    def __init__(self, epsilon=1, table=None, learning=None,
                 decay_rate=0.01, min_epsilon=0.1, agent_type=None):
        """
        Args:
            epsilon : Probability that the policy will go into exploration
            table : Q-Table for accessing best action
            learning : Learning algorithm for accessing the Q-Table
            decay_rate : Rate of decay for reducing epsilon (go for exploitation)
            min_epsilon : Minimum rate of exploration needed
        """
        
        assert table is not None or learning is not None, \
            "Policy has access to either learning or q-table"
        
        assert 0 <= epsilon <= 1, \
            "Epsilon should be between 0 and 1"
        
        self.__epsilon = epsilon
        self.__table = table
        self.__learning = learning
        self.__decay_rate = decay_rate
        self.__min_epsilon = min_epsilon
        super().__init__(agent_type)
    
    @property
    def table(self):
        """
        Access the q-table either through the stored table or learning class
        """
        return self.__table if self.__learning is None else self.__learning.table
    
    def update(self, history, max_iter):
        """
        Update the epsilon using weight decaying function, i.e.
        epsilon = epsilon * e^{-lambda * t}
        where t is time step and lambda is decay rate
        
        To allows more exploration, 1 time step is equivalent to 1000 episodes
        
        Args:
            history (History) : History object that keep counts of number of episodes
        """
        fraction = history.num_episodes / max_iter * self.__decay_rate
        self.__epsilon = max(self.__min_epsilon, self.__epsilon - self.__epsilon * fraction)
    
    def get_action(self, state, actions):
        """
        Provide the action using epsilon-greedy definition
        The probability of choosing best action is 1 - epsilon + r
        The probability of choosing other actions is r
        where r = uniform probability of choosing an action from random policy
                = epsilon / number of actions
        
        Args:
            state : current state that the agent is in
            actions : possible movements by the agent in the particular position
        """
        probs = []
        best_action = self.table.max_action(state, actions)
        
        if best_action is None: best_action = np.random.choice(actions)
        
        prob = self.__epsilon / len(actions)
        for action in actions:
            if action == best_action:
                probs.append(1 - self.__epsilon + prob)
            else:
                probs.append(prob)

        return np.random.choice(a=actions, p=probs)
    
    def get_action_prob(self, state, action):
        """
        Get the probability of choosing the action given the state. Note that
        it will assume that the agent can make any movements from the action
        space in the current state
        
        The probability is calculated using epsilon-greedy definition
        
        Args:
            state : current state that the agent is in
            action : the action that you want to check against the state
        """
        best_action = self.table.max_action(state, self.actions)
        
        if action == best_action:
            return 1 - self.__epsilon + (self.__epsilon / len(self.actions))
        else:
            return self.__epsilon / len(self.actions)

# Boltzmann -> using boltzmann distribution to obtain the probability and hence
# balancing between exploitation and exploration using those probabilities
class Boltzmann(Policy):
    def __init__(self, temperature=1000, table=None, learning=None,
                 decay_rate=0.01, min_temperature=10, agent_type=None):
        """
        Args:
            temperature : hyperparameter for balancing between exploitation and
                          exploration. When temperature -> 0, it maximizes
                          exploitation. When temperature -> ∞, it maximizes
                          exploration
            table : Q-Table for accessing the best action
            learning : Learning algorithm for accessing the table
            decay_rate : Rate of decay for temperature (go for exploitation)
            min_temperature : Minimum temperature to have minimum exploration
        """
        
        assert table is not None or learning is not None, \
            "Policy has access to either learning or q-table"
            
        assert temperature > 0, \
            "Temperature should be positive values"
        
        self.__temperature = temperature
        self.__table = table
        self.__learning = learning
        self.__decay_rate = decay_rate
        self.__min_temperature = min_temperature
        super().__init__(agent_type)
    
    @property
    def table(self):
        """
        Access Q-table
        """
        return self.__table if self.__learning is None else self.__learning.table

    def update(self, history, max_iter):
        """
        Update the temperature based on weight decay function
        temperature = temperature * e^{-lambda * t}
        where lambda is decay rate and t is number of time steps
        
        To allow more exploration at the beginning of the training,
        1 time step is taken as 1000 episodes
        
        Args:
            history (History) : history object for accessing number of episodes
        """
        fraction = history.num_episodes / max_iter * self.__decay_rate
        self.__temperature = max(self.__min_temperature, self.__temperature - self.__temperature * fraction)
    
    def get_action(self, state, actions):
        """
        Get the action based on probabilities evaluated
        using softmax/boltzmann distribution
        
        Args:
            state : current state that the agent is in
            actions : possible actions made by the agent
        """
        probs = np.array([np.exp(self.table[(state, action)].q_value \
            / self.__temperature) for action in actions])
        factor = probs.sum()
        return np.random.choice(a=actions, p=probs/factor)
    
    def get_action_prob(self, state, action):
        """
        Get the probability of choosing the action given the state. Note that
        it will assume that the agent can make any movements from the action
        space in the current state
        
        The probability is calculated using boltzmann distribution
        
        Args:
            state : current state that the agent is in
            action : the action that you want to check against the state
        """
        factor = np.sum([np.exp(self.table[(state, a)].q_value \
            / self.__temperature) for a in self.actions])
        return np.exp(self.table[(state, action)].q_value / self.__temperature) / factor

# Learn the probabilities for actions in a given state
# Learning would balance between exploitation and exploration
# go all the way to exploitation after certain time steps
class Pursuit(Policy):
    def __init__(self, agent_type=None, lr=0.1, decay_rate=0.001, max_lr=1):
        """
        Args:
            lr : learning rate
            decay_rate : rate of decay
            max_lr : maximum learning rate
        """
        assert lr > 0, \
            "Learning rate should be positive values"
        
        assert 0 < max_lr <= 1, \
            "maximum learning rate should between 0 and 1"
        
        super().__init__(agent_type)
        self.__lr = lr
        self.__decay_rate = decay_rate
        self.__max_lr = max_lr
        self.__table = Table(1 / len(self.actions))
    
    @property
    def table(self):
        """
        Have its own q-table where each q(s, a) represents the probability
        of choosing that particular action in that state
        """
        return self.__table
    
    def update(self, history, max_iter):
        """
        Update the learning rate based on weight decay function
        increment = lr * e^{-lambda * t}
        
        The lower the learning rate, the higher chance of exploration.
        Hence, the learning rate would be incremented to approach 1
        for exploitation
        """
        fraction = history.num_episodes / max_iter * self.__decay_rate
        self.__lr = min(self.__max_lr, self.__lr + self.__lr * fraction)
    
    def get_action(self, state, actions):
        """
        Provide the action based on learnt probability of q(s, a)
        When the learning converges, it will provide best action
        with q(s, a) = 1
        
        Args:
            state : state that the agent is currently in
            actions : possible actions performed by the agent in current position
        """
        probs = []
        best_ind = 0
        
        best_action = self.table.max_action(state, actions)
        if best_action is None: best_action = np.random.choice(actions)
        
        # Update probabilities
        for ind, action in enumerate(actions):
            prob = self.table[(state, action)].q_value
            
            if action == best_action:
                self.table[(state, action)] = min(1, prob + self.__lr * (1 - prob))
                best_ind = ind
            else:
                self.table[(state, action)] = max(0, prob + self.__lr * (0 - prob))
            
            probs.append(self.table[(state, action)].q_value)

        # Included in case the probability does not sum to 1
        total_probs = sum(probs)
        if total_probs < 1:
            probs[best_ind] += 1 - total_probs
        else:
            probs = [v / total_probs for v in probs]
        
        return np.random.choice(a=actions, p=probs)

    def get_action_prob(self, state, action):
        """
        Access the probability of action given state, which is
        learnt probability of q(s, a)
        """
        return self.table[(state, action)]

# Monte Carlo Control with weighted sampling
# It will update based on the actions in the episodes
# Before it learn to give the actions, it will use Epsilon Greedy to provide
# the action for a given state
class MonteCarlo(Policy):
    def __init__(self, learning, gamma=0.9, agent_type=None):
        """
        Args:
            learning : To access the q-table of learning algorithm
                       used for epsilon-greedy policy
            gamma : hyperparameter for controlling the update weightage
        """
        self.__table = Table()
        self.__ctable = Table()
        self.__gamma = gamma
        
        self.__behaviour = EpsilonGreedy(0.5, learning=learning)
        super().__init__(agent_type)
    
    @property
    def table(self):
        """
        Return learnt q-table from sample episodes
        """
        return self.__table
    
    def update(self, history, max_iter):
        """
        Update the q-table based on sampled episodes included in history object.
        
        The update is based on MonteCarlo Control algorithm which treats action
        taken by the agent as 1 episode
        
        Args:
            history : History object for accessing past episodes
        """
        histories = [agent for agent in history.latest_agent_history.values() if agent['agent_type'] == self.agent_type]
        
        for history in histories:
            states, moves, rewards = history['state'], history['move'], history['reward']
            assert len(states) == len(moves) + 1 and len(moves) == len(rewards)
            
            g, w = 0, 1
            for t in range(len(states)-2, -1, -1):
                state, move, reward = states[t], moves[t], rewards[t]
                g = self.__gamma * g + reward
                self.__ctable[(state, move)] = self.__ctable[(state, move)].q_value + w
                self.table[(state, move)] = self.table[(state, move)].q_value + \
                    (w / self.__ctable[(state, move)].q_value) * (g - self.table[(state, move)].q_value)
                w = w * self.get_action_prob(state, move) / self.__behaviour.get_action_prob(state, move)
                
                if w == 0:
                    break
    
    def get_action(self, state, actions):
        """
        Provide action based on the policy. If the MonteCarlo
        policy is not yet learnt, epsilon-greedy policy is
        used
        """
        best_action = self.table.max_action(state, actions)
        
        if best_action is None:
            return self.__behaviour.get_action(state, actions)
        else:
            return best_action
    
    def get_action_prob(self, state, action):
        """
        Get the probability of action given the state. The probability
        provided here is greedy policy once the policy is learnt, else
        provide epsilon-greedy probability.
        
        It assumes that the agent in the state could make all actions
        in the action space.
        """
        best_action = self.table.max_action(state, self.actions)
        
        if best_action is None:
            return self.__behaviour.get_action_prob(state, action)
        else:
            return 1 if action == best_action else 0