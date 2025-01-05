from abc import ABC, abstractmethod
from .env import EnvironmentElement
from .history import History
from .utils import get_min_steps

class Loss(ABC):
    """Implements the loss between agent path and ideal path"""

    @staticmethod
    @abstractmethod
    def distance(a, b):
        """Given two values a, b calculate distance between them

        Args:
            a (list): first value
            b (list): second seconds
        """
        pass
    
    @classmethod
    def evaluate_single(cls, history, episode=-1):
        """Calculates loss for a single episode

        Args:
            history (History): contains all environment and agent configurations from all episodes
            episode (int, optional): episode number. Defaults to -1 which takes the last episode.

        Returns:
            int, int: loss incurred by the agent in path taken to A, B for given episode
        """
        agent_hists = history.get_agent_history(episode=episode)
        env_hist = history.get_env_history(episode)
        A_pos = env_hist['loc_a']
        B_pos = env_hist['loc_b']
        a_loss, b_loss = None, None
        loss_1, loss_2 = None, None
        
        for agent_hist in agent_hists.values():
            if agent_hist['agent_type'] == EnvironmentElement.AGENT_TYPE_1:
                init_pos = agent_hist['pos'][0]
                
                # if agent reached location A during that episode
                if len(agent_hist["target_reached"]) > 0:
                    # calculate minimum steps taken to reach A from starting agent position
                    min_step_to_A = get_min_steps(A_pos, init_pos)
                    # get number of moves taken by agent to reach A
                    steps_taken_to_A = agent_hist["target_reached"][0]
                    # calculate distance between ideal and actual number of steps taken to A
                    a_loss = cls.distance(steps_taken_to_A, min_step_to_A)
                
                if len(agent_hist["target_reached"]) > 1:
                    # calculate minimum steps taken to reach type II agent from A
                    pos_idx = agent_hist["target_reached"][0]
                    other_agent = agent_hists[agent_hist["other_agent"]]
                    min_step_to_other = get_min_steps(agent_hist['pos'][pos_idx], other_agent['pos'][pos_idx]) // 2
                    
                    # gets number of moves taken by agent
                    steps_taken_to_other = agent_hist["target_reached"][1] - agent_hist["target_reached"][0]
                    
                    # calculate the distance
                    loss_1 = cls.distance(steps_taken_to_other, min_step_to_other)
                    
            if agent_hist['agent_type'] == EnvironmentElement.AGENT_TYPE_2:
                if len(agent_hist["target_reached"]) > 0:
                    # Calculate shortest distance from Type I agent to A
                    other_agent = agent_hists[agent_hist['other_agent']]
                    steps_to_A = get_min_steps(other_agent['pos'][0], A_pos)
                    
                    # Calculate shortest distance between Type I at A to Type II
                    pos_idx = other_agent["target_reached"][0]
                    min_step_to_other = get_min_steps(agent_hist['pos'][pos_idx], other_agent['pos'][pos_idx]) // 2
                    
                    # calculate minimum steps taken to reach nearest agent with item
                    min_step_to_other += steps_to_A
                    
                    # get steps taken by agent
                    steps_taken_to_other = agent_hist["target_reached"][0]
                    # calculate the distance
                    loss_2 = cls.distance(steps_taken_to_other, min_step_to_other)
                
                # if agent reached location B during that episode
                if len(agent_hist["target_reached"]) > 1:
                    pos_idx = agent_hist["target_reached"][0]
                    # calculate minimum steps taken to reach B from handoff
                    min_step_to_B = get_min_steps(B_pos, agent_hist['pos'][pos_idx])
                    # get number of moves taken by agent to reach B
                    steps_taken_to_B = agent_hist["target_reached"][1] - agent_hist["target_reached"][0]
                    # calculate distance between ideal and actual number of steps taken to B
                    b_loss = cls.distance(steps_taken_to_B, min_step_to_B)

        return a_loss, loss_1, loss_2, b_loss

    @classmethod
    def evaluate(cls, history: History, start_id=1):
        """Calculate the loss for all episodes up to current

        Args:
            history (History): contains all environment and agent configurations from all episodes

        Returns:
            List[int], List[int]: losses incurred by the agent in path taken to A, B for each episode
        """
        return zip(*((cls.evaluate_single(history, episode=episode_num) for episode_num in range(start_id, history.num_episodes))))


class L2_Loss(Loss):
    """Implements L2 loss where distance is calculated using squared difference"""
    @staticmethod
    def distance(a, b):
        return (a-b)**2

class L1_Loss(Loss):
    """Implements L1 loss where distance is calculated using absolute difference"""
    @staticmethod
    def distance(a, b):
        return abs(a-b)