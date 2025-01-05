class Reward:
    """Gives rewards based on environment and agent condition."""
    BASE_REWARD = -1 # penalty for each step
    TARGET_REWARD = 200 # reward if agent reaches A or B or handoffpoint
    
    def __init__(self, env):
        """
        Args:
            env (Environment): environment where the agent is located
            agents (Agent): List of agents which we want to give the rewards to
        """
        self.__env = env
        self.__agents = env.agents

    def __get_history(self):
        """Get current environment history.

        Returns:
            History: The history of the current environment
        """
        return self.__env.history

    def get_reward(self, ind=None):
        """Get reward for agent in its current position.
        
        Args:
            ind (int): Index of the agent
        
        Returns:
            int: reward given if ind is specified
            list: reward givens for all agents if ind is not specified
        """
        assert 0 <= ind < self.__env.num_agents, \
            f"Index must be matched with agent index"
            
        latest_env_history = self.__get_history().latest_env_history
        history = self.__get_history()
        return [self.__calc_reward(agent, history.latest_agent_history(agent.agent_index), latest_env_history) for agent in self.__env.agents] \
            if ind is None else self.__calc_reward(self.__agents[ind], history.latest_agent_history(ind), latest_env_history)

    @staticmethod
    def __calc_reward(agent, agent_hist, env_hist):
        """
        Helper function for calculating the reward for the agent
        
        Args:
            agent_hist: Dictionary storing the agent's history within current episode
            env_hist: Dictionary storing the history of the environment
        
        Return:
            reward (int): Reward value for the movement made by the agent in current episode
        """
        # Get the number of tiles visited by the agent in this episode
        occupancy_grid = env_hist['occupancy'][agent.agent_index][-1]
        
        # Check if the agent reaching target
        agent_reached_target = len(agent_hist['target_reached']) > 0 and \
            agent.num_moves - 1 == agent_hist['target_reached'][-1] 
        
        agent_x, agent_y = agent.pos
        if agent_reached_target:
            # Get the number of movements to reach this target
            agent_moves_num = agent_hist['target_switched'][-1]
            # Additional penalty for number of moves
            reward = Reward.TARGET_REWARD - (agent_moves_num ** 2)
        else:
            # Additional penalty if agent steps on the same grid
            reward = Reward.BASE_REWARD * occupancy_grid[agent_y, agent_x]
        return reward
