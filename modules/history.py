import pickle

class History:
    """Stores important information regarding the agent, environment, and Q learning details
    for testing, debugging, visualization.
    """
    def __init__(self):
        self.__agent_histories = []
        self.__env_history = []
        self.__metric_history = []

    @property
    def num_episodes(self):
        """Gets number of episodes so far"""
        return len(self.__env_history)

    @property
    def latest_agent_histories(self):
        """Gets agent history from the current episode"""
        return self.__agent_histories[-1]

    @property
    def latest_env_history(self):
        """Gets environment history from the current episode"""
        return self.get_env_history(-1)
    
    #! REMOVE LATER
    def set(self, env_hist, agent_hist):
        self.__env_history = env_hist
        self.__agent_history = agent_hist

    # just a shortcut
    def latest_agent_history(self, agent_idx):
        return self.get_agent_history(agent_idx, -1)

    def get_agent_history(self, agent_idx=None, episode=None):
        """Get agent history for a given episode
        NOTE: if episode is -1, it will get the latest episode's history

        Args:
            episode (int, optional): episode number. 
                Defaults to None in which case agent history from ALL episodes are returned.

        Returns:
            dict|list[dict]: Agent history for current episode or list of agent histories from all episodes
        """
        if agent_idx is not None:
            if episode is None:
                return [h[agent_idx] for h in self.__agent_histories]
            else:
                return self.__agent_histories[episode][agent_idx]
        else:
            if episode is None:
                return self.__agent_histories
            else:
                return self.__agent_histories[episode]
    
    def get_env_history(self, episode=None):
        """Get environment history for a given episode

        Args:
            episode (int, optional): episode number. 
                Defaults to None in which case environment history from ALL episodes are returned.

        Returns:
            dict|list[dict]: Environment history for current episode or 
                list of environment histories from all episodes
        """
        return self.__env_history if episode is None else self.__env_history[episode]

    def get_metric_history(self, episode=None):
        """Get metrics for a given episode

        Args:
            episode (int, optional): episode number. 
                Defaults to None in which case metric history from ALL episodes are returned.

        Returns:
            dict|list[dict]: Metric history for current episode or 
                list of metric histories from all episodes
        """
        return self.__metric_history if episode is None else self.__metric_history[episode]
    
    def new_episode(self, loc_a, loc_b):
        self.__agent_histories.append(dict())
        self.__env_history.append(dict(
            occupancy = dict(),
            loc_a = loc_a,
            loc_b = loc_b
        ))

    def new_episode_agent(self, agent, occ):
        # NOTE: must be called after self.new_episode not in the middle
        self.__agent_histories[-1][agent.agent_index] = dict(
            agent_type = agent.AGENT_TYPE,
            pos = [agent.pos],
            state = [agent.state],
            move = [],
            reward = [],
            target_switched = [], # Store number of steps when switching target
            target_reached = [], # Store number of steps to reach target
            other_agent = None # Store the agent with whom performs the handover
        )

        agent_x, agent_y = agent.pos

        occ[agent_y, agent_x] = 1
        self.__env_history[-1]['occupancy'][agent.agent_index] = [occ]
        
    def add_metrics(self, episode, loss_types, losses, reward):
        """Add metrics to history

        Args:
            episode (int): episode number
            loss_types (List[str]): contains name of losses
            losses (List[tuple(int, int)]): contains corresponding losses for location A and B
            reward (int): reward for the run
        """
        self.__metric_history.append(
            dict(
                episode=episode,
                loss_type=loss_types,
                loss=losses,
                reward=reward
            )
        )

    def save(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)