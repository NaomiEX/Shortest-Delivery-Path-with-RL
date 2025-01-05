from abc import ABC, abstractmethod
from .env import EnvironmentElement
from .utils import get_min_steps

class Movement(ABC):
    """Abstract movement class"""
    @classmethod
    @abstractmethod
    def move(cls, pos):
        """Given current position, should return the updated position after making the move

        Args:
            pos (list): current [x,y] position
        """
        pass

    @classmethod
    def is_possible(cls, pos, available_cells):
        """Checks whether a move can be executed from the given position

        Args:
            pos (list): current [x,y] position
            available_cells (list[[int,int]]): list of [x,y] positions available to move to

        Returns:
            boolean: True if move is possible, False otherwise
        """
        return cls.move(pos) in available_cells
    
    @classmethod
    def name_to_cls(cls, move_name):
        """
        Given a move name (string), returns the associated subclass of the class it's invoked on.

        Args:
            move_name (str): Name of the subclass.

        Returns:
            type: The subclass associated with the given move_name.
        """
        name_to_cls_dict = {subcls.__name__: subcls for subcls in cls.__subclasses__()}
        assert move_name in name_to_cls_dict, f"Given move name: {move_name} is invalid"

        return name_to_cls_dict[move_name]
    
    @classmethod
    def cls_to_idx(cls, class_):
        """
        Given a Movement subclass, returns the associated index (integer).

        Args:
            class_ (Movement): Movement subclass

        Returns:
            int: the index for the given movement
        """
        cls_to_idx_dict = {subcls: idx for idx, subcls in enumerate(cls.__subclasses__())}
        assert class_ in cls_to_idx_dict, f"Given class, {class_}, is invalid"
        return cls_to_idx_dict[class_]

    @classmethod
    def idx_to_cls(cls, idx):
        """
        Given an index (integer), returns the associated subclass of the class it's invoked on
        
        Args:
            idx (int): Index of the subclass
        
        Returns:
            Movement: The subclass associated with the given move_name
        """
        idx_to_cls_dict = {ind: subcls for ind, subcls in enumerate(cls.__subclasses__())}
        assert idx in idx_to_cls_dict, f"Given index, {idx}, is invalid"
        return idx_to_cls_dict[idx]
    
    @classmethod
    def name_to_idx(cls, name):
        """
        Given a move name (string), returns the index of associated subclass of the class it's invoked on
        
        Args:
            name (string): Name of the subclass
        
        Returns:
            index: The index of associated subclass with the given name
        """
        cls_to_idx_dict = {subcls.__name__: idx for idx, subcls in enumerate(cls.__subclasses__())}
        assert name in cls_to_idx_dict, f"Given class, {name}, is invalid"
        return cls_to_idx_dict[name]

class MoveLeft(Movement):
    """Move left which involves decreasing x-coordinate by 1."""
    @classmethod
    def move(cls, pos):
        return [pos[0]-1, pos[1]]
    
    @classmethod
    def direction(cls):
        return '←'

class MoveRight(Movement):
    """Move right which involves increasing x-coordinate by 1."""
    @classmethod
    def move(cls, pos):
        return [pos[0]+1, pos[1]]

    @classmethod
    def direction(cls):
        return '→'

class MoveUp(Movement):
    """Move up which involves decreasing y-coordinate by 1.
    NOTE: y-axis is top-to-bottom"""
    @classmethod
    def move(cls, pos):
        return [pos[0], pos[1]-1]

    @classmethod
    def direction(cls):
        return '↑'

class MoveDown(Movement):
    """Move down which involves increasing y-coordinate by 1.
    NOTE: y-axis is top-to-bottom"""
    @classmethod
    def move(cls, pos):
        return [pos[0], pos[1]+1]

    @classmethod
    def direction(cls):
        return '↓'

class Agent(ABC):
    """Abstract class to represent a generic agent in the grid world."""

    def __init__(self, env, l_algo, target, loc=None, mode="absolute_positions"):
        """
        Args:
            env (Environment): environment where the agent is located
            l_algo (Learning): the learning/model used to determine the agent's movement at a particular state
            target (Agent, list, tuple): initial target for the agent
            loc (list, optional): initial spawn position of the agent. Defaults to None which means the agent is spawned on a random position.
            mode (str, optional): determines what the agent returns as its state. Defaults to "absolute_positions".
                        - "single_relative_target": the agent's state is 2 values: [pos_x-target_pos_x, pos_y-target_pos_y] 
                                                    relative to a single target
                        - "absolute_positions": the agent's state is 5 values: [pos_x, pos_y, target_pos_x, target_pos_y, has_item]
                            > NOTE: when has_item is False:
                                        - target (for type 1) refers to location A
                                        - target (for type 2) refers to the agent closest to A, 
                                    when has_item is True:
                                        - target (for type 1) refers to the closest type 2 agent
                                        - target (for type 2) refers to location B
        """
        assert mode in ["single_relative_target", "absolute_positions"], \
            f'agent mode must be one of ["single_relative_target", "absolute_positions"] but instead got: {mode}'
        self.__env = env
        
        self.__target = target
        self._initial_target = target # only used for reset
        self.__l_algo = l_algo
        self.__has_item = False
        self.__is_active = True

        self.mode = mode
        
        # add agent to environment
        self.__agent_idx = self.env.add_agent(self, self.AGENT_TYPE,loc)

    @property
    def is_active(self):
        """True if the agent has finished its episode, False otherwise"""
        return self.__is_active
        
    @property
    def env(self):
        """Agent's environment"""
        return self.__env
    
    @property
    def l_algo(self):
        """Agent's learning algorithm/model"""
        return self.__l_algo
    
    @property
    def target_elem(self):
        """Reference to the target object"""
        return self.__target
    
    @property
    def target(self):
        """[x,y] position of the target"""
        target_elem = self.__target
        return target_elem.pos if isinstance(target_elem, Agent) else list(target_elem)
    
    @property
    def agent_index(self):
        """index of the agent within the environment"""
        return self.__agent_idx
    
    @property
    def has_item(self):
        """True if agent is currently holding an item, False otherwise"""
        return self.__has_item

    @property
    def pos(self):
        """Gets agent [x,y] position"""
        return self.env.get_loc_agent(self.agent_index)
    
    @property
    def state(self):
        """Gets current state of agent based on state mode"""
        if (self.mode == "single_relative_target"):
            return [a - b for a, b in zip(self.pos, self.target)]
        elif(self.mode == "absolute_positions"):
            return [*self.pos, *self.target, int(self.has_item)]
        else:
            raise Exception(f"mode: {self.mode} is not supported")
    
    @property
    def history(self):
        """Reference to history object"""
        return self.__env.history

    @property
    def num_moves(self):
        """Number of moves made by the agent so far in the current episode"""
        return len(self.history.latest_agent_history(self.agent_index)['move'])
    
    @is_active.setter
    def is_active(self, bool_val):
        self.__is_active = bool_val
    
    @has_item.setter
    def has_item(self, bool_val):
        self.__has_item = bool_val

    @target.setter
    def target(self, new_target):
        """Change the agent's target element.

        Args:
            new_target (Agent, tuple, list): reference to the target object.
        """
        assert isinstance(new_target, (Agent, tuple, list)), \
            f"Target of an agent must either be another Agent or a position (as a list or tuple), instead got: {new_target}"
        # as long as the target exists and is different from current target, record in history
        if self.__target is not None and new_target != self.__target:
            latest_agent_history = self.history.latest_agent_history(self.agent_index)
            latest_env_history = self.history.latest_env_history
            latest_agent_history["target_switched"].append(
                    len(latest_agent_history["move"]))
            # add brand new occupancy grid whenever target changes
            latest_env_history['occupancy'][self.agent_index].append(
                self.env.initialize_grid()
            )
            agent_x, agent_y = self.pos
            latest_env_history['occupancy'][self.agent_index][-1][agent_y, agent_x] = 1
        
        self.__target = new_target


    @abstractmethod
    def switch_target(self):
        """Based on certain conditions, switch the agent's target."""
        pass

    def reset(self, initial_target=None):
        """Reset the agent back to its initial state for a new episode.

        Args:
            initial_target (Agent | tuple | list, optional): reference to the initial target of the agent. 
                                                             Defaults to None wherein the agent's previous initial target is used.
        """
        self.history.new_episode_agent(self, self.env.initialize_grid())
        self.__target = self._initial_target if initial_target is None else initial_target
        self.has_item = False
        self.is_active = True

    def get_other_agents(self, filter_func=None):
        """Gets the list of agents (of a different type) currently in the environment.

        Args:
            filter_func (func, optional): filter function which accepts an Agent and returns False if the agent should be filtered out, True otherwise. 
                                          Defaults to None, in which case no agents are filtered out.

        Returns:
            List[Agent]: list of agents of a different type
        """
        agents = self.env.agents
        other_agents = []
        for agent in agents:
            if agent.AGENT_TYPE != self.AGENT_TYPE and (filter_func is None or filter_func(agent)):
                other_agents.append(agent)
        return other_agents
    
    
    def get_closest_other_agent(self, filter_func=None):
        """Gets the closest agent (of a different type) from the agent's current location.

        Args:
            filter_func (func, optional): filter function which accepts an Agent and returns False if the agent should be filtered out, True otherwise. 
                                          Defaults to None, in which case no agents are filtered out.

        Returns:
            Agent | None: the closest agent (of a different type) which fulfills the filter function or None if no such agent exists.
        """
        other_agents = self.get_other_agents(filter_func=filter_func)
        return self._get_closest_agent(other_agents) if len(other_agents) > 0 else None
    

    def _get_closest_agent(self, agent_lst, from_pos=None):
        """Gets the agent closest (determined by L1 distance) from a particular position.

        Args:
            agent_lst (List[Agent]): list of agents
            from_pos (list, optional): [x,y] position from which we want to get the closest agent to. 
                                       Defaults to None, in which case the agent's current position is used.

        Returns:
            Agent: the closest agent to from_pos
        """
        if from_pos is None: from_pos = self.pos
        min_steps, closest_agent = None, None
        for agent in agent_lst:
            # get L1 distance between from_pos and agent_i's position
            step_dist = get_min_steps(from_pos, agent.pos)
            if min_steps is None or step_dist < min_steps:
                min_steps = step_dist
                closest_agent = agent
        return closest_agent


    def get_possible_movement(self):
        """Gets all possible movements from agent's current position

        Returns:
            list[Movement]: list of Movement subclasses which the agent can execute from its position
        """
        surrounding_cells = self.env.get_cells_around_pos(self.pos)
        possible_movements = []
        # check all subclasses of Movement to see whether the agent can execute them
        for movement in Movement.__subclasses__():
            if movement.is_possible(self.pos, surrounding_cells):
                possible_movements.append(movement)
        return possible_movements
    
    def move(self, movement=None):
        """Move the agent within the environment. 

        Args:
            movement (Movement, optional): Movement subclass to execute. 
                                           Defaults to None, in which case the agent's learning algorithm is used to suggest the movement to perform.
        """
        assert movement is None or movement in Movement.__subclasses__(), \
            f"Given move: {movement} must either be None or a Movement subclass"

        if movement is None:
            movement = self.l_algo.get_action(self.state, self.get_possible_movement())
        # change the agent's position in the environment based on the resulting position after executing the move
        self.env.set_loc_agent(self.agent_index, movement.move(self.pos))

        # update agent history
        latest_agent_history = self.history.latest_agent_history(self.agent_index)
        latest_agent_history["pos"].append(self.pos)
        latest_agent_history["move"].append(movement.__name__)
        

    def __repr__(self):
        target_elem = f'(Type: {self.target_elem.AGENT_TYPE} agent index: {self.target_elem.agent_index})' \
            if isinstance(self.target_elem, Agent) else self.target_elem
        return f"Type: {self.AGENT_TYPE} agent index: {self.__agent_idx}\n" + \
               f"loc: {self.pos}, target pos: {self.target}, target elem: {target_elem}, has item: {self.has_item}"
    
    def __str__(self):
        return f"Type: {self.AGENT_TYPE} agent index: {self.__agent_idx}"

class TypeOneAgent(Agent):
    """Represents a Type I agent which is responsible for:
    1. Picking up an item at location A
    2. Handing over the item to a Type II agent
    """

    AGENT_TYPE = EnvironmentElement.AGENT_TYPE_1

    def __init__(self, env, l_algo, initial_target=None, mode="absolute_positions"):
        """
        Args:
            env (Environment): the environment where the agent is in
            l_algo (Learning): the agent's learning algorithm/model which suggests its movements.
            initial_target (Agent | list | tuple, optional): agent's initial target. 
                                Defaults to None, in which case location A is used as the initial target.
            mode (str, optional): determines what the agent returns as its state. Defaults to "absolute_positions".
                        - "single_relative_target": the agent's state is 2 values: [pos_x-target_pos_x, pos_y-target_pos_y] 
                                                    relative to a single target
                        - "absolute_positions": the agent's state is 5 values: [pos_x, pos_y, target_pos_x, target_pos_y, has_item]
                            > NOTE: when has_item is False:
                                        - target (for type 1) refers to location A
                                        - target (for type 2) refers to the agent closest to A, 
                                    when has_item is True:
                                        - target (for type 1) refers to the closest type 2 agent
                                        - target (for type 2) refers to location B. 
                        Defaults to "absolute_positions".
        """
        target = env.loc_a if initial_target is None else initial_target
        super().__init__(env, l_algo, target, mode=mode)

        # add agent to history
        self.history.new_episode_agent(self, self.env.initialize_grid())

    def switch_target(self):
        """If the type I agent is holding an item, its target is the closest Type II agent which is *not* holding an item.
        If it is not holding an item, its target remains at location A.
        """
        if self.has_item:
            new_target = self.get_closest_other_agent(lambda a: a.has_item == False)
            assert new_target is not None, "Existing Type I agent has an item but no available Type II agents!"
        else:
            new_target = self.env.loc_a
        self.target = new_target
    
    def move(self, movement=None):
        """Moves the type I agent. 

        ## Before moving:
        - updates the agent's target

        ## After moving:
        - checks whether it has reached A, and if so picks up the item at A.
        - checks whether it is holding an item and has reached a type II agent, if so perform handover.

        Args:
            movement (Movement, optional): fixed movement to perform. 
                Defaults to None, in which case the agent's learning algorithm/model is used to suggest movements.
        """
        ## before move
        # update agent's target
        self.switch_target()

        # perform the move and update the agent's position in the environment
        super().move(movement)

        ## after move
        latest_agent_history = self.history.latest_agent_history(self.agent_index)
        if (self.pos == self.target):
            # if agent reaches location A, and does not yet have an item, pick up an item
            if (self.target == self.env.loc_a and not self.has_item):
                print(f"{str(self)} PICKED UP ITEM AT A!")
                self.has_item = True
                self.switch_target()
                
                # store the move in which the agent picked up the item
                latest_agent_history["target_reached"].append(self.num_moves)
                
            if self.has_item:
                # gets the list of type II agents which are in the same position as this agent and are not holding an item
                target_agent_lst = self.get_other_agents(lambda a: a.pos == self.pos and not a.has_item)
                # if there is such an agent, perform handover to the first available type II agent.
                if len(target_agent_lst) > 0:
                    target_agent = target_agent_lst[0]
                    target_agent.has_item = True
                    self.has_item = False
                    target_agent.switch_target()
                    
                    # update this agent's history
                    latest_agent_history["target_reached"].append(self.num_moves)
                    latest_agent_history["other_agent"] = target_agent.agent_index
                    
                    # Update target agent's history
                    target_hist = self.history.latest_agent_history(target_agent.agent_index)
                    target_hist["other_agent"] = self.agent_index
                    target_hist["target_reached"].append(target_agent.num_moves)

                    # corner case where it meets the type II agent on B
                    if (target_agent.pos == target_agent.target == self.env.loc_b):
                        # in that case, right after the handover, the type II agent automatically delivers the item and finishes its episode
                        target_hist["target_reached"].append(target_agent.num_moves)
                        target_agent.is_active = False

                    print(f"handover between {str(self)} and {str(target_agent)}")
                    
                    # type I agent finishes its job and deactivates after a handover
                    self.is_active=False
        
        # update agent history with its new state
        latest_agent_history["state"].append(self.state)

class TypeTwoAgent(Agent):
    """Represents a Type II agent which is responsible for:
    1. Receiving an item from a Type I agent
    2. Delivering the item to location B
    """

    AGENT_TYPE = EnvironmentElement.AGENT_TYPE_2

    def __init__(self, env, l_algo, initial_target=None, mode="absolute_positions"):
        """
        Args:
            env (Environment): the environment where the agent is in
            l_algo (Learning): the agent's learning algorithm/model which suggests its movements.
            initial_target (Agent | list | tuple, optional): agent's initial target. 
                                Defaults to None, in which case the closest type I agent is used as the initial target.
            mode (str, optional): determines what the agent returns as its state. Defaults to "absolute_positions".
                        - "single_relative_target": the agent's state is 2 values: [pos_x-target_pos_x, pos_y-target_pos_y] 
                                                    relative to a single target
                        - "absolute_positions": the agent's state is 5 values: [pos_x, pos_y, target_pos_x, target_pos_y, has_item]
                            > NOTE: when has_item is False:
                                        - target (for type 1) refers to location A
                                        - target (for type 2) refers to the agent closest to A, 
                                    when has_item is True:
                                        - target (for type 1) refers to the closest type 2 agent
                                        - target (for type 2) refers to location B. 
                        Defaults to "absolute_positions".
        """
        super().__init__(env, l_algo, initial_target, mode=mode)
        self.target = self.get_agent_target() if initial_target is None else initial_target
        self._initial_target = self.target_elem
        # add agent to history
        self.history.new_episode_agent(self, self.env.initialize_grid())
    
    def get_agent_target(self):
        """Get the agent's type I target based on the state mode.
        - "single_relative_target": the closest type I agent to the current agent is the target
        - "absolute_positions": the type I agent closest to location A is the target

        Returns:
            Agent: the closest type I agent target
        """
        if (self.mode == "single_relative_target"):
            # keeping old method of chasing closest type I agent to self
            return self.get_closest_other_agent()
        elif (self.mode == "absolute_positions"):
            # in this method, target is the type I agent closest to loc A instead of self
            return self._get_closest_agent(self.get_other_agents(), from_pos=self.env.loc_a)
        else:
            raise Exception(f"mode: {self.mode} not supported")
    
    def reset(self):
        """Reset the type II agent to its initial state for a new episode 
        where the initial target is the closest type I agent in the new episode."""
        super().reset(initial_target=self.get_agent_target())
    
    def switch_target(self):
        """If the type II agent is holding an item, its target is location B. 
        If it is not holding an item, its target is the closest Type I agent holding an item, 
        if no such agent exists, the target is the closest Type I agent not holding an item
        (closeness is determined by state mode).
        """
        if self.has_item:
            new_target = self.env.loc_b
        else:
            # get closest other agent holding an item
            new_target = self.get_closest_other_agent(lambda a: a.has_item == True)
            # if no type I agent with item, target is the closest type I agent
            if new_target is None:
                new_target = self.get_agent_target()
        self.target = new_target

    def move(self, movement=None):
        """Moves the type II agent.

        ## Before moving
        - updates the agent's target

        ## After moving
        - checks whether it is not holding an item and has met with a type I agent holding an item, if so perform handover
        - checks whether it is holding an item and has reached B, if so deliver item to B.

        Args:
            movement (Movement, optional): fixed movement to perform. 
                Defaults to None, in which case the agent's learning algorithm/model is used to suggest movements.
        """
        ## before move
        # update agent's target
        self.switch_target()

        # perform the move and update the agent's position in the environment
        super().move(movement)

        ## after move
        latest_agent_history = self.history.latest_agent_history(self.agent_index)
        # if agent reaches target agent and agent is currently not holding an item, perform handover
        if (self.pos == self.target and isinstance(self.target_elem, Agent) and not self.has_item):
            target_agent = self.target_elem
            assert target_agent.pos == self.pos, \
                f"Supposedly type 2 agent reached target, but positions are different, self: {self.pos}, target agent: {target_agent.pos}"

            # execute handover
            # NOTE: both agent types have handovers because either agent can meet at their turns
            if target_agent.has_item:
                target_agent.has_item = False
                target_agent.is_active = False
                self.has_item = True
                self.switch_target()
                target_agent.switch_target()

                print(f"handover between {str(self)} and {str(target_agent)}")
                
                # update current agent's history
                latest_agent_history["target_reached"].append(self.num_moves)
                latest_agent_history["other_agent"] = target_agent.agent_index
                
                # Update target_agent history
                target_hist = self.history.latest_agent_history(target_agent.agent_index)
                target_hist["other_agent"] = self.agent_index
                target_hist["target_reached"].append(target_agent.num_moves)
        
        # if the agent reaches B with the item, deliver item to B and finish episode
        if (self.pos == self.target == self.env.loc_b and self.has_item):
            print(f"{str(self)} succesfully delivered to B!")
            
            latest_agent_history["target_reached"].append(self.num_moves)

            # type II agent finishes its job and deactivates after it delivers to B
            self.is_active=False
        
        # update agent history with its new state
        latest_agent_history["state"].append(self.state)
