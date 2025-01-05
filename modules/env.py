import random
from enum import Enum, IntEnum
import numpy as np
from copy import deepcopy

# from .agent_movement import Agent

class EnvironmentElement(IntEnum):
    """Enum type to define the characters representing 
    elements present in the grid world.
    """
    EMPTY = 0
    # agent types are prefixed by AGENT
    AGENT_TYPE_1 = 1
    AGENT_TYPE_2 = 2
    LOCATION_A = 3
    LOCATION_B = 4

    @classmethod
    def contains(cls, elem):
        """Check whether the given element is one of the pre-defined
        grid world elements.

        Args:
            elem (int or enum): either the integers representing the element or the enum names 
                (such as: EnvironmentElement.EMPTY)

        Returns:
            boolean: True if given element is a valid environment element, False otherwise
        """
        return elem in list(cls)

    @classmethod
    def name(cls, value):
        """Gets the name of the element corresponding to the integer value

        Args:
            value (int): integer representing the element in the grid world

        Returns:
            str: string name of the environment element
        """
        assert cls.contains(value), f"{value} is not a valid environment element"
        for name, member in cls.__members__.items():
            if (member == value):
                return name
    
    @classmethod
    def is_agent_type(cls, value):
        """Checks whether the given value is an agent or not.

        Args:
            value (int or enum): value for which we want to determine whether it is an agent or not.

        Returns:
            bool: True if the value is an agent, False otherwise
        """
        assert cls.contains(value), f"{value} is not a valid EnvironmentElement"
        elem_name = cls.name(value)
        return elem_name.split('_')[0] == "AGENT"
            
class Location:
    """Represents the location for an entity which can be fixed or random
    and the character to place in that location.
    """

    def __init__(self, character, pos=None):
        """
        Args:
            character (int): integer representing the entity
            pos (list, optional): fixed position of the entity. 
                Defaults to None which means that it has no fixed position.
        """
        self.__character = character
        pos = list(pos) if pos is not None else pos
        self.__fixed_pos = pos
        self.pos = pos

    @property
    def character(self):
        """integer value representing the element"""
        return self.__character

    @property
    def fixed_pos(self):
        """The fixed position of the entity if it exists, None otherwise"""
        return self.__fixed_pos
    
    def __repr__(self):
        return f"character: {self.__character} at pos: {self.pos} (fixed pos = {self.fixed_pos})"
    
# NOTE: GRID_SIZE ASSUMES START AT POS (0,0)
# NOTE: COORDINATES ARE ALWAYS ASSUMED TO BE (X, Y) BUT WHEN INDEXING, GRID[Y,X]
class Environment:
    """Represents the Grid World and is in charge of managing positions of elements within the environment
    """

    def __init__(self, grid_size, history, loc_a=None, loc_b=None):
        """
        Args:
            grid_size (list): [width, height] of the 2D grid world
            history (History): History object to store occupancy of elements within the grid world
            loc_a (list, optional): [x,y] position of location A if fixed. Defaults to None.
            loc_b (list, optional): [x,y] position of location B if fixed. Defaults to None.

        Raises:
            ValueError: if grid_size is not 2d or a negative width/height is given
        """
        if (len(grid_size) != 2 or any(x <= 0 for x in grid_size)):
            raise ValueError("Only non-zero 2D grid sizes are supported")
        self.__grid_size = grid_size
        self.__grid = self.initialize_grid()

        # reserve positions which are fixed so that other elements do not get randomly spawned on them
        self.__reserved_pos = [list(pos) for pos in [loc_a, loc_b] if pos is not None]

        # initialize A, B locations
        self.__loc_a = self.__initialize_loc(EnvironmentElement.LOCATION_A.value, loc_a)
        self.__loc_b = self.__initialize_loc(EnvironmentElement.LOCATION_B.value, loc_b)

        self.__loc_agents = []
        self.__agents = []

        self.history = history
        # initialize history for current environment configuration
        self.history.new_episode(self.loc_a, self.loc_b)
    
    @property
    def grid_size(self):
        """[width, height] of the grid world"""
        return self.__grid_size
    
    @property
    def grid(self):
        """numpy.array representation of the grid world"""
        return self.__grid

    @property
    def label_grid(self):
        """numpy.array representation of the grid world where integer values are replaced by their element names"""
        val_to_labels = lambda cell: EnvironmentElement.name(cell)
        return np.vectorize(val_to_labels)(self.grid)
    
    @property
    def loc_b(self):
        """[x,y] position of Location B"""
        return self.__loc_b.pos

    @property
    def loc_a(self):
        """[x,y] position of Location A"""
        return self.__loc_a.pos

    @property
    def loc_agents(self):
        """[x,y] position of agent location"""
        return self.__loc_agents
    
    @property
    def agents(self):
        """list of active agents in the environment"""
        return [a for a in self.__agents if a.is_active]
    
    @property
    def num_agents(self):
        """number of agents in the environment"""
        return len(self.loc_agents)

    @property
    def target_loc(self):
        """[x,y] position of the target (either location A or B)"""
        return self.__target_loc

    @property
    def history(self):
        """Gets the environment history from the latest episode"""
        return self.__history
    
    @history.setter
    def history(self, new_history):
        self.__history = new_history

    def get_loc_agent(self, idx):
        """Get the position of an agent with index idx"""
        return self.loc_agents[idx].pos

    def add_agent(self, agent, character, loc=None):
        """Add an agent to the environment.

        Args:
            agent (Agent): the agent to add to the environment
            character (int or enum): the int character representing the agent based on their type
            loc (list, optional): [x,y] starting position of the agent. Defaults to None, in which case a random spawn position is generated.

        Returns:
            int: index of the newly added agent
        """
        if isinstance(character, Enum):
            character = character.value
        # initialize the agent location
        agent_loc = self.__initialize_loc(character, loc)
        self.__loc_agents.append(agent_loc)
        self.__agents.append(agent)

        return self.num_agents - 1

    def set_loc_agent(self, idx, new_loc, fill_value=None):
        """Set the agent's location, to be used when moving the agent.

        Args:
            idx (int): agent index
            new_loc (list): new [x,y] position of the agent
            fill_value (int, optional): integer character to fill in the spot in the agent's previous location. Defaults to None.
        """
        assert idx in range(self.num_agents), \
            f"Given index: {idx} must be between 0 and num agents - 1: {self.num_agents-1}"
        assert self.check_pos(new_loc), f"Given position: {new_loc} is invalid"

        loc_agent = self.get_loc_agent(idx)
        agent_x, agent_y = loc_agent

        # determine the character to fill in the spot in the agent's previous location
        if fill_value is None:
            fill_value = EnvironmentElement.EMPTY.value 
            for loc in [*[self.__loc_agents[i] for i in range(self.num_agents) if i != idx], 
                        self.__loc_a, self.__loc_b]:
                if loc_agent == loc.pos:
                    fill_value = loc.character
                    break

        self.__grid[agent_y, agent_x] = fill_value
        new_x, new_y = new_loc
        # move the agent to the new position
        self.__grid[new_y, new_x] = self.__agents[idx].AGENT_TYPE.value
        self.__loc_agents[idx].pos = new_loc

        # update occupancy for agent's new position
        self.history.latest_env_history['occupancy'][idx][-1][new_y, new_x] += 1

    def initialize_grid(self):
        """create empty nxn grid"""
        return np.full(self.grid_size, EnvironmentElement.EMPTY.value)

    def __initialize_loc(self, character, pos=None):
        # create a Location object for the element
        loc = Location(character, pos)
        # if no pos is given, randomly generate
        if pos is None:
            loc.pos = self.get_rand_pos()
        self.occupy_grid(loc.pos, loc.character)
        return loc

    def reset(self, **kwargs):
        """Reset environment for new episode. 
        Resets locations of A, B, and agents if not fixed.

        Returns:
            List[List[Int, Int]]: list of [x,y] initial state of agents
        """
        assert "loc_agents" not in kwargs.keys() or \
            (isinstance(kwargs.get("loc_agents"), (list, tuple)) and len(kwargs.get("loc_agents")) == self.num_agents), \
            f"loc_agents must either be not provided or is a list of length: {self.num_agents} for each agent in the environment"
        
        def reset_loc(loc_ref, fixed_pos=None):
            curr_pos = loc_ref.pos
            # reset the current position to hold an empty element
            if curr_pos is not None and self.grid[curr_pos[1], curr_pos[0]] == loc_ref.character: 
                self.__grid[curr_pos[1], curr_pos[0]] = EnvironmentElement.EMPTY.value 

            new_pos = loc_ref.fixed_pos or fixed_pos
            if new_pos is None: # Location does not have a fixed position, get random position
                new_pos = self.get_rand_pos()
            
            loc_ref.pos = new_pos
            self.occupy_grid(loc_ref.pos, loc_ref.character)

        # reset location A
        reset_loc(self.__loc_a, kwargs.get('loc_a', None))
        # reset location B
        reset_loc(self.__loc_b, kwargs.get('loc_b', None))

        # move history to new episode
        self.history.new_episode(self.loc_a, self.loc_b)

        # reset agents
        for i in range(self.num_agents):
            loc_agent = kwargs.get("loc_agents", None)
            reset_loc(self.__loc_agents[i], loc_agent[i] if loc_agent is not None else None)
            self.__agents[i].reset()
        
        return [agent.state for agent in self.agents]


    def occupy_grid(self, pos, element):
        """Set grid cell at a particular position to the given environment element.

        Args:
            pos (list): [x,y] position of the element
            element (int or Enum): character representing the element
        """
        assert EnvironmentElement.contains(element), f"{element} is not a valid environment element"
        assert self.check_pos(pos), f"Given position: {pos} is not valid"
        
        pos_x, pos_y = pos
        if isinstance(element, Enum):
            element = element.value
        self.__grid[pos_y, pos_x] = element

    def get_rand_pos(self):
        """Randomly generate [x,y] position which is unoccupied by any other non-EMPTY elements 

        Returns:
            list: randomly generated [x,y] position
        """
        rand_x, rand_y = (random.randint(0, size-1) for size in self.grid_size)
        # if the randomly generated position is already occupied or reserved, re-generate position
        while self.grid[rand_y, rand_x] != EnvironmentElement.EMPTY or \
            [rand_x, rand_y] in self.__reserved_pos:
            rand_x, rand_y = (random.randint(0, size-1) for size in self.grid_size)
        return [rand_x, rand_y]
    
    def check_pos(self, pos):
        """Check whether given position is valid for this grid world

        Args:
            pos (list): [x,y] position

        Returns:
            boolean: True if position is within grid world, False otherwise
        """
        assert len(pos) == len(self.__grid_size), \
            f"Given position: {pos}, has different dimensions than the grid world"
        pos_x, pos_y = pos
        grid_w, grid_h = self.grid_size
        return grid_w > pos_x >= 0 and grid_h > pos_y >= 0

    def get_cells_around_pos(self, pos):
        """Get positions of cells around given position

        Args:
            pos (list): [x,y] position for which we want its surrounding cells

        Returns:
            List[[int, int]]: list of [x,y] positions of existing cells around position 
        """
        pos_x, pos_y = pos
        grid_w, grid_h = self.grid_size
        cells = []
        for x in range(pos_x-1, pos_x+2):
            # if x position is invalid, skip
            if not grid_w > x >= 0:
                continue
            for y in range(pos_y-1, pos_y + 2):
                # if y position is invalid or (x,y) is the given position, skip
                if not (grid_h > y >= 0) or (x,y) == (pos_x,pos_y):
                    continue
                cells.append([x, y])
        return cells