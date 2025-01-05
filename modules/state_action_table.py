# Note : QInfo would store the information related to Q-values
# Note : Currently it stores only q-value and counter (times of updates)
# Note : But it can extend to store information for UCB policy
class QInfo:
    def __init__(self, value, counter):
        """
        Args:
            value : q-value, q(s, a)
            counter : number of times the pair (s, a) is updated
        """
        assert counter >= 0, "Counter should be non-negative"
        
        self.__qv = value
        self.__counter = counter
    
    def __eq__(self, other):
        """
        Allow equality comparison for the class using q-value
        """
        return self.q_value == other.q_value
    
    def __lt__(self, other):
        """
        Allow ordering comparison for the class using q-value
        """
        return self.q_value < other.q_value
    
    def __gt__(self, other):
        """
        Allow ordering comparison for the class using q-value
        """
        return self.q_value > other.q_value
    
    @property
    def q_value(self):
        """
        Return the q-value
        """
        return self.__qv

    @q_value.setter
    def q_value(self, value):
        """
        Update the q-value

        Args:
            value : used to initialize the q-value
        """
        self.__qv = value
    
    @property
    def counter(self):
        """
        Get the number of updates of this pair (s, a)
        """
        return self.__counter

    @counter.setter
    def counter(self, val):
        """
        Set the number of updates of this pair (s, a)

        Args:
            val : used to initialize the counter
        """
        assert val >= 0, "Counter should be non-negative"
        self.__counter = val

# Note : Assume state can be converted to string & every action is a class
# Note : __str__ | __repr__ should be defined for state and action classes (if those are user-defined)
class Table:
    def __init__(self, init_value=0):
        """
        Table for storing q-values, q(s, a)
        
        Args:
            init_value : used to initialize the q-value
        """
        self.__table = dict()
        self.__init_value = init_value
        
    def __repr__(self):
        """
        Return the whole table in string
        """
        return repr(self.__table)
    
    def __getitem__(self, key):
        """
        Access the q-value using (state, action) pair as the key

        Args:
            key: A tuple or list containing two elements: the state and the action. 
        """
        assert isinstance(key, (tuple, list)) and len(key) == 2, \
            "Key consists of state and/or action in form of tuple or list, e.g. (state, action)"

        state, action = key
        
        try:
            return self.__table[str(state)][action]
        except KeyError:
            return QInfo(self.__init_value, 0)
    
    def __setitem__(self, key, elem):
        """
        Update the q-value of (state, action) pair

        Args:
            key: A tuple or list containing two elements: the state and the action. 
            elem: The new Q-value to set for the given state-action pair.
        """
        assert isinstance(key, (tuple, list)) and len(key) == 2, \
            "Key consists of state and action in form of tuple or list, e.g. (state, action)"
        state, action = key
        state = str(state)
        
        if state not in self.__table:
            self.__table[state] = {action: QInfo(elem, 1)}
        else:
            if action not in self.__table[state]:
                self.__table[state][action] = QInfo(elem, 1)
            else:
                info = self.__table[state][action]
                info.q_value = elem
                info.counter += 1
    
    def max_action(self, state, actions):
        """
        Return the best action with maximum q-values based on the
        given state and possible actions

        Args:
            state: The current state for which we want to find the 
                    action that has the highest Q-value.
            actions: A list or set of possible actions that can be 
                    taken in the provided state.
        """
        try:
            all_actions = self.__table[str(state)]
            actions = dict(filter(lambda x: x[0] in actions, all_actions.items()))
            return max(actions, key=lambda x: actions.get(x).q_value)
        except (KeyError, ValueError):
            return
    
    # Return the max q-value given the state
    def max_value(self, state):
        """
        Return the maximum q-value of the given state. It assumes that
        the agent could perform all the action in the action space
        in the current position

        Args:
            state: The current state for which we want to retrieve the 
                    maximum Q-value among all possible actions.
        """
        try:
            return max(self.__table[str(state)].values()).q_value
        except (KeyError, ValueError):
            return 0
    
    def items(self):
        """
        Return (state, actions) pair
        """
        return self.__table.items()