from datetime import datetime
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent_movement import Movement
from .state_action_table import QInfo
from copy import deepcopy


def get_activation(act):
    """
    Given the name of the activation function, return the associated function
    from pytorch library
    
    Args:
        act (string): name of activation function
        
    Return:
        type: function associated with the name
    """
    act = act.lower()
    if act == 'relu':
        return F.relu
    elif act == 'mish':
        return F.mish
    elif act == 'gelu':
        return F.gelu
    else:
        raise ValueError("Activation is not supported!")


class Baseline(nn.Module):
    """
    Simple Linear Model for approximating the q-values for each
    actions given the state as the input.
    """
    def __init__(self, in_channel, out_channels=[], activation='relu'):
        """
        Class constructor for creating class instance
        
        Args:
            in_channel (int): Number of channels of input tensor (values representing state)
            out_channels (List[int]): list of number of channels in each hidden layer, including last layer
            activation (string): Activation function to be used in the model
        """
        assert isinstance(out_channels, (list, tuple)), \
            "Provide neurons of hidden layers in a list/tuple"
        assert isinstance(in_channel, int), \
            "Povide input channel as integer"

        if out_channels:
            assert len(out_channels) > 2, \
                "Must have at least 2 hidden layers"
            assert out_channels[-1] == 4, \
                "Last layer should have 4 neurons, corresponding to 4 actions"
        
        super().__init__()
        self.in_channel = in_channel
        self.__act = get_activation(activation)
        
        # Instantiate the layers used in the model
        if out_channels:
            self.__head = nn.Linear(in_channel, out_channels[0])
            self.__hidden = nn.ModuleList([nn.Linear(out_channels[i], out_channels[i+1]) for i in range(len(out_channels)-1)])
        else:
            # Use default setting if not specified
            self.__head = nn.Linear(in_channel, 4)
            self.__hidden = nn.ModuleList([
                nn.Linear(4, 8),
                nn.Linear(8, 4)
            ])
        
        # Initialize the weights for each layer
        self.apply(self._init_weights)
    
    def __getitem__(self, idx):
        """
        Retrieve particular layer in the model using indexing
        
        Args:
            idx (int): Index of the layer
        """
        assert isinstance(idx, int), "Index must be integer"
        assert -1 < idx < len(self.__hidden) + 1, "Only positive indices accepted"
        if idx == 0:
            return self.__head
        else:
            return self.__hidden[idx-1]
    
    def _init_weights(self, m):
        """
        Initialize the weights of layer using Xavier Uniform
        distribution and bias of layer to 0
        
        Args:
            m: layer of the model
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass in the model given the input x (state)
        
        Args:
            x: states of the agent
            
        Return:
            q_values for each action given the state
        """
        # Forward pass the input
        x = self.__head(x)
        
        for layer in self.__hidden:
            x = layer(self.__act(x))
        
        # Get the q-value for action
        return x
    
class Dueling(nn.Module):
    """
    Simple Linear Model with additional branch for estimating the v function
    to guide the approximation of q-values for each action. This dual stream
    architecture forms the Dueling architecture in variant of DQN.
    """
    def __init__(self, in_channel, out_channels=[], activation='relu'):
        """
        Class constructor for creating class instance
        
        Args:
            in_channel (int): Number of channels of input tensor (values representing state)
            out_channels (List[int]): list of number of channels in each hidden layer, including last layer
            activation (string): Activation function to be used in the model
        """
        assert isinstance(out_channels, (list, tuple)), \
            "Provide neurons of hidden layers in a list/tuple"
        assert isinstance(in_channel, int), \
            "Povide input channel as integer"

        if out_channels:
            assert len(out_channels) > 2, \
                "Must have at least 2 hidden layers"
            assert out_channels[-1] == 4, \
                "Last layer should have 4 neurons, corresponding to 4 actions"
        
        super().__init__()
        self.in_channel = in_channel
        self.__act = get_activation(activation)

        # Instantiate the layers of the model
        if out_channels:
            self.__head = nn.Linear(in_channel, out_channels[0])
            self.__hidden = nn.ModuleList([nn.Linear(out_channels[i], out_channels[i+1]) for i in range(len(out_channels)-1)])
            self.__hidden_v = nn.ModuleList([nn.Linear(out_channels[i], out_channels[i+1]) for i in range(len(out_channels)-2)])
            self.__hidden_v.append(nn.Linear(out_channels[-2], 1))
        else:
            self.__head = nn.Linear(in_channel, 4)
            self.__hidden = nn.ModuleList([
                nn.Linear(4, 8),
                nn.Linear(8, 4)
            ])
            self.__hidden_v = nn.ModuleList([
                nn.Linear(4, 8),
                nn.Linear(8, 1)
            ])
        
        # Initialize the weights of the layer
        self.apply(self._init_weights)
    
    def __getitem__(self, idx):
        """
        Retrieve particular linear layer of the model
        using indexing
        
        Args:
            idx (int): Index of the layer
        """
        assert isinstance(idx, int), "Index must be integer"
        assert -1 < idx < len(self.__hidden) + len(self.__hidden_v) + 1, "Only positive indices accepted"
        if idx == 0:
            return self.__head
        elif idx < len(self.__hidden) + 1:
            return self.__hidden[idx-1]
        else:
            return self.__hidden_v[idx-len(self.__hidden)-1]
    
    def _init_weights(self, m):
        """
        Initialize weights of the layer using Xavier uniform
        distribution and bias of the layer to 0
        
        Args:
            m: layer of the model
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass in the model given the input x (state)
        
        Args:
            x: states of the agent
            
        Return:
            q_values for each action given the state
        """
        x = self.__head(x)
        value = adv = x
        
        # Obtain advantage value
        for layer in self.__hidden: adv = layer(self.__act(adv))
        
        # Obtain action value
        for layer in self.__hidden_v: value = layer(self.__act(value))
        
        # Calculate the mean value
        value = value.expand(x.size(0), 4)
        adv_hat = adv.mean(1).unsqueeze(1).expand(x.size(0), 4)
        return value + adv - adv_hat

class QNetwork(nn.Module):
    """
    A wrapper for Deep Q-Network using the pre-defined models
    """
    def __init__(self, model, tau=0):
        """
        Class constructor for initializing instance attributes
        
        Args:
            model (nn.Module): Either Baseline or Dueling implemented above
            tau (float): control the weightage of update in target network
        """
        assert tau < 1, "Must have update on target network"
        super().__init__()
        
        # Use for soft updates in target
        self.__tau = tau
        
        # Set 2 models for prediction and target
        self.__predict_model = model
        self.__target_model = deepcopy(model)
        
        # Only train the prediction model
        self.__predict_model.train()
        self.__target_model.eval()
    
    def __getitem__(self, key):
        """
        Allows backward compatibility for extracting q-value using
        tuple (state, action)
        
        Args:
            key (tuple): state-action value pair
        """
        state, action = key
        pred = self(state, False)
        return QInfo(pred[Movement.cls_to_idx(action)])
    
    def forward(self, x, target=False):
        """
        Forward pass with specified model
        
        Args:
            x: state of the agent
            target: flag to use target model
            
        Return:
            q_values for each actions given the states
        """
        if not isinstance(x, torch.Tensor):
            x = self.__convert_to_tensor(x)

        # Use predict network
        if not target:
            result = self.__predict_model(x)
        # Use target network
        else:
            with torch.no_grad():
                result = self.__target_model(x)
        return result
    
    def update_target(self):
        """
        Update the target network weights to match with that of prediction
        network. Soft update is used here to ensure a smooth and stable
        training process.
        """
        # Get the parameters of the model
        target = self.__target_model.state_dict()
        predict = self.__predict_model.state_dict()
        
        # Soft update based on pre-defined tau value
        for key in target:
            target[key] = (1 - self.__tau) * predict[key] + self.__tau * target[key]
        
        # Update the parameters
        self.__target_model.load_state_dict(target)
    
    @staticmethod
    def __convert_to_tensor(x):
        """
        Convert the state to tensor object.
        
        NOTE: Assume the learning algorithm does not apply any transformation
        onto the input state
        
        Args:
            x: state of the agent
        Returns:
            tensor of the state
        """
        x = torch.tensor(x, dtype=torch.float)
        if x.dim() == 1: x = x.unsqueeze(0)
        return x
    
    def max_action(self, state, actions):
        """
        Return the best action (with highest q-value) given the
        state of the agent and allowable actions of the state.
        
        Args:
            state: state of the agent
            actions: allowable actions in the state
        Return
            Best movement in the allowable actions (with highest q-value)
        """
        if not isinstance(state, torch.Tensor):
            state = self.__convert_to_tensor(state)
        
        # Return the action based on allowable actions
        q_values = self(state).squeeze(0).tolist()
        action_value_pair = [(Movement.idx_to_cls(idx), value) for idx, value in enumerate(q_values)]
        action_value_pair = max(filter(lambda x: x[0] in actions, action_value_pair), key=lambda x: x[1])
        return action_value_pair[0]
    
    def max_value(self, state):
        """
        Return the maximum q-value of the given state. It assumes that
        the agent could perform all the action in the action space
        in the current position

        Args:
            state: The current state for which we want to retrieve the 
                    maximum Q-value among all possible actions.
        """
        if not isinstance(state, torch.Tensor):
            state = self.__convert_to_tensor(state)
        return self(state).max(1).float()

    def save(self, fname=None, fname_prefix=""):
        """
        Save the model as .pth file to be loaded next time
        
        Args:
            fname (string): filename of the saved model
            fname_prefix (string): prefix to be added to the filename
        """
        # Use current datetime if not filename specified
        if fname is None:
            fname = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        fname = fname_prefix + fname
        
        # save both target and predict network
        torch.save(self.__predict_model.state_dict(), f"{fname}_predict_model.pth")
        print(f"saved predict model to {fname}_predict_model.pth")
        torch.save(self.__target_model.state_dict(), f"{fname}_target_model.pth")
        print(f"saved target model to {fname}_target_model.pth")
        
    def load(self, path):
        """
        Load the pretrained model from .pth file using the path provided
        
        Args:
            path (string): File path, up to the filename + prefix, pointing to the saved model
        """
        predict_model_path = f"{path}_predict_model.pth"
        assert osp.exists(predict_model_path), f"path: {predict_model_path} does not exist"
        target_model_path = f"{path}_target_model.pth"
        assert osp.exists(target_model_path), f"path: {target_model_path} does not exist"
        self.__predict_model.load_state_dict(torch.load(predict_model_path))
        self.__target_model.load_state_dict(torch.load(target_model_path))