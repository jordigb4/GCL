import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """
    Neural network-based policy.

    Methods:
        forward(s):
            Performs a forward pass through the network.
        
        predict_probs(states):
            Predicts the action probabilities for given states.
        
        generate_session(env, t_max=1000):
            Generates a trajectory by interacting with the environment.
    """

    def __init__(self, state_shape, n_actions):
        """
        Initializes the Policy class with the given state shape and number of actions.
        NN consists of 2 hidden layers with 128 and 64 neurons, respectively. 

        Args:
            state_shape (tuple): The shape of the input state.
            n_actions (int): The number of possible actions.
        """
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.L1 = nn.Linear(in_features=state_shape[0], out_features=128)
        self.L2 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=n_actions)


    def forward(self, s):
        """
        Performs a forward pass through the network.

        Args:
            s (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output logits. Logits are the raw scores output by the last layer that
            represent the unnormalized probabilities of the actions. To get a probability distribution,
            we should apply the softmax function to the logits and to get the log_prob we should apply
            the log_softmax function.
        
        Raises:
            ValueError: If the input states contain NaN or Inf values.
        """
        x = F.relu(self.L1(s))
        x = F.relu(self.L2(x))
        logits = self.out(x)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Input states contain NaN or Inf values.")
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
    
        return probs, logprobs


    def predict_probs(self, states):
        """
        Predicts the action probabilities for given states.

        Args:
            states (numpy.ndarray): The input states.

        Returns:
            numpy.ndarray: The predicted action probabilities.
        
        """
        states = torch.FloatTensor(states)
        probs, _ = self.forward(states)
        return probs.detach().numpy()
    

    def generate_session(self, env, t_max=1000):
        """
        Generates a trajectory by interacting with the environment.

        Args:
            env (gym.Env): The environment to interact with.
            t_max (int, optional): The maximum number of steps in the session. Defaults to 1000.

        Returns:
            tuple: A tuple containing lists of states, action probabilities, actions, and rewards 
            generated in the trajectory. Notice action probabilities are necessary for learning using 
            the policy gradient method.
        """
        states, traj_probs, actions, probs, rewards = [], [], [], [], []
        s, _ = env.reset()

        for t in range(t_max):
            ## and choose one action according to the probabilities

            # Predict action probabilities
            action_probs = self.predict_probs(np.array([s]))[0]

            # Sample action
            a = np.random.choice(self.n_actions,p=action_probs)

            new_s,r,term,trunc,_ = env.step(a)
            done = term or trunc

            states.append(s)
            actions.append(a)
            probs.append(action_probs[a])
            rewards.append(r)

            s = new_s
            if done:
                break

        return states, probs, actions, rewards
