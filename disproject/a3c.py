import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# AC3 is the actual on-policy algorithm to generate the policy π
# Used for environments with discrete action spaces --> AC3
# This class is going to inherit from the nn module
class ActorCritic(nn.Module):
    #The __init__ method lets the class initialize the object’s attributes
    # tau is the constant lamda from the paper
    def __init__(self, input_dims, n_actions, gamma= 0.0, tau= 0.98):
        # super() corresponds to nn.Module and it is running the initialisation for the nn.Module as well as (self)
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.tau = tau
        # Our network will need an input layer which will take an input and translate that into 256
        self.input = n.Linear(*input_dims, 256)
        # A dense layer
        self.dense = nn.Linear(256, 256)
        # Lstm type layer
        self.gru = nn.GRUCell(256, 256)
        # Policy
        self.pi = nn.Linear(256, n_actions)
        # Value function
        self.v = nn.Linear(256, 1)
    
    # It will take a state and a hidden state for our GRU as an input
    def forward(self, state, hx):
        x = F.relu(self.input(state))
        x = F.relu(self.dense(x))
        hx = self.gru(x, (hx))
        
        # Pass hidden state into our pi and v layer to get our logs for our policy and out value function
        pi = self.pi(hx)
        v = self.v(hx)
        
        # Choose action function/ Get the actual probability distribution
        probs = T.softmax(pi, dim=1) # soft max activation on the first dimension
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.numpy([0]), v, log_prob, hx
    
    # Functions to handle the calculation of the loss
    # https://arxiv.org/pdf/1602.01783.pdf
    def calc_R(self, done, rewards, values): # done/terminal flag, set of rewards, set of values--> stored in a list of tensors
        # we want to convert this list of tensors to a single tensor and squeeze it because we dont want T time steps by 1
        values = T.cat(values).squeeze()
        # A3C must get triggered every T timestep or everytime an episode ends / we could have a batch of states or a single state
        # if we have batch of states then the length of values.size is one
        if len(values.size()) == 1: # batch of states
            # last value of the value array 
            # multiplied by (1- int(done)) because the value of the terminal state is identically 0
            R = values[-1] * (1- int(done))
        elif len(values.size()) == 0: # single state
            R = values*(1- int(done))
        
        # Calculate the returns at each time step of R sequence
        batch_return = []
        # Iterate backwards in our rewards
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)  
            
        batch_return.reverse() # reverse it
        # convert it to a tensor 
        batch_return = T.tensor(batch_return, dtype = T.float).reshape(values.size())
        return batch_return
    # r_i_t --> intrisic reward
    def calc_loss(seld, new_states, hx, done, rewards, values, log_probs, r_i_t = None):
        # if we are supplying an intrinsic reward them we want to add the reward from ICM
        if r_i_t is not None:
            rewards += r_i_t.detach().numpy() # convert r_i_t to a numpy array because r_i_t is a tensor while rewards is a list of floating point values
        returns = self.calc_R(done, rewards, values)
        # calculate generalised advantage
        # We need a value function for the state one step after our horizon
        # get the first element because other elements that the forward function returns are not the value function (we want the element v )
        next_v = T.zeros(1, 1) if done else self.forward(T.tensor([new_states],
                                         dtype=T.float), hx)[1]
        values.append( next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs) # concatinate -> cat
        rewards = T.tensor(rewards)
        #                   state of time at t+1  state of time at t
        delta_t = rewards + self.gamma*values[1:] - values[:-1]
        
        n_steps = len(delta_t)
        '''generalised advantage estimate : https://arxiv.org/pdf/1506.02438.pdf'''
        # There is gonna be an advantage for each time step in the sequence
        # So gae is gonna be a batch of states, T in length
        # So we have an advantage for each timestep, which is proportional to a sum of all the rewards that follow
        gae = np.zeros(n_steps)
        # For each step in the sequence
        for t in range(n_steps):
            # for from that step onwards to the end
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k*delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)
        
        # Calculate losses 
        actor_losses = -(log_probs*gae).sum()
        entropy_loss = (-log_probs*T.exp(log_probs)).sum()
        # bc we stuck with the value function for the state that occurs at T timestep
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns) # mean squared error
        total_loss = actor_loss + critic_loss - 0.01*entropy_loss
       
        return total_loss
                

        

