import torch as T
import torch.nn as nn
import torch.nn.functional as F

'''In the inverse model you want to predict the action the agent took to cause this state to transition from time t to t+1
So you are comparing an integer vs an actula label/ the actual action the agent took
Multi-class classification problem
This is a cross entropy loss between the predicted action and the actual action the agent took'''
"The loss for the forward model is the mse between the predicted state at time t+1 and the actua state at time t+1  "
"So we have two losses : one that comes from the inverse model and one that comes from the forward model "
class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=2, alpha=1, beta=0.2):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # hard-coded for the cartpole environment
        # Inverse takes two successive states and tries to predict one action
        # The input is gonna be 4*2 bc the input vector from the * environment has 4 elements, and we are gonna have two of them 
        self.inverse = nn.Linear(4*2, 256) # This is gonna feed to a layer of 256
        # The logits of our policy is what we want to predict 
        self.pi_logits = nn.LInear(256, n_actions)
        # Then the forward model takes a state and an action and asks what is the resulting state
        self.dense1 = nn.Linear(4+1) # so inputs are 4+1 : 4 for the environment and 1 for one action --> it fits into a layer of 256 units
        # state that we output
        self.new_state = nn.Linear(256, 4) # if not hard-coded 4 = *input_dims
        
        device = T.device('cpu')
        self.to(device)
        
        def forward(self, state, new_state, action):
            "We have to concatenate a state and action and pass it through the inverse layer "
            "and activate it with an elu activation--> exponential linear"
            # Create inverse layer
            inverse = F.elu(self.inverse(T.cat([state, new_state], dim=1)))
            pi_logits = self.pi_logits(inverse)
            
            # Forward model
            # from [T] to [T,1]
            action = action.reshape((action.size()[0], 1))
            forward_input = T.cat([state, action], dim=1)
            # Activate the forward input and get a new state on the other end
            dense = F.elu(self.dense1(forward_input))
            state_ = self.new_state
            
            return pi_logits, state_
        
        
        def calc_loss(self, state, new_state, action):
            state = T.tensor(state, dtype=T.float)
            action = T.tensor(action, dtype=T.float)
            new_state = T.tensor(new_state, dtype=T.float)
            # feed/pass state, new_state , action through our network
            pi_logits, state_ = self.forward(state, new_state, action)
            "Our inverse loss is a cross entropy loss because this will generally have more than two actions"
            inverse_loss = nn.CrossEntropyLoss()
            L_I = (1-self.beta)*inverse_loss(pi_logits, action.to(T.long))
            "Forward loss is mse between predicted new state and actual new state"
            forward_loss = nn.MSELoss()
            L_F = self.beta*forward_loss(state_, new_state)
            "dim=1 for mean(dim=1) is very important. If you take that out it will take the mean across all dimensions and you just get a single number, which is not useful"
            "because the curiosity reward is associated with each state, so you have to take the mean across that first dimension which is the number of states"
            intrinsic_reward = self.alpha*((state_ - new_state).pow(2)).mean(dim=1)
            return intrinsic_reward, L_I, L_F 


        
