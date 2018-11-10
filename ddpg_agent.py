#######################################################################
#                                                                     #
# Original file provided in RL UDACITY NANODEGREE with ddpg-pendulum  #
# for the pendulum open ai gym environment                            #
#                                                                     #  
#######################################################################

#######################################################################
#                                                                     #  
#  Modifications to original file :                                   #
#  1. Global variables integrated as parameters in init functional    #
#  2. DDPG agent memory is not initialised.This agent is used by      #
#        maddpg who owns a central memory                             #
#  3. step function is removed. It is implemented at maddpg level     #
#  4. learn function is updated. It now needs to considerate          #
#        additional inputs related to actions taken by other agents   # 
#        in the environment                                           #
#  5. Removed the ReplayBuffer class and added maddpg.py              #
#######################################################################

import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, hidden_in_actor, hidden_out_actor, action_size, all_state_size, hidden_in_critic, hidden_out_critic, all_action_size,random_seed=0.0,lr_actor=1.0e-4, lr_critic=1.0e-3, tau =1.0e-3 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.tau = tau

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=hidden_in_actor, fc2_units=hidden_out_actor).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=hidden_in_actor, fc2_units=hidden_out_actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(all_state_size, all_action_size, random_seed, fcs1_units=hidden_in_critic, fc2_units=hidden_out_critic).to(device)
        self.critic_target = Critic(all_state_size, all_action_size, random_seed, fcs1_units=hidden_in_critic, fc2_units=hidden_out_critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0.0)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        # The memory is now centralised in maddpg
        self.memory = None
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        # clip action compatible with Unity env where actions are in the range [-1,1]
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma,all_next_actions,all_pred_actions,agent_id=0):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            all_next_actions : actions of all agents computed using the actor_target network and next actions
            all_pred_actions : actions of all agents computed using the actor_local network and actions
            agent_id : integer. Id of the agent. Used to get
        """
        states, actions, rewards, next_states, dones = experiences
        #cast to tensor as tensor select param can only be tensor
        id = agent_id
        agent_id = torch.tensor([agent_id]).to(device)
                
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        # Critic take all actions from all actor as input for MADDPG. 
        # Hence concatenate actions to adapt dimensions
        # As actions from other actors need to be known the call to target actor is 
        # done in MADDPG . DDPG only use the input param
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        # Rewards contain rewards from all agents. Select this agents rewards
        agent_rewards = rewards.index_select(1, agent_id)
        # same with terminals
        agent_dones = dones.index_select(1, agent_id)
        Q_targets = agent_rewards + (gamma * Q_targets_next * (1 - agent_dones))
        # Compute critic loss
        # Actions is already a vector concatenating actions from all agents 
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_optimizer.zero_grad()
        # action pred computed using actor local in MADDPG as we need to use all agents
        # adapt dimensions to adapt to batch size
        actions_pred = torch.cat(all_pred_actions, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

