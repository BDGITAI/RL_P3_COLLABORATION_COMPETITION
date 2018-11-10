#######################################################################
#                                                                     #
#  MADDPG class agent for Udacity Collaboration Competition project   #
#  Contains :                                                         #
#        - MADDPG class : create a MADDPG agent that is made of       #
#          several DDPG agent. Contains function to interact with     #   
#          environment and to call DDPG functions to act and learn    #
#        - ReplayBuffer class : Memory buffer used at MADDPG level.   #
#          All DDPG agent use the same memory                         #
#                                                                     #
#######################################################################

from ddpg_agent import DDPG_Agent
import torch
import numpy as np
from collections import namedtuple, deque
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, observation_dim, action_dim, num_agent, buffer_size=int(1e5), batch_size=64, discount_factor=0.99, tau=0.001):
        """Initialize parameters and build model.
        Params
        ======
            observation_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            num_agent (int): Number of DDPG agent used to interact with environment
            buffer_size (int): Number of experiences that can be stored in memory
            batch_size (int): Batch size used for learning
            discount_factor (int): Discount factor used to compute Q_targets
            tau (float) : mixing param for soft update
        """        
        super(MADDPG, self).__init__()
        self.num_agent = num_agent
        self.seed = 0.0
        self.action_dim = action_dim
        self.observation_dim = observation_dim 
        self.discount_factor = discount_factor
        # number of stepping iteration. Used to learn at given intervals
        self.iter = 0
        # when not trained an episode is about 13 steps (time for ball to fall)
        self.update_every_iter = 10 
        self.buffer_size = buffer_size
        # Implementation only using an agent playing against itself
        # Goal for each tennis competitor is the same. Same actions.
        # First attempt included 2 different DDPG_Agent but learning was unstable
        # Changed to only one agent and achieved more stable learning
        self.ddpg_agents = []
        self.temp_agent = DDPG_Agent(state_size= observation_dim,       # input layer for actor take obs dim
                                        hidden_in_actor=400,            # first HL with 400 : ref MADDPG paper
                                        hidden_out_actor=300,           # second HL with 300 : ref MADDPG paper
                                        action_size=action_dim,         # output of actor is the action 
                                        all_state_size=observation_dim*num_agent, # used to size critic input layer
                                        hidden_in_critic=400,
                                        hidden_out_critic=300, 
                                        all_action_size=action_dim*num_agent,# used to size critic input layer
                                        lr_actor=1.0e-4,                # optim learning rate. value as per MADDPG paper
                                        lr_critic=1.0e-3,               # optim learning rate. value as per MADDPG paper
                                        tau =tau,                       # mixing param for soft update (target,local)
                                        random_seed =self.seed )
        #Add same agent twice to play against itself
        for i in range (num_agent):
            self.ddpg_agents.append(self.temp_agent)
        
        # create central memory
        self.memory = ReplayBuffer(buffer_size,batch_size,self.seed)


    def act(self, all_states, add_noise=True):
        """get actions from all agents in the MADDPG object
        Params
        ======
            all_states (Array of num agent length): Array returned by Unity environment
            add_noise (boolean): Activate noise for exploration. Set to False for evaluation
        """     
        actions=[]
        for id,agent in enumerate(self.ddpg_agents):
            # call to act of each ddpg_agent
            action = agent.act((all_states[id]),add_noise)
            actions.append(action)
        return actions
        
    def step(self, state, action, reward, next_state, done):
        """Save experiences in buffer and decide to learn
        Params
        ======
            All param are arrays of num agent length
            state ([floats]): Current states of all agents
            action ([floats]): Action taken at current time step by all agents
            reward ([floats]): all rewards for current time step
            next_state ([floats]): Next states returned by Unity for all agents
            done([floats]): indicate end of the current episodes
            
        """         
        # Reshape inputs for compatibility with buffer structure
        # also easier to concatenate while passing param to critic
        state = state.reshape(1, -1)  
        next_state = next_state.reshape(1, -1)  
        action= np.array(action).reshape(1, -1)
        # Add experiences into memory
        self.memory.add(state, action, reward, next_state, done)

        # Count number of time steps. Enables learning every "update_every_iter"
        # do not wait end of episode to learn
        self.iter = (self.iter + 1) % self.update_every_iter

        # If memory contains more experiences than required for a batch
        if self.iter == 0 and len(self.memory) > self.memory.batch_size:
            experiences = [] 
            # get different experiences for the different agent
            for i in range(self.num_agent):
                experiences.append(self.memory.sample())
            self.learn(experiences, self.discount_factor)
                   
    def learn(self, experiences, gamma):
        """Prepare some central data for the ddpg agent learning then call ddpg learning
        Params
        ======
            experiences (namedtuple): batch of previous experiences 
            gamma : discount factor used in Qtarget computation            
        """ 
        # Learn for each agent
        for id, agent in enumerate(self.ddpg_agents): 
            # cast to allow using the torch index select
            agent_id = torch.tensor([id]).to(device)
            # get experiences for the current agent
            agent_experiences = experiences [id]
            # only get states
            all_states, _, _, all_next_states, _ = agent_experiences
            #Q_targets_next computation is done using critic network which requires actions of all agents
            #Q_targets_next = self.critic_target(next_states, actions_next)
            actions_next = []
            for jd,target in enumerate(self.ddpg_agents) :
                # as all states are concatenated into the memory need to reshape to get only states for jd agent
                agent_next_states = all_next_states.reshape(-1, self.num_agent, self.observation_dim).index_select(1, agent_id).squeeze(1)
                actions_next.append(target.actor_target(agent_next_states))
            #actions_pred used to compute actor_loss through the critic
            #hence need to compute centrally actions.
            actions_pred = []
            for jd,local in enumerate(self.ddpg_agents) :
                agent_states = all_states.reshape(-1, self.num_agent, self.observation_dim).index_select(1, agent_id).squeeze(1)
                actions_pred.append(local.actor_local(agent_states))
                # need to detach actions for other agent to avoid double back propagation
                if jd != id:
                    actions_pred[jd] = actions_pred[jd].detach()
            agent.learn(agent_experiences,gamma,actions_next,actions_pred,id)

    def save(self,tag):
        """save ddpg network param
        Params
        ======
            tag (string): string to differentiate different save 
        """ 
        for i,agent in enumerate(self.ddpg_agents):
            torch.save(agent.actor_local.state_dict(), './'+ tag + str(i) + '.actor.pth')
            torch.save(agent.critic_local.state_dict(),'./'+ tag + str(i) + '.critic.pth')
            
    def load(self,actors_files,critics_files=None):
        """Load saved network param
        Params
        ======
            actors_files (array of strings): path to file containing weights 
            critics_files (array of strings): path to file containing weights .
        """ 
        for agent,file in zip(self.ddpg_agents,actors_files):
            actor_file = torch.load(file)
            agent.actor_local.load_state_dict(actor_file)
        # no need to init critic weights in evaluation
        if critics_files!=None:
            for agent,file in zip(self.ddpg_agents,critics_files):
                critics_file = torch.load(file)
                agent.critic_local.load_state_dict(critics_file)

                           
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)           
            




