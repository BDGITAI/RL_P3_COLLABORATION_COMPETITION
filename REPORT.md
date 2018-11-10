# UDACITY REINFORCED LEARNING NANODEGREE: Project 3 Collaboration Competition  
---
# Report

## 1. Description

### Algorithm: MADDPG
The chosen algorithm is MADDPG (see [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275))

The MADDPG algorithm is based on the DDPG algorithm used in the 2nd project of the RL nanodegree. MADDPG implements sub-agent based on DDPG. Here the principal agent uses a central memory to store experiences shared by all agents. 
The learning is controlled from the MADDP agent.

### Actor Critic

The Actor Critic algorithm aims is at the crossroads of Value based method and Policy based method. In the first project, I implemented a DQN algorithm that associated a value to each pair (state, action). In that case the action space was both finite and discrete. Which is not the case in this new environment. Policy based method aimed a finding directly the best policy. However, with this method, by waiting until the end of the episode to compute the reward we may not see good actions if the episode was a failure.

The actor critic method implements two "brains" represented by two Neural Networks. The actor will decide of the best action to take while the critic assess independently if this was a good choice (see [Actor - Critic Algorithms](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf))

## 2. Hyperparameters

* MADDPG

Define the replay buffer size (number of experiences stored) . When the agent are untrained an episode is about 15 steps long.
The benchmark implementation performed by the Udacity team showed that the target score could be achieved with less 3000 episodes. To store all episodes buffer can be about 4.5e4 deep. However due to unstabilities I had to trained the network for longer and fixed the size to 1e5. 
`Replay(memory_size=int(1e5), batch_size=128)    # replay buffer size`

DDPG uses a local AC Network and a target AC Network for stability purposes. The target network weights are updated with local weights using soft update using parameter TAU (mixing parameter section 7 of DDPG paper 1e-3
Parametre from DDPG [publication](https://arxiv.org/abs/1509.02971)
`config.target_network_mix = 1e-3             # for soft update of target parameters`

Learning rate used for optimiser. Value from research paper 1e-4 for actor and 1 e-3 for critic
`actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3)))               # learning rate `

GAMMA used in expected reward computation. 
`    config.discount = 0.99           # discount factor`

* Optimizer

Adam optimizer is chosen over SGD as it converges faster in early stages of training (see [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/pdf/1712.07628.pdf)

`torch.optim.Adam(params, lr=1e-4)`

## 3. Result
The environment is considered solved when the agent is able to reach a score of 0.5 over 100 episodes.

### 3.1 Diverse network and unstability

In a first attempt 

### 3.2 Bootstrapping the memory

### 3.3 Single network

## 3. Improvement

To accelerate training, we could try to reduce the length of the episode to find the optimum value that would give enough experience while keeping the length to a minimum.
We could also experiment with another algorithm. In the paper [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778.pdf), TNPG and TRPO algorithm are shown to perform better than DDPG. 


