# UDACITY REINFORCED LEARNING NANODEGREE: Project 3 Collaboration Competition  
---
# Report

## 1. Description

### Algorithm: MADDPG
The chosen algorithm is MADDPG (see [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275))

The MADDPG algorithm is based on the DDPG algorithm used in the 2nd project of the RL nanodegree. MADDPG implements sub-agent based on DDPG. Here the principal agent uses a central memory to store experiences shared by all agents. 
The learning is controlled from the MADDP agent.

### Actor Critic

The Actor Critic algorithm aims is at the crossroads of Value based method and Policy based method. In the first project, I implemented a DQN algorithm that associated a value to each pair (state, action). In that case the action space was both finite and discrete. Which is not the case in this new environment. Policy based method aimed at finding directly the best policy. However, with this method, by waiting until the end of the episode to compute the reward we may not see good actions if the episode was a failure.

The actor critic method implements two "brains" represented by two Neural Networks. The actor will decide of the best action to take while the critic assess independently if this was a good choice (see [Actor - Critic Algorithms](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf))

## 2. Hyperparameters

* MADDPG

Define the replay buffer size (number of experiences stored) . When agents are untrained an episode is about 15 steps long.
The benchmark implementation performed by the Udacity team showed that the target score could be achieved with less 3000 episodes. To store all episodes, the buffer can be about 4.5e4 deep. However, due to instabilities I had to trained the network for longer and fixed the size to 1e5. 
`Replay(memory_size=int(1e5), batch_size=128)    # replay buffer size`

DDPG uses a local AC Network and a target AC Network for stability purposes. The target network weights are updated with local weights using soft update using parameter TAU (mixing parameter section 7 of DDPG paper 1e-3
Parameter from DDPG [publication](https://arxiv.org/abs/1509.02971)
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

### 3.1 Diverse network and instability

In a first attempt I implemented a MADDPG agent that contained 2 distinct DDPG agent. As a consequence, the MADDPG had 2 different actors and each one had their respective critic. At first the training was unstable with an average score increasing up to 0.1 or 0.2 and then collapsing without recovery.
The figure below shows an attempt where after 10 000 episodes the result still did not converge.

![result1](/images/first_attempt.jpg)

I thought of two the reasons that could explain the instability :
* When the agent is not trained the number of positive rewards is very low. As a result, the signal generated can be weak and prevent the algorithm from converging towards a solution. 
* The two DDPG actors need to progress in parallel. If one agent improves and the other not, then the ball never comes back. This lack of improvement from one of the agents prevent the progression of other by limiting the maximum score to 0.1 

### 3.2 Bootstrapping the memory

To test my first hypothesis, I created a function that would pre-fill the MADDPG buffer. While interacting with the environment using a random policy for the actions, experiences are stored using a simple selection criteria. 
I set a ratio of positive, negative and neutral memories. 
To maintain the balance between the experiences I allow the neutral memories to reach 40% of the pre-fill. It helps in increasing the number of stored negative experiences to 40%. After 10 000 episodes the buffer contained : 15% of positive memories, and 40% for each of the other two categories.
The figure below shows that after 1000 episodes, the agent starts to learn steadily until it reaches an average of 0.52 after 2912 episodes.

![result2](/images/second_attempt.jpg)

### 3.3 Single network

Even though the previous hypothesis worked I investigated the influence of using a single DDPG agent within the MADDPG. This DDPG would act twice for each step. In fact, it would play against himself.
With no pre-filled memory, we can see in the figure below that the algorithm was quite unstable. In several occasions the algorithm progressed and then collapsed until it reached the target goal after 7571 episodes

![result3](/images/third_attempt.jpg)

I then combined the single network agent with the pre-filled memory and achieved the best performance where the agent obtained an average of 0.52 after 1768 episodes

![result4](/images/fourth_attempt.jpg)

In evaluation mode over 10 episodes, this agent was able to achieve an average score of 2.16 with a maximum at 2.7

![success](/images/tennis.gif)

## 3. Improvement

To accelerate training, we could try to reduce the complexity of the network and see if we could achieve the same performance
We could also experiment an adaptative learning rate for the optimiser. We could decrease the learning rate when the algorithm reaches a certain score and see if it would avoid the collapsing observed during the training.
Trying to reuse this implementation to solve a soccer game with more agents and different teams would also allow searching for new ways of improving this project


