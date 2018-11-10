# UDACITY REINFORCED LEARNING NANODEGREE: Project 3 Collaboration Competition  
---
# Report

  
## 1. Single Agent Training 

### Description

#### Algorithm: MADDPG
The chosen algorithm is DDPG (see [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971))

The DDPG algorithm is an Actor Critic method built on the principles used in Deep Q learning (algorithm use in my first project). It uses a memory to store experiences and learned from them using a random sampling, thus breaking the correlation between consecutive experiences. It also uses a target network to help with the error computation stability. Instead of using a direct weight update a soft target update is used during the learning.

*Replay* 

`self.replay.feed([self.state, action, reward, next_state, int(done)])`

*Soft update* 

`   def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)`

#### Actor Critic

The Actor Critic algorithm aims is at the crossroads of Value based method and Policy based method. In the first project, I implemented a DQN algorithm that associated a value to each pair (state, action). In that case the action space was both finite and discrete. Which is not the case in this new environment. Policy based method aimed a finding directly the best policy. However, with this method, by waiting until the end of the episode to compute the reward we may not see good actions if the episode was a failure.

The actor critic method implements two "brains" represented by two Neural Networks. The actor will decide of the best action to take while the critic assess independently if this was a good choice (see [Actor - Critic Algorithms](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf))


We can see the two networks being trained in by the agent while he steps:

`critic_loss.backward()`
`policy_loss.backward()`

### Hyperparameters

* DDPG

Define the replay buffer size (number of experiences stored) Increased compared to default to help with training
`Replay(memory_size=int(1e7), batch_size=64)    # replay buffer size`

DDPG uses a local AC Network and a target AC Network for stability purposes. The target network weights are updated with local weights using soft update using parameter TAU (mixing parameter section 7 of DDPG paper 1e-3

`config.target_network_mix = 1e-3             # for soft update of target parameters`

Learning rate used for optimiser. Value from research paper 1e-4 for actor and 1 e-3 for critic

`actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3)))               # learning rate `

GAMMA used in expected reward computation. 
`    config.discount = 0.99           # discount factor`

* Optimizer

Adam optimizer is chosen over SGD as it converges faster in early stages of training (see [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/pdf/1712.07628.pdf)

`torch.optim.Adam(params, lr=1e-4)`

### Result
The environment is considered solved when the agent is able to reach a score of 30 over 100 episodes.
In a first attempt the agent was trained over 2000 episodes reaching an average score of 16. ![single_agent_2000](./images/single_agent_1_2000.png)

I noticed the length of the episode compared to the memory could be an issue. At first he memory size was 1e6 with an episode lasting 1e4 steps only the latest 100 episodes were stored in the memory. The better the algorithm got the more similar the memory would be for the same episodes. This meant that the random sampling over 1e6 episodes would almost be equivalent to a random sampling over 100. 

With a memory size increase to 1e7, the agent was trained over 2000 episodes reaching an average score of 23. 
The figure below shows that after the 1400th episode the results improve but reach a plateau.  ![single_agent_2000](./images/single_agent_2_2000.png)

I reduced the length of an episode by bounding the number of time steps to 500. 
The agent achieved an average of 16 after 1630 episodes. When running evaluations the agent was able to perform with an average of 33.
 ![single_agent_solved](./images/single_agent_3_solved.png)
 

## 2. Multiple agent training

Intuitively, one can learn from others' experiences. If you see someone fall on a tile flooring you would approach the area carefully as you would assume the area is slippery.
By training different agent in parallel, we will be able to gather quickly uncorrelated data and then distribute the knowledge to all agents for them to improve. In the multi agent Unity environment we benefit from having 20 different agents acting in their own "sphere". At the end of one episode we would have 20 times more experiences gathered compared to our single agent environment.

As explained in the Jupyter notebook, the Unity environment still require the 20 actions to be passed when stepping it. It means all agents need to decide what action to take before the environment is stepped. Due to this constraint agents are not implemented in separate thread but are called consecutively.

With one agent we saw that reducing the length of the episode and increasing the memory size would improve learning. That is why we start training the multi agent with same parameters

As result a score of 16 is achieved within 131 episodes.

![multi_agent](./images/multiple_para_agent.png)

## 3. Improvement

To accelerate training, we could try to reduce the length of the episode to find the optimum value that would give enough experience while keeping the length to a minimum.
We could also experiment with another algorithm. In the paper [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778.pdf), TNPG and TRPO algorithm are shown to perform better than DDPG. 


