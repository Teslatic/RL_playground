import sys
import numpy as np
import itertools
from collections import defaultdict, namedtuple

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def make_epsilon_greedy_policy(Q, epsilon, nA):
  """
  Creates an epsilon-greedy policy based on a given Q-function and epsilon.
  
  Args:
    Q: A dictionary that maps from state -> action-values.
      Each value is a numpy array of length nA (see below)
    epsilon: The probability to select a random action . float between 0 and 1.
    nA: Number of actions in the environment.
  
  Returns:
    A function that takes the observation as an argument and returns
    the probabilities for each action in the form of a numpy array of length nA. 
  """
  
  def policy_fn(observation):
    act_prob = np.ones(nA, dtype=float)* epsilon/nA
    max_action = np.argmax(Q[observation])
    act_prob[max_action] += (1.0-epsilon)
    return act_prob
  return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
  """
  Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
  while following an epsilon-greedy policy
  
  Args:
    env: OpenAI environment.
    num_episodes: Number of episodes to run for.
    discount_factor: Lambda time discount factor.
    alpha: TD learning rate.
    epsilon: Chance the sample a random action. Float betwen 0 and 1.
  
  Returns:
    A tuple (Q, episode_lengths).
    Q is the optimal action-value function, a dictionary mapping state -> action values.
    stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """
  
  # The final action-value function.
  # A nested dictionary that maps state -> (action -> action-value).
  Q = defaultdict(lambda: np.zeros(env.action_space.n))

  # Keeps track of useful statistics
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
    episode_rewards=np.zeros(num_episodes))    
  
  # The policy we're following
  policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
  #print(num_episodes)
  for i_episode in range(num_episodes):
    # Print out which episode we're on, useful for debugging.
    if (i_episode + 1) % 100 == 0:
      print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
      sys.stdout.flush()
   
    # first action
    state = env.reset()
   
    # generate episode, i.e. 
    # 1.) choose action based on epsilon-greedy
    # 2.) take a step 
    # 3.) update statistics and TD error
    # -> do until done == 1
    # techically a DO-WHILE loop, that breaks when the done flag is raised  
    counter = 0
    while(True):        
        # take one step, based on teacher policy, here: epsilon greedy
        action = np.random.choice(np.arange(len(policy(state))), p=policy(state))
        next_state, reward, done, info = env.step(action)
        
        # Keeps track of useful statistics
        # count steps taken in i-th episode
        stats.episode_lengths[i_episode] = counter
        # accumulate reward in i-th episode -> total return
        stats.episode_rewards[i_episode] += reward
        
        # update temporal-difference
        Q[state][action] += alpha* (reward + discount_factor * 
                             Q[next_state][np.argmax(Q[next_state])]-Q[state][action])
 
        if done:
            break
        state = next_state
        counter += 1
    
  return Q, stats
