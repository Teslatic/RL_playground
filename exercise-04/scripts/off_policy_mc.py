from collections import defaultdict
import numpy as np
import sys

def create_random_policy(nA):
  """
  Creates a random policy function.
  
  Args:
    nA: Number of actions in the environment.
  
  Returns:
    A function that takes an observation as input and returns a vector
    of action probabilities
  """
  A = np.ones(nA, dtype=float) / nA
  def policy_fn(observation):
    return A
  return policy_fn

def create_greedy_policy(Q):
  """
  Creates a greedy policy based on Q values.
  
  Args:
    Q: A dictionary that maps from state -> action values
      
  Returns:
    A function that takes an observation as input and returns a vector
    of action probabilities.
  """
  
  def policy_fn(observation):    
    act_prob = np.zeros_like(Q[observation], dtype=float)
    max_action = np.argmax(Q[observation])
    act_prob[max_action] = 1.0
    return act_prob
  return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Importance Sampling.
    Finds an optimal greedy policy.

    Args:
    env: OpenAI gym environment.
    num_episodes: Nubmer of episodes to sample.
    behavior_policy: The behavior to follow while generating episodes.
        A function that given an observation returns a vector of probabilities for each action.
    discount_factor: Lambda discount factor.

    Returns:
    A tuple (Q, policy).
    Q is a dictionary mapping state -> action values.
    policy is a function that takes an observation as an argument and returns
    action probabilities. This is the optimal greedy policy.
    """
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # C: denominator in the action value update step, has the same structure as the
    # action value function, so I will reuse the form from above
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
    
    # Implement this!
    
    ### loop over all episodes

    for i_episode in range(num_episodes):
        done = 0
        # states visited in current episode
        visited = []
        episode = []
        # make first move
        observation = env.reset()        
        # sample the start state for first visit check
        visited.append(observation)
        ## Generate one episode
        while done==0:            
            # act based on teacher policy ~> take random action from teacher
            probabilities = behavior_policy(observation)
            # acions are 0 or 1, here chosen uniformly.
            action = np.random.choice(2, p=probabilities)            
            observation,reward,done,info = env.step(action)
            
            # take only first visit into visited states list, save observation,actions,rewards
            # for update purpose 
            if(not(observation in visited)):
                # list to check history ~> first visit check
                visited.append(observation)
                # needed for later update steps
                episode.append((observation,action,reward))
            # debugging output: check what is passed down to update steps
            if i_episode % 1000 == 0:
                print("\r Episode {}/{}| packed episode: {}|  ".format(i_episode,num_episodes,episode, end=""))
                sys.stdout.flush()
        
        G = 0.0
        W = 1.0
       
        ## Update Q values from last to first entry, i.e. rewind the episode
        for t in range(len(episode))[::-1]:
            # unpack state, action and reward for handling update computations
            #print(episode[t])
            state, action, reward = episode[t]
            #if t % 1000 == 0:
                #print("\r state, action, reward {}".format(episode[t]),end="")
                #sys.stdout.flush()
            #print(state,action,reward)
            # update total return
            G = discount_factor*G + reward
            # update denominator term
            C[state][action] += W
            # update action value
            Q[state][action] += (W/C[state][action]) * (G-Q[state][action])
            # update W
           
            target_policy(state)[action] = np.argmax(Q[state][action])
           
            if action != target_policy(state)[action]:
                break
            W = W * (1.0/behavior_policy(state)[action])
            
                                                  
    return Q, target_policy
