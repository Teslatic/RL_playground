#!/usr/bin/python3
from collections import defaultdict
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools
import pandas as pd
from PIL import Image
import time
if "../../" not in sys.path:
  sys.path.append("../../")

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
VALID_ACTIONS = [0, 1, 2]

def reinforce(policy, best_policy):
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

  for ep in range(num_episodes):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset()
    # print("generate episode {}/{}".format(ep+1,num_episodes))
    for t in range(500):
      action, prob_actions = policy.predict(sess,state)
      # action = np.random.choice(np.arange(len(prob_actions)), p=prob_actions)
      next_state, reward, done, _ = env.step(action)
      episode.append((state,action,reward))
      # cumulative reward per episode
      stats.episode_rewards[ep] += reward
      stats.episode_lengths[ep] = t
      if done:
          break

      state = next_state
    test_return = []
    for t, ep in enumerate(episode):
        s = ep[0]
        a = ep[1]
        r = ep[2]
        total_return = sum(discount_factor**i * r for i,t in enumerate(episode[t:]))
        policy.update(sess,s,a,total_return)
  return stats

# if __name__ == "__main__":
#   p = Policy()
#   bp = BestPolicy()
#   action_verbose = {0:"left",1:"nothing",2:"right"}
#   sess = tf.Session()
#   tf.global_variables_initializer().run(session=sess)
#   stats = reinforce(sess, env, p, bp, EPISODES)
#   success = 0
#   plot_episode_stats(stats)
#   # saver = tf.train.Saver()
#   # saver.save(sess, "./policies.ckpt")
#
#   for _ in range(5):
#     state = env.reset()
#     for i in range(500):
#       env.render()
#       # time.sleep(0.05)
#       chosen_action = p.predict(sess, state)[0]
#       print("chosen action: {}".format(action_verbose[chosen_action]))
#       # sys.stdout.flush()
#       _, reward, done, _ = env.step(chosen_action)
#       if done:
#         if reward==10:
#             success +=1
#         break
# print("success rate: {}%".format(success/EPISODES * 100))
