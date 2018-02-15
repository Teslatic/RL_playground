def make_epsilon_greedy_policy(estimator, epsilon, nA):
  """
  Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

  Args:
      estimator: An estimator that returns q values for a given state
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.

  Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.

  """
  def policy_fn(sess, observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    q_values = estimator.predict(sess, observation)
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A
  return policy_fn

def q_learning(env, approx, num_episodes, max_time_per_episode, discount_factor=0.99, epsilon=0.1, use_experience_replay=False, batch_size=128, target=None):
  """
  Q-Learning algorithm for off-policy TD control using Function Approximation.
  Finds the optimal greedy policy while following an epsilon-greedy policy.
  Implements the options of online learning or using experience replay and also
  target calculation by target networks, depending on the flags. You can reuse
  your Q-learning implementation of the last exercise.

  Args:
    env: OpenAI environment.
    approx: Action-Value function estimator
    num_episodes: Number of episodes to run for.
    max_time_per_episode: maximum number of time steps before episode is terminated
    discount_factor: gamma, discount factor of future rewards.
    epsilon: Chance to sample a random action. Float betwen 0 and 1.
    use_experience_replay: Indicator if experience replay should be used.
    batch_size: Number of samples per batch.
    target: Slowly updated target network to calculate the targets. Ignored if None.

  Returns:
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
  """

  # Keeps track of useful statistics
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

  for i_episode in range(num_episodes):

	  # The policy we're following
	  policy = make_epsilon_greedy_policy(
	  estimator, epsilon, env.action_space.n)

	  # Print out which episode we're on, useful for debugging.
	  # Also print reward for last episode
	  last_reward = stats.episode_rewards[i_episode - 1]
	  print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
	  sys.stdout.flush()

	  # TODO: Implement this!

	  state = env.reset()

	  counter = 0
	  for t in range(max_time_per_episode):

		  action = np.random.choice(np.arange(len(policy(sess,state))),p=policy(sess,state))

		  state_next, reward_next, done, _ = env.step()
		  stats.episode_lengths[i_episode] = t
		  stats.episode_rewards[i_episode] += reward_next
		  q_values_next = approx.predict(sess, state_next)

		  if done:
			  td_target = reward_next
		  else:
			  td_target = reward_next + discount_factor*np.max(q_values_next)
			  approx.update(sess,state,action,td_target)

			  if done:
				  break
			  state = state_next


  return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)

if __name__ == "__main__":

    NUM_EPISODES= 3000
    MAX_STEPS_PER_EPISODE = 1000
    env = MountainCarEnv()
    approx = NeuralNetwork()
    target = TargetNetwork()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Choose one.
    #stats = q_learning(sess, env, approx, 3000, 1000)
    stats = q_learning(sess, env, approx, NUM_EPISODES, MAX_STEPS_PER_EPISODE)

    #stats = q_learning(sess, env, approx, 1000, 1000, use_experience_replay=True, batch_size=128, target=target)
    #plot_episode_stats(stats)

    for _ in range(100):
        state = env.reset()
        for _ in range(1000):
            env.render()
            state,_,done,_ = env.step(np.argmax(approx.predict(sess, state)))
            if done:
                break
