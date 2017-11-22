#!/usr/bin/python3

import unittest
import numpy as np
import sys
if "../../" not in sys.path:
  sys.path.append("../../")
from lib.envs.gridworld import GridworldEnv
if "../" not in sys.path:
  sys.path.append("../")
from scripts import policy_iteration
from scripts import value_iteration

def setUpModule():
  global env
  env = GridworldEnv()

class TestPolicyEval(unittest.TestCase):
  def test_policy_eval(self):
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_iteration.policy_eval(random_policy, env)
    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
    
#if __name__ == '__main__':
#  unittest.main()



global env
env = GridworldEnv()

random_policy = np.ones([env.nS, env.nA]) / env.nA
#v = policy_iteration.policy_eval(random_policy, env)
    
for i in range(0,10):
	policy, v = policy_iteration.policy_improvement(env)
