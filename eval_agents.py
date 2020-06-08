"""
Multiagent example.

Evaluate the performance of different trained models in zoo against each other.

This file can be modified to test your custom models later on against existing models.

Model Choices
=============

BaselinePolicy: Default built-in opponent policy (trained in earlier 2015 project)

baseline: Baseline Policy (built-in AI). Simple 120-param RNN.
ppo: PPO trained using 96-cores for a long time vs baseline AI (train_ppo_mpi.py)
cma: CMA-ES with small network trained vs baseline AI using estool
ga: Genetic algorithm with tiny network trained using simple tournament selection and self play (input x(train_ga_selfplay.py)
random: random action agent
"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import os
import numpy as np
import argparse
import slimevolleygym
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite # simple pretrained models
from slimevolleygym import BaselinePolicy
from time import sleep

#import cv2

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

PPO1 = None # from stable_baselines import PPO1 (only load if needed.)
class PPOPolicy:
  def __init__(self, path):
    self.model = PPO1.load(path)
  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action

class RandomPolicy:
  def __init__(self, path):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

def makeBaselinePolicy(_):
  return BaselinePolicy()

def rollout(env, policy0, policy1, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs0 = env.reset()
  obs1 = obs0 # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  #count = 0

  while not done:

    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.step(action0, action1)
    obs1 = info['otherObs']

    total_reward += reward

    if render_mode:
      env.render()
      """ # used to render stuff to a gif later.
      img = env.render("rgb_array")
      filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
      cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      count += 1
      """
      sleep(0.01)

  return total_reward

def evaluate_multiagent(env, policy0, policy1, render_mode=False, n_trials=1000, init_seed=721):
  history = []
  for i in range(n_trials):
    env.seed(seed=init_seed+i)
    cumulative_score = rollout(env, policy0, policy1, render_mode=render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)
  return history

if __name__=="__main__":

  APPROVED_MODELS = ["baseline", "ppo", "ga", "cma", "random"]

  def checkchoice(choice):
    choice = choice.lower()
    if choice not in APPROVED_MODELS:
      return False
    return True

  PATH = {
    "baseline": None,
    "ppo": "zoo/ppo/best_model.zip",
    "cma": "zoo/cmaes/slimevolley.cma.64.96.best.json",
    "ga": "zoo/ga_sp/ga.json",
    "random": None,
  }

  MODEL = {
    "baseline": makeBaselinePolicy,
    "ppo": PPOPolicy,
    "cma": makeSlimePolicy,
    "ga": makeSlimePolicyLite,
    "random": RandomPolicy,
  }

  parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
  parser.add_argument('--left', help='choice of (baseline, ppo, cma, ga, random)', type=str, default="baseline")
  parser.add_argument('--leftpath', help='path to left model (leave blank for zoo)', type=str, default="")
  parser.add_argument('--right', help='choice of (baseline, ppo, cma, ga, random)', type=str, default="ga")
  parser.add_argument('--rightpath', help='path to right model (leave blank for zoo)', type=str, default="")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)
  parser.add_argument('--day', action='store_true', help='daytime colors?', default=False)
  parser.add_argument('--pixel', action='store_true', help='pixel rendering effect? (note: not pixel obs mode)', default=False)
  parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
  parser.add_argument('--trials', help='number of trials (default 1000)', type=int, default=1000)

  args = parser.parse_args()

  if args.day:
    slimevolleygym.setDayColors()

  if args.pixel:
    slimevolleygym.setPixelObsMode()

  env = gym.make("SlimeVolley-v0")
  env.seed(args.seed)

  render_mode = args.render

  assert checkchoice(args.right), "pls enter a valid agent"
  assert checkchoice(args.left), "pls enter a valid agent"

  c0 = args.right
  c1 = args.left

  path0 = PATH[c0]
  path1 = PATH[c1]

  if len(args.rightpath) > 0:
    assert os.path.exists(args.rightpath), args.rightpath+" doesn't exist."
    path0 = args.rightpath
    print("path of right model", path0)

  if len(args.leftpath):
    assert os.path.exists(args.leftpath), args.leftpath+" doesn't exist."
    path1 = args.leftpath
    print("path of left model", path1)

  if c0.startswith("ppo") or c1.startswith("ppo"):
    from stable_baselines import PPO1

  policy0 = MODEL[c0](path0) # the right agent
  policy1 = MODEL[c1](path1) # the left agent

  history = evaluate_multiagent(env, policy0, policy1,
    render_mode=render_mode, n_trials=args.trials, init_seed=args.seed)

  print("history dump:", history)
  print(c0+" scored", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3), "vs",
    c1, "over", args.trials, "trials.")
