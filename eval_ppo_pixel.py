#!/usr/bin/env python3

# test ppo1-trained CNN agent on pixel version of the task

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import numpy as np
import argparse
import gym
import slimevolleygym
from slimevolleygym import FrameStack, render_atari

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv, WarpFrame
from stable_baselines import PPO1

from time import sleep

SEED = 831

RENDER_MODE = True

viewer = None
RENDER_ATARI = True # Render the game using the actual downsampled 84x84x4 greyscale inputs

cv2 = None
rendering = None
if RENDER_ATARI or RENDER_MODE:
  import cv2
  from gym.envs.classic_control import rendering as rendering

def make_env(seed):
  env = gym.make("SlimeVolleyNoFrameskip-v0")
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndSkipEnv(env, skip=4)
  env = WarpFrame(env)
  env = FrameStack(env, 4)
  env.seed(seed)
  return env

def rollout(env, model):
  obs = env.reset()
  if RENDER_MODE:
    env.render()
  cumulative_reward = 0
  done = False
  while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    cumulative_reward += reward
    if RENDER_MODE:
      env.render()
    if RENDER_ATARI:
      viewer.imshow(render_atari(obs))
    if RENDER_MODE or RENDER_ATARI:
      sleep(0.08)

  return cumulative_reward

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Evaluate pre-trained PPO1 CNN agent.')
  parser.add_argument('--model-path', help='path to stable-baselines model.',
                        type=str, default="zoo/ppo_cnn/best_model.zip")
  parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
  args = parser.parse_args()

  SEED = args.seed
  env = make_env(SEED)
  model = PPO1.load(args.model_path)

  if RENDER_ATARI:
    viewer = rendering.SimpleImageViewer(maxwidth=2160)

  rewards = []
  for i in range(1000):
    cumulative_reward = rollout(env, model)
    print(i, cumulative_reward)
    rewards.append(cumulative_reward)

  print("mean", np.mean(rewards))
  print("stdev", np.std(rewards))

  env.close()
  if RENDER_ATARI:
    viewer.close()
