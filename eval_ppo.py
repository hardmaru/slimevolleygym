"""
Simple evaluation example.

run: python eval_ppo.py --render

Evaluate PPO1 policy (MLP input_dim x 64 x 64 x output_dim policy) against built-in AI

"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse

import slimevolleygym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

def rollout(env, policy, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs = env.reset()

  done = False
  total_reward = 0

  while not done:

    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    total_reward += reward

    if render_mode:
      env.render()

  return total_reward

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate pre-trained PPO agent.')
  parser.add_argument('--model-path', help='path to stable-baselines model.',
                        type=str, default="zoo/ppo/best_model.zip")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

  args = parser.parse_args()
  render_mode = args.render

  env = gym.make("SlimeVolley-v0")

  # the yellow agent:
  print("Loading", args.model_path)
  policy = PPO1.load(args.model_path, env=env) # 96-core PPO1 policy

  history = []
  for i in range(1000):
    env.seed(seed=i)
    cumulative_score = rollout(env, policy, render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)

  print("history dump:", history)
  # this is what I got: [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 2, 1, 1, 1, 1, 4, 0, 0, 0, 2, 2, 0, 1, 2, 2, 2, 1, 2, 1, 0, 2, 0, 1, 1, 1, 0, 0, 1, 0, 1, 2, 0, 0, 1, 1, 4, 1, 0, 2, 2, 3, 2, 4, 4, 1, 1, 2, 0, 0, 0, 4, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 3, 2, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 5, 3, 3, 0, 0, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0, 1, 0, 1, 1, 2, 2, 1, 3, 4, 0, 0, 0, 3, 0, 1, 5, 2, 4, 0, 1, 1, 1, 3, 0, 1, 2, 1, 1, 2, 1, 1, 2, 0, 1, 1, 0, 1, 0, 1, 2, 0, 2, 0, 2, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 3, 0, 1, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 3, 2, 2, 1, 2, 0, 2, 2, 0, 1, 0, 1, 0, 0, 2, 1, 2, 1, 0, 2, 1, 0, 1, 0, 2, 1, 1, 1, 2, 2, 2, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 2, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 2, 0, 0, 1, 1, 0, 1, 2, 1, 0, 2, 3, 3, 4, 0, 0, 1, 0, 1, 1, 2, 0, 1, 0, 1, 0, 2, 1, 0, 3, 0, 0, 1, 1, 1, 2, 2, 0, 0, 2, 0, 0, 1, 2, 4, 0, 2, 0, 1, 1, 1, 0, 1, 2, 1, 0, 0, 4, 1, 0, 0, 0, 0, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 4, 2, 3, 0, 3, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 0, 0, 0, 1, 0, 2, 1, 2, 0, 1, 1, 2, 1, 0, 1, 0, 1, 1, 2, 0, 2, 0, 0, 1, 1, 0, 0, 2, 0, 2, 0, 1, 2, 2, 3, 1, 1, 0, 0, 1, 1, 4, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 3, 0, 0, 2, 2, 0, 3, 1, 0, 2, 0, 1, 0, 0, 2, 1, 2, 3, 1, 0, 1, 0, 1, 2, 1, 0, 2, 0, 0, 1, 0, 0, 1, 1, 1, 0, 2, 1, 0, 2, 2, 0, 1, 0, 1, 0, 5, 2, 2, 0, 1, 2, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 2, 2, 1, 0, 1, 1, 2, 0, 0, 2, 0, 0, 3, 2, 2, -1, 3, 1, 1, 2, 0, 0, 2, 1, 1, 0, 1, 1, 3, 0, 2, 1, 1, 0, 3, 2, 1, 0, 2, 1, 2, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 3, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 4, 3, 0, 2, 1, 0, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 0, 2, 2, 1, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 4, 3, 0, 0, 1, 0, 1, 1, 3, 3, 1, 0, 1, 1, 0, 0, 3, 3, 0, 2, 3, 1, 2, 1, 3, 2, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 3, 1, 2, 0, -1, 0, 1, 0, 1, 4, 4, 0, 0, 0, 1, 0, 1, 0, 1, 3, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 0, 2, 0, 0, 2, 1, 1, 1, 0, 1, 3, 1, 0, 0, 0, 1, 1, 0, 1, 2, 0, 2, 2, 0, 1, 0, 2, 3, 1, 1, 1, 1, 0, 2, 2, 1, 2, 0, 0, 2, 0, 1, 3, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 2, 0, 2, 1, 1, 3, 1, 2, 2, 0, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 0, 0, 2, 1, 1, 3, 1, 2, 2, 2, 2, 0, 1, 1, 1, 2, 0, 1, 4, 0, 0, 0, 1, 4, 0, 1, 4, 1, 2, 1, 1, 3, 3, 3, 4, 1, 0, 1, 0, 0, 3, 1, 4, 1, 3, 1, 1, 1, 0, 2, 4, 1, 0, 3, 2, 1, 0, 0, 3, 1, 2, 0, 0, 0, 4, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 2, 3, 0, 1, 0, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 0, 1, 0, 0, 1, 2, 1, 3, 2, 0, 2, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 0, 1, 0, 2, 3, 3, 0, -1, 2, 0, 1, 1, 3, 0, 1, 0, 0, 3, 0, 2, 0, 0, 1, 0, 2, 2, -1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 3, 1, 0, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 2, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 3, 2, 3, 0, 3, 2, 3, 2]
  print("average score", np.mean(history), "standard_deviation", np.std(history))
