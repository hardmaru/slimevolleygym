#!/usr/bin/env python3

# Trains a convnet PPO agent to play SlimeVolley from pixels (SlimeVolleyNoFrameskip-v0)
# requires stable_baselines (I used 2.10)

# run with
# mpirun -np 96 python train_ppo_pixel.py (replace 96 with number of CPU cores you have.)

import os
import gym
import slimevolleygym

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv, WarpFrame
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import bench, logger, PPO1
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback

from slimevolleygym import FrameStack # doesn't use Lazy Frames, easier to debug


NUM_TIMESTEPS = 2000000000
SEED = 101
EVAL_FREQ = 100000 # note, will evaluate ever 96 * 100000 steps (where 96 is number of workers)
EVAL_EPISODES = 100 # make sure to rerun on 1000 episodes afterwards to measure that performance, don't use only 100.
LOGDIR = "ppo1_cnn" # moved to zoo afterwards.


def make_env(seed):
  # almost the same a typical Atari processing for CNN agent.
  # (I removed reward clipping, but used survival reward bonus)
  env = gym.make("SlimeVolleySurvivalNoFrameskip-v0")
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndSkipEnv(env, skip=4)
  env = WarpFrame(env)
  #env = ClipRewardEnv(env)
  env = FrameStack(env, 4)
  env.seed(seed)
  return env


def make_eval_env(seed):
  env = gym.make("SlimeVolleyNoFrameskip-v0")
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndSkipEnv(env, skip=4)
  env = WarpFrame(env)
  #env = ClipRewardEnv(env)
  env = FrameStack(env, 4)
  env.seed(seed)
  return env


def train():
  """
  Train PPO1 model for cartpole swingup, for testing purposes.
  """
  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
    logger.configure(folder=LOGDIR)

  else:
    logger.configure(format_strs=[])
  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  env = make_env(workerseed)
  eval_env = make_eval_env(workerseed)

  env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
  env.seed(workerseed)

  # hyperparameters from stable baseline's ppo1 atary example:
  model = PPO1(CnnPolicy, env, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01, optim_epochs=4,
               optim_stepsize=1e-3, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)

  eval_callback = EvalCallback(eval_env, best_model_save_path=LOGDIR,
                             log_path=LOGDIR, eval_freq=EVAL_FREQ,
                             deterministic=True, render=False,
                             n_eval_episodes=EVAL_EPISODES)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  env.close()
  del env
  if rank == 0:
    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.


if __name__ == '__main__':
  train()
