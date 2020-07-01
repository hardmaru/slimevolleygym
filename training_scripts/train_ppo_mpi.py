#!/usr/bin/env python3
# trains slime agent from states with multiworker via MPI (fast wallclock time)
# run with
# mpirun -np 96 python train_ppo_mpi.py (replace 96 with number of CPU cores you have.)

import os
import gym
import slimevolleygym

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import bench, logger, PPO1
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e8)
SEED = 831
EVAL_FREQ = 200000
EVAL_EPISODES = 1000
LOGDIR = "ppo1_mpi"

def make_env(seed):
  env = gym.make("SlimeVolley-v0")
  env.seed(seed)
  return env

def train():
  """
  Train PPO1 model for slime volleyball, in MPI multiprocessing. Tested for 96 CPUs.
  """
  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
    logger.configure(folder=LOGDIR)

  else:
    logger.configure(format_strs=[])
  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  env = make_env(workerseed)

  env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
  env.seed(workerseed)

  model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
               optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear',
               verbose=1)

  eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  env.close()
  del env
  if rank == 0:
    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.


if __name__ == '__main__':
  train()
