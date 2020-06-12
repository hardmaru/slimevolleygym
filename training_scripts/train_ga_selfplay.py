# Trains an agent from scratch (no existing AI) using evolution
# GA with no cross-over, just mutation, and random tournament selection
# Not optimized for speed, and just uses a single CPU (mainly for simplicity)

import os
import json
import numpy as np
import gym
import slimevolleygym
import slimevolleygym.mlp as mlp
from slimevolleygym.mlp import Model
from slimevolleygym import multiagent_rollout as rollout

# Settings
random_seed = 612
population_size = 128
total_tournaments = 500000
save_freq = 1000

def mutate(length, mutation_rate, mutation_sigma):
  # (not used, in case I wanted to do partial mutations)
  # create an additive mutation vector of some size
  mask = np.random.randint(int(1/mutation_rate), size=length)
  mask = 1-np.minimum(mask, 1)
  noise = np.random.normal(size=length) * mutation_sigma
  return mask * noise

# Log results
logdir = "ga_selfplay"
if not os.path.exists(logdir):
  os.makedirs(logdir)


# Create two instances of a feed forward policy we may need later.
policy_left = Model(mlp.games['slimevolleylite'])
policy_right = Model(mlp.games['slimevolleylite'])
param_count = policy_left.param_count
print("Number of parameters of the neural net policy:", param_count) # 273 for slimevolleylite

# store our population here
population = np.random.normal(size=(population_size, param_count)) * 0.5 # each row is an agent.
winning_streak = [0] * population_size # store the number of wins for this agent (including mutated ones)

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

history = []
for tournament in range(1, total_tournaments+1):

  m, n = np.random.choice(population_size, 2, replace=False)

  policy_left.set_model_params(population[m])
  policy_right.set_model_params(population[n])

  # the match between the mth and nth member of the population
  score, length = rollout(env, policy_right, policy_left)

  history.append(length)
  # if score is positive, it means policy_right won.
  if score == 0: # if the game is tied, add noise to the left agent.
    population[m] += np.random.normal(size=param_count) * 0.1
  if score > 0:
    population[m] = population[n] + np.random.normal(size=param_count) * 0.1
    winning_streak[m] = winning_streak[n]
    winning_streak[n] += 1
  if score < 0:
    population[n] = population[m] + np.random.normal(size=param_count) * 0.1
    winning_streak[n] = winning_streak[m]
    winning_streak[m] += 1

  if tournament % save_freq == 0:
    model_filename = os.path.join(logdir, "ga_"+str(tournament).zfill(8)+".json")
    with open(model_filename, 'wt') as out:
      record_holder = np.argmax(winning_streak)
      record = winning_streak[record_holder]
      json.dump([population[record_holder].tolist(), record], out, sort_keys=True, indent=0, separators=(',', ': '))

  if (tournament ) % 100 == 0:
    record_holder = np.argmax(winning_streak)
    record = winning_streak[record_holder]
    print("tournament:", tournament,
          "best_winning_streak:", record,
          "mean_duration", np.mean(history),
          "stdev:", np.std(history),
         )
    history = []