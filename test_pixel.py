"""
Human vs AI in pixel observation environment

Note that for multiagent mode, otherObs's image is horizontally flipped

Performance, 100,000 frames in 144.839 seconds, or 690 fps.
"""

import gym
import slimevolleygym
from time import sleep
from pyglet.window import key

from gym.envs.classic_control import rendering as rendering # to show actual obs2

if __name__=="__main__":

  manualAction = [0, 0, 0] # forward, backward, jump
  manualMode = False

  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode, manualAction
    if k == key.LEFT:  manualAction[0] = 1
    if k == key.RIGHT: manualAction[1] = 1
    if k == key.UP:    manualAction[2] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

  def key_release(k, mod):
    global manualMode, manualAction
    if k == key.LEFT:  manualAction[0] = 0
    if k == key.RIGHT: manualAction[1] = 0
    if k == key.UP:    manualAction[2] = 0

  viewer = rendering.SimpleImageViewer(maxwidth=2160)

  env = gym.make("SlimeVolleySurvivalNoFrameskip-v0")

  policy = slimevolleygym.BaselinePolicy() # throw in a default policy (based on state, not pixels)

  obs = env.reset()
  env.render()

  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release

  defaultAction = [0, 0, 0]

  for t in range(10000):
    if manualMode: # override with keyboard
      action = manualAction # now just work w/ multibinary if it is not scalar
    else:
      action = defaultAction
    obs, reward, done, info = env.step(action)
    otherObs = info['otherObs']

    state = info['state'] # cheat and look at the actual state (to find default actions quickly)
    defaultAction = policy.predict(state)
    sleep(0.02)
    #viewer.imshow(otherObs) # show the opponent's observtion (horizontally flipped)
    env.render()
    if done:
      obs = env.reset()
    if (t+1) % 5000 == 0:
      print(t+1)

  viewer.close()
  env.close()
