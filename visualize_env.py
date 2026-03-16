import numpy as np
from time import sleep

from slime_volleyball.core import constants
from slime_volleyball.core.game import SlimeVolleyGame
from slime_volleyball.core import utils
from slime_volleyball.core.observations import compute_observations


BOOST_MODE = True


if __name__ == "__main__":
    """
    Interactive visualization using the legacy game engine (numpy, pyglet).

    Humans can override controls:

    Right Agent:
    Up Arrow - Jump
    Left Arrow - Left
    Right Arrow - Right
    Space - Boost (if BOOST_MODE)

    Left Agent:
    W - Jump
    A - Left
    D - Right
    """

    from pyglet.window import key

    manualAction = [0, 0, 0, 0] if BOOST_MODE else [0, 0, 0]
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False

    def key_press(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:
            manualAction[0] = 1
        if k == key.RIGHT:
            manualAction[1] = 1
        if k == key.UP:
            manualAction[2] = 1
        if k == key.SPACE and len(manualAction) > 3:
            manualAction[3] = 1
        if k == key.LEFT or k == key.RIGHT or k == key.UP:
            manualMode = True

        if k == key.D:
            otherManualAction[0] = 1
        if k == key.A:
            otherManualAction[1] = 1
        if k == key.W:
            otherManualAction[2] = 1
        if k == key.D or k == key.A or k == key.W:
            otherManualMode = True

    def key_release(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:
            manualAction[0] = 0
        if k == key.RIGHT:
            manualAction[1] = 0
        if k == key.UP:
            manualAction[2] = 0
        if k == key.SPACE and len(manualAction) > 3:
            manualAction[3] = 0
        if k == key.D:
            otherManualAction[0] = 0
        if k == key.A:
            otherManualAction[1] = 0
        if k == key.W:
            otherManualAction[2] = 0

    # Use the legacy OOP game engine directly for interactive visualization.
    # This avoids the dual-backend functional pipeline and uses pyglet rendering.
    constants.PIXEL_MODE = False
    constants.RENDER_MODE = True

    from slime_volleyball import rendering

    viewer = rendering.Viewer(constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT)
    viewer.window.on_key_press = key_press
    viewer.window.on_key_release = key_release

    for episode in range(10):
        rng = np.random.default_rng(np.random.randint(0, 10000))
        game = SlimeVolleyGame(rng)

        steps = 0
        total_reward = 0

        done = False
        while not done and steps < 3000:
            # Render
            game.display(viewer)
            viewer.render()

            # Get actions
            if manualMode:
                right_action = manualAction
            else:
                right_action = [
                    np.random.randint(0, 2),
                    np.random.randint(0, 2),
                    np.random.randint(0, 2),
                ]

            if otherManualMode:
                left_action = otherManualAction
            else:
                left_action = [
                    np.random.randint(0, 2),
                    np.random.randint(0, 2),
                    np.random.randint(0, 2),
                ]

            game.agent_left.set_action(left_action)
            game.agent_right.set_action(right_action)

            result = game.step()
            total_reward += result
            steps += 1

            if game.agent_left.life <= 0 or game.agent_right.life <= 0:
                done = True

            sleep(0.01)
            if manualMode or otherManualMode:
                sleep(0.02)

        print(f"Episode {episode+1}: score={total_reward}, steps={steps}")

    viewer.close()
