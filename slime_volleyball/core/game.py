import numpy as np

from slime_volleyball.core import constants
from slime_volleyball.core.objects import (
    Wall,
    Particle,
    create_canvas,
)
from slime_volleyball.core.agent import Agent
from slime_volleyball.core import utils


class SlimeVolleyGame:
    """
    the main slime volley game.
    can be used in various settings, such as ai vs ai, ai vs human, human vs human
    """

    def __init__(self, np_random: np.random.Generator):
        self.ball = None
        self.ground = None
        self.fence = None
        self.fence_stub = None
        self.agent_left = None
        self.agent_right = None
        self.delay_screen = None
        self.np_random = np_random
        self.reset()

    def reset(self):
        self.ground = Wall(
            0, 0.75, constants.REF_W, constants.REF_U, c=constants.GROUND_COLOR
        )
        self.fence = Wall(
            0,
            0.75 + constants.REF_WALL_HEIGHT / 2,
            constants.REF_WALL_WIDTH,
            (constants.REF_WALL_HEIGHT - 1.5),
            c=constants.FENCE_COLOR,
        )
        self.fence_stub = Particle(
            0,
            constants.REF_WALL_HEIGHT,
            0,
            0,
            constants.REF_WALL_WIDTH / 2,
            c=constants.FENCE_COLOR,
        )
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(
            0,
            constants.REF_W / 4,
            ball_vx,
            ball_vy,
            0.5,
            c=constants.BALL_COLOR,
        )
        self.agent_left = Agent(
            -1,
            -constants.REF_W / 4,
            1.5,
            c=constants.AGENT_LEFT_COLOR,
        )
        self.agent_right = Agent(
            1,
            constants.REF_W / 4,
            1.5,
            c=constants.AGENT_RIGHT_COLOR,
        )
        self.agent_left.update_state(self.ball, self.agent_right)
        self.agent_right.update_state(self.ball, self.agent_left)
        self.delay_screen = utils.DelayScreen()

    def new_match(self):
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(
            0,
            constants.REF_W / 4,
            ball_vx,
            ball_vy,
            0.5,
            c=constants.BALL_COLOR,
        )
        self.delay_screen.reset()

    def step(self):
        """main game loop"""

        self.between_game_control()
        self.agent_left.update()
        self.agent_right.update()
        if self.delay_screen.status():
            self.ball.apply_acceleration(0, constants.GRAVITY)
            self.ball.limit_speed(0, constants.MAX_BALL_SPEED)
            self.ball.move()

        if self.ball.is_colliding(self.agent_left):
            self.ball.bounce(
                self.agent_left,
                factor=(1 + 0.5 * int(self.agent_left.powered_up_timer > 0)),
            )
        if self.ball.is_colliding(self.agent_right):
            self.ball.bounce(
                self.agent_right,
                factor=(1 + 0.5 * int(self.agent_right.powered_up_timer > 0)),
            )
        if self.ball.is_colliding(self.fence_stub):
            self.ball.bounce(self.fence_stub)

        # negated, since we want reward to be from the persepctive of right agent being trained.
        result = -self.ball.check_edges()

        if result != 0:
            if constants.MAXLIVES > 1:
                self.new_match()  # not reset, but after a point is scored

            if result < 0:  # baseline agent won
                self.agent_left.emotion = "happy"
                self.agent_right.emotion = "sad"
                self.agent_right.life -= 1
            else:
                self.agent_left.emotion = "sad"
                self.agent_right.emotion = "happy"
                self.agent_left.life -= 1
            return result

        # update internal states (the last thing to do)
        self.agent_left.update_state(self.ball, self.agent_right)
        self.agent_right.update_state(self.ball, self.agent_left)

        return result

    def display(self, canvas):
        # background color
        # if PIXEL_MODE is True, canvas is an RGB array.
        # if PIXEL_MODE is False, canvas is viewer object
        canvas = create_canvas(canvas, c=constants.BACKGROUND_COLOR)
        canvas = self.fence.display(canvas)
        canvas = self.fence_stub.display(canvas)
        canvas = self.agent_left.display(canvas, self.ball.x, self.ball.y)
        canvas = self.agent_right.display(canvas, self.ball.x, self.ball.y)
        canvas = self.ball.display(canvas)
        canvas = self.ground.display(canvas)
        return canvas

    def between_game_control(self):
        agent = [self.agent_left, self.agent_right]
        if self.delay_screen.life > 0:
            pass
            """
            for i in range(2):
              if (agent[i].emotion == "sad"):
                agent[i].setAction([0, 0, 0]) # nothing
            """
        else:
            agent[0].emotion = "happy"
            agent[1].emotion = "happy"
