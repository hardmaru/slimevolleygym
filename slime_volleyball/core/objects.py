import math

import cv2
import numpy as np

from slime_volleyball.core import constants
from slime_volleyball.core import utils

try:
    from slime_volleyball import rendering
except ImportError as e:
    print(
        f"Unable to import rendering. This means you won't be able to render the game: {e}"
    )


def make_half_circle(radius=10, res=20, filled=True):
    """helper function for pyglet renderer"""
    points = []
    for i in range(res + 1):
        ang = math.pi - math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
    if filled:
        return rendering.FilledPolygon(points)
    else:
        return rendering.PolyLine(points, True)


def _add_attrs(geom, color):
    """help scale the colors from 0-255 to 0.0-1.0 (pyglet renderer)"""
    r = color[0]
    g = color[1]
    b = color[2]
    geom.set_color(r / 255.0, g / 255.0, b / 255.0)


def create_canvas(canvas, c):
    if constants.PIXEL_MODE:
        result = np.ones(
            (constants.WINDOW_HEIGHT, constants.WINDOW_WIDTH, 3), dtype=np.uint8
        )
        for channel in range(3):
            result[:, :, channel] *= c[channel]
        return result
    else:
        rect(
            canvas,
            0,
            0,
            constants.WINDOW_WIDTH,
            -constants.WINDOW_HEIGHT,
            color=constants.BACKGROUND_COLOR,
        )
        return canvas


def rect(canvas, x, y, width, height, color):
    """Processing style function to make it easy to port p5.js program to python"""
    if constants.PIXEL_MODE:
        canvas = cv2.rectangle(
            canvas,
            (round(x), round(constants.WINDOW_HEIGHT - y)),
            (round(x + width), round(constants.WINDOW_HEIGHT - y + height)),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        return canvas
    else:
        box = rendering.make_polygon(
            [(0, 0), (0, -height), (width, -height), (width, 0)]
        )
        trans = rendering.Transform()
        trans.set_translation(x, y)
        _add_attrs(box, color)
        box.add_attr(trans)
        canvas.add_onetime(box)
        return canvas


def half_circle(canvas, x, y, r, color):
    """Processing style function to make it easy to port p5.js program to python"""
    if constants.PIXEL_MODE:
        return cv2.ellipse(
            canvas,
            (round(x), constants.WINDOW_HEIGHT - round(y)),
            (round(r), round(r)),
            0,
            0,
            -180,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    else:
        geom = make_half_circle(r)
        trans = rendering.Transform()
        trans.set_translation(x, y)
        _add_attrs(geom, color)
        geom.add_attr(trans)
        canvas.add_onetime(geom)
        return canvas


def circle(canvas, x, y, r, color):
    """Processing style function to make it easy to port p5.js program to python"""
    if constants.PIXEL_MODE:
        return cv2.circle(
            canvas,
            (round(x), round(constants.WINDOW_HEIGHT - y)),
            round(r),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    else:
        geom = rendering.make_circle(r, res=40)
        trans = rendering.Transform()
        trans.set_translation(x, y)
        _add_attrs(geom, color)
        geom.add_attr(trans)
        canvas.add_onetime(geom)
        return canvas


class Particle:
    """used for the ball, and also for the round stub above the fence"""

    def __init__(self, x, y, vx, vy, r, c):
        self.x = x
        self.y = y
        self.prev_x = self.x
        self.prev_y = self.y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.c = c
        self.speed_factor = 1

    def display(self, canvas):
        return circle(
            canvas,
            utils.toX(self.x),
            utils.toY(self.y),
            utils.toP(self.r),
            color=self.c,
        )

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.vx * constants.TIMESTEP
        self.y += self.vy * constants.TIMESTEP

    def apply_acceleration(self, ax, ay):
        self.vx += ax * constants.TIMESTEP
        self.vy += ay * constants.TIMESTEP

    def check_edges(self):
        if self.x <= (self.r - constants.REF_W / 2):
            self.vx *= -constants.FRICTION
            self.x = (
                self.r
                - constants.REF_W / 2
                + constants.NUDGE * constants.TIMESTEP
            )

        if self.x >= (constants.REF_W / 2 - self.r):
            self.vx *= -constants.FRICTION
            self.x = (
                constants.REF_W / 2
                - self.r
                - constants.NUDGE * constants.TIMESTEP
            )

        if self.y <= (self.r + constants.REF_U):
            self.vy *= -constants.FRICTION
            self.y = (
                self.r + constants.REF_U + constants.NUDGE * constants.TIMESTEP
            )
            if self.x <= 0:
                return -1
            else:
                return 1
        if self.y >= (constants.REF_H - self.r):
            self.vy *= -constants.FRICTION
            self.y = (
                constants.REF_H - self.r - constants.NUDGE * constants.TIMESTEP
            )
        # fence:
        if (
            (self.x <= (constants.REF_WALL_WIDTH / 2 + self.r))
            and (self.prev_x > (constants.REF_WALL_WIDTH / 2 + self.r))
            and (self.y <= constants.REF_WALL_HEIGHT)
        ):
            self.vx *= -constants.FRICTION
            self.x = (
                constants.REF_WALL_WIDTH / 2
                + self.r
                + constants.NUDGE * constants.TIMESTEP
            )

        if (
            (self.x >= (-constants.REF_WALL_WIDTH / 2 - self.r))
            and (self.prev_x < (-constants.REF_WALL_WIDTH / 2 - self.r))
            and (self.y <= constants.REF_WALL_HEIGHT)
        ):
            self.vx *= -constants.FRICTION
            self.x = (
                -constants.REF_WALL_WIDTH / 2
                - self.r
                - constants.NUDGE * constants.TIMESTEP
            )
        return 0

    def get_dist2(self, p):  # returns distance squared from p
        dy = p.y - self.y
        dx = p.x - self.x
        return dx * dx + dy * dy

    def is_colliding(self, p):  # returns true if it is colliding w/ p
        r = self.r + p.r
        return r * r > self.get_dist2(
            p
        )  # if distance is less than total radius, then colliding.

    def bounce(
        self, p, factor=1
    ):  # bounce two balls that have collided (this and that)
        abx = self.x - p.x
        aby = self.y - p.y
        abd = math.sqrt(abx * abx + aby * aby)
        abx /= abd  # normalize
        aby /= abd
        nx = abx  # reuse calculation
        ny = aby
        abx *= constants.NUDGE
        aby *= constants.NUDGE
        while self.is_colliding(p):
            self.x += abx
            self.y += aby
        ux = self.vx - p.vx
        uy = self.vy - p.vy
        un = ux * nx + uy * ny
        unx = nx * (un * 2.0)  # added factor of 2
        uny = ny * (un * 2.0)  # added factor of 2
        ux -= unx
        uy -= uny
        self.vx = (ux + p.vx) * factor
        self.vy = (uy + p.vy) * factor
        self.speed_factor = factor

    def limit_speed(self, minSpeed, maxSpeed):
        maxSpeed *= self.speed_factor
        mag2 = self.vx * self.vx + self.vy * self.vy
        if mag2 > (maxSpeed * maxSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed

        if mag2 < (minSpeed * minSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= minSpeed
            self.vy *= minSpeed


class Wall:
    """used for the fence, and also the ground"""

    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def display(self, canvas):
        return rect(
            canvas,
            utils.toX(self.x - self.w / 2),
            utils.toY(self.y + self.h / 2),
            utils.toP(self.w),
            utils.toP(self.h),
            color=self.c,
        )


class RelativeState:
    """
    keeps track of the obs.
    Note: the observation is from the perspective of the agent.
    an agent playing either side of the fence must see obs the same way
    """

    scale_factor = 10.0

    def __init__(self):
        # agent
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.powerups_available = 0
        self.powered_up_timer = 0

        # ball
        self.bx = 0
        self.by = 0
        self.bvx = 0
        self.bvy = 0

        # opponent
        self.ox = 0
        self.oy = 0
        self.ovx = 0
        self.ovy = 0
        self.opponent_powerups_available = 0
        self.opponent_powered_up_timer = 0

    def get_observation(self, powerups=False):
        result = [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.bx,
            self.by,
            self.bvx,
            self.bvy,
            self.ox,
            self.oy,
            self.ovx,
            self.ovy,
        ]
        if powerups:
            result.append(self.powerups_available)
            result.append(self.powered_up_timer)
            result.append(self.opponent_powerups_available)
            result.append(self.opponent_powered_up_timer)
        result = np.array(result) / self.scale_factor
        return result.astype(np.float32)
