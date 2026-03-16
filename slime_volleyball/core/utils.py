import cv2

from slime_volleyball.core import constants


def upsize_image(img):
    return cv2.resize(
        img,
        (
            constants.PIXEL_WIDTH * constants.PIXEL_SCALE,
            constants.PIXEL_HEIGHT * constants.PIXEL_SCALE,
        ),
        interpolation=cv2.INTER_NEAREST,
    )


def downsize_image(img):
    return cv2.resize(
        img,
        (constants.PIXEL_WIDTH, constants.PIXEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


# conversion from space to pixels (allows us to render to diff resolutions)
def toX(x):
    return (x + constants.REF_W / 2) * constants.FACTOR


def toP(x):
    return (x) * constants.FACTOR


def toY(y):
    return y * constants.FACTOR


class DelayScreen:
    """initially the ball is held still for INIT_DELAY_FRAMES(30) frames"""

    def __init__(self, life=constants.INIT_DELAY_FRAMES):
        self.life = 0
        self.reset(life)

    def reset(self, life=constants.INIT_DELAY_FRAMES):
        self.life = life

    def status(self):
        if self.life == 0:
            return True
        self.life -= 1
        return False
