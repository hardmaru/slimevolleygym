RENDER_MODE = True

REF_W = 24 * 2
REF_H = REF_W
REF_U = 1.5  # ground height
REF_WALL_WIDTH = 1.0  # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 20  # 10 * 1.75
PLAYER_SPEED_Y = 10 * 1.75  # 10 * 1.35
MAX_BALL_SPEED = 15 * 1.5
TIMESTEP = 1 / 30.0
NUDGE = 0.1
FRICTION = 1.0  # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8 * 1.5 * 1.5  # -9.8 * 2 * 1.5

MAXLIVES = 1  # game ends when one agent loses this many games

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = True
PIXEL_MODE = True
PIXEL_SCALE = 4  # first render at multiple of Pixel Obs resolution, then downscale. Looks better.

PIXEL_WIDTH = 84 * 2 * 1
PIXEL_HEIGHT = 84 * 1

BALL_COLOR = (217, 79, 0)
AGENT_LEFT_COLOR = (35, 93, 188)
AGENT_RIGHT_COLOR = (255, 236, 0)
PIXEL_AGENT_LEFT_COLOR = (255, 191, 0)  # AMBER
PIXEL_AGENT_RIGHT_COLOR = (255, 191, 0)  # AMBER

BACKGROUND_COLOR = (11, 16, 19)
FENCE_COLOR = (102, 56, 35)
COIN_COLOR = FENCE_COLOR
POWERUP_COLOR = (0, 255, 255)
GROUND_COLOR = (116, 114, 117)


def setPixelObsMode():
    """
    used for experimental pixel-observation mode
    note: new dim's chosen to be PIXEL_SCALE (2x) as Pixel Obs dims (will be downsampled)

    also, both agent colors are identical, to potentially facilitate multiagent
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR, PIXEL_MODE
    PIXEL_MODE = True
    WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
    WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
    FACTOR = WINDOW_WIDTH / REF_W
    AGENT_LEFT_COLOR = PIXEL_AGENT_LEFT_COLOR
    AGENT_RIGHT_COLOR = PIXEL_AGENT_RIGHT_COLOR
