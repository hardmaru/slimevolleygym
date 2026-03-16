import numpy as np


class BaselinePolicy:
    """Tiny RNN policy with only 120 parameters of otoro.net/slimevolley agent"""

    def __init__(self):
        self.nGameInput = 8  # 8 states for agent
        self.nGameOutput = 3  # 3 buttons (forward, backward, jump)
        self.nRecurrentState = 4  # extra recurrent states for feedback.

        self.nOutput = self.nGameOutput + self.nRecurrentState
        self.nInput = self.nGameInput + self.nOutput

        # store current inputs and outputs
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

        """See training details: https://blog.otoro.net/2015/03/28/neural-slime-volleyball/ """
        self.weight = np.array(
            [
                7.5719,
                4.4285,
                2.2716,
                -0.3598,
                -7.8189,
                -2.5422,
                -3.2034,
                0.3935,
                1.2202,
                -0.49,
                -0.0316,
                0.5221,
                0.7026,
                0.4179,
                -2.1689,
                1.646,
                -13.3639,
                1.5151,
                1.1175,
                -5.3561,
                5.0442,
                0.8451,
                0.3987,
                -2.9501,
                -3.7811,
                -5.8994,
                6.4167,
                2.5014,
                7.338,
                -2.9887,
                2.4586,
                13.4191,
                2.7395,
                -3.9708,
                1.6548,
                -2.7554,
                -1.5345,
                -6.4708,
                9.2426,
                -0.7392,
                0.4452,
                1.8828,
                -2.6277,
                -10.851,
                -3.2353,
                -4.4653,
                -3.1153,
                -1.3707,
                7.318,
                16.0902,
                1.4686,
                7.0391,
                1.7765,
                -1.155,
                2.6697,
                -8.8877,
                1.1958,
                -3.2839,
                -5.4425,
                1.6809,
                7.6812,
                -2.4732,
                1.738,
                0.3781,
                0.8718,
                2.5886,
                1.6911,
                1.2953,
                -9.0052,
                -4.6038,
                -6.7447,
                -2.5528,
                0.4391,
                -4.9278,
                -3.6695,
                -4.8673,
                -1.6035,
                1.5011,
                -5.6124,
                4.9747,
                1.8998,
                3.0359,
                6.2983,
                -4.8568,
                -2.1888,
                -4.1143,
                -3.9874,
                -0.0459,
                4.7134,
                2.8952,
                -9.3627,
                -4.685,
                0.3601,
                -1.3699,
                9.7294,
                11.5596,
                0.1918,
                3.0783,
                0.0329,
                -0.1362,
                -0.1188,
                -0.7579,
                0.3278,
                -0.977,
                -0.9377,
            ]
        )

        self.bias = np.array([2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685])

        # unflatten weight, convert it into 7x15 matrix.
        self.weight = self.weight.reshape(
            self.nGameOutput + self.nRecurrentState, self.nGameInput + self.nGameOutput + self.nRecurrentState
        )

    def reset(self):
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

    def _forward(self):
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(np.dot(self.weight, self.inputState) + self.bias)

    def _setInputState(self, obs):
        # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
        [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
        self.inputState[0 : self.nGameInput] = np.array([x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy])
        self.inputState[self.nGameInput :] = self.outputState

    def _getAction(self):
        forward = 0
        backward = 0
        jump = 0
        if self.outputState[0] > 0.75:
            forward = 1
        if self.outputState[1] > 0.75:
            backward = 1
        if self.outputState[2] > 0.75:
            jump = 1
        return [forward, backward, jump]

    def predict(self, obs):
        """take obs, update rnn state, return action"""
        if type(obs) == dict:
            obs = list(obs.values())[0]
        self._setInputState(obs)
        self._forward()
        return self._getAction()
