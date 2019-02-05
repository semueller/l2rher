import numpy as np
# from osim.env import RunEnv
from osim.env import ProstheticsEnv
from gym.spaces import Box, MultiBinary

'''
 file from github.com/0123Andrew/L2R.git
'''

class RunEnv2(ProstheticsEnv):
    def __init__(self, visualize=True, integrator_accuracy=5e-5, model='2D', prosthetic=False, difficulty=0, skip_frame=3, reward_mult=1., args=None):
        # print('#############################################')
        # print('RunEnv2(ProstheticsEnv)  »  __init__  »  args = ' + str(args))
        # print('#############################################')

        super(RunEnv2, self).__init__(visualize, integrator_accuracy)
        self.args = args
        self.param = (model, prosthetic, difficulty)
        self.change_model(*self.param)
        # self.state_transform = state_transform
        # self.observation_space = Box(-1000, 1000, [state_size], dtype=np.float32)
        # self.observation_space = Box(-1000, 1000, [state_transform.state_size], dtype=np.float32)
        self.noutput = self.get_action_space_size()
        self.action_space = MultiBinary(self.noutput)
        self.skip_frame = skip_frame
        self.reward_mult = reward_mult

    def reset(self, difficulty=0, seed=None):
        self.change_model(self.param[0], self.param[1], difficulty, seed)
        d = super(RunEnv2, self).reset(False)
        s = self.dict_to_vec(d)
        # self.state_transform.reset()
        # s = self.state_transform.process(s)
        return s

    def is_done(self):  # ndrw
        # state_desc = self.get_state_desc()
        return self.osim_model.state_desc["body_pos"]["pelvis"][1] < 0.3

    def _step(self, action):
        action = np.clip(action, 0, 1)
        info = {'original_reward':0}
        reward = 0.
        for _ in range(self.skip_frame):
            s, r, t, _ = super(RunEnv2, self).step(action, False)
            pelvis = s['body_pos']['pelvis']
            r = self.x_velocity_reward(s)
            s = self.dict_to_vec(s)  # ndrw subtract pelvis_X
            # s = self.state_transform.process(s)
            info['original_reward'] += r
            reward += r
            if t:
                break
        info['pelvis'] = pelvis
        return s, reward*self.reward_mult, t, info

    def x_velocity_reward(self, state):
        lim_foot_rell_pelv = 0.35
        if self.args.reward_func == '2D':
            penalty = -1.0
            reward = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward
            # if agent is falling, return negative reward
            if state['body_pos']['pelvis'][1] < 0.75:
                return penalty + reward #  -10
            if state['body_pos']['head'][0] - state['body_pos']['pelvis'][0] < -0.35:
                return penalty + reward #  -10
            # x velocity of pelvis
            # return state['body_vel']['pelvis'][0]
            return state['misc']['mass_center_vel'][0]  #  X velocity - forward/backward

        elif self.args.reward_func == '3D_pen01_VelMass':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty = -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty = -1.0
                penalty_dict['pelvis_low'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty = -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty = -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty = -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty = -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty = -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty = -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            # return penalty + reward_mass_vel + reward_foot_l_vel_0 + reward_foot_r_vel_0

        elif self.args.reward_func == '3D_penAdd_VelMass':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty += -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty += -1.0
                penalty_dict['pelvis_low'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty += -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty += -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty += -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty += -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            # return penalty + reward_mass_vel + reward_foot_l_vel_0 + reward_foot_r_vel_0

        elif self.args.reward_func == '3D_penAdd2_VelMass':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty += -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty += -5.0
                penalty_dict['pelvis_low'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty += -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty += -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty += -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty += -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            # return penalty + reward_mass_vel + reward_foot_l_vel_0 + reward_foot_r_vel_0

        elif self.args.reward_func == '3D_penAdd2f_VelMass':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty += -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty += -5.0
                penalty_dict['pelvis_low'] = 1

            # penalty for moving backwards
            penalty_dict['pelvis_vel_0'] = 0
            if state['body_vel']['pelvis'][0] < 0.:  # forward/backward
                penalty += -1.0
                penalty_dict['pelvis_vel_0'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty += -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty += -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty += -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty += -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            # return penalty + reward_mass_vel + reward_foot_l_vel_0 + reward_foot_r_vel_0

        elif self.args.reward_func == '3D_penAdd2f_075_VelMass':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty += -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.75:  # up/down - absolute values
                penalty += -5.0
                penalty_dict['pelvis_low'] = 1

            # penalty for moving backwards
            penalty_dict['pelvis_vel_0'] = 0
            if state['body_vel']['pelvis'][0] < 0.:  # forward/backward
                penalty += -1.0
                penalty_dict['pelvis_vel_0'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty += -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty += -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty += -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty += -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            # return penalty + reward_mass_vel + reward_foot_l_vel_0 + reward_foot_r_vel_0

        elif self.args.reward_func == '3D_penAdd3_VelMass':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty += -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty += -10.0
                penalty_dict['pelvis_low'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty += -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty += -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty += -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty += -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            # return penalty + reward_mass_vel + reward_foot_l_vel_0 + reward_foot_r_vel_0

        elif self.args.reward_func == '3D_pen01_VelMassFoot':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty = -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty = -1.0
                penalty_dict['pelvis_low'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.5 or tmp > 0.25:  # forward/backward
                penalty = -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty = -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty = -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty = -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty = -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            feet_diff_2 = foot_r - foot_l
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty = -1.0
                penalty_dict['feet_diff_2'] = 1

            self.penalty_dict = penalty_dict
            # return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            return penalty + reward_mass_vel + reward_foot_l_vel_0/2. + reward_foot_r_vel_0/2.

        elif self.args.reward_func == '3D_penAdd_VelMassFoot2':
            penalty_dict = {}
            penalty = 0.0
            reward_mass_vel = state['misc']['mass_center_vel'][0]  # X velocity - forward/backward

            # turn left/right penalty (rotation around Y) - 0 = strait forward
            penalty_dict['turn'] = 0
            turn = abs(state['body_pos']['femur_l'][0] - state['body_pos']['femur_r'][0]) / 0.167  # 0-1
            if turn > 0.5:
                penalty += -1.0
                penalty_dict['turn'] = 1
            # print('turn = ' + format(turn, '.3f') + '    penalty_turn = ' + format(penalty_turn, '.3f'))

            # if the agent is falling, return negative reward
            penalty_dict['pelvis_low'] = 0
            if state['body_pos']['pelvis'][1] < 0.66:  # up/down - absolute values
                penalty += -1.0
                penalty_dict['pelvis_low'] = 1

            # it the head is far ahead or behind
            penalty_dict['head_rel_pelv_0'] = 0
            tmp = state['body_pos']['head'][0] - state['body_pos']['pelvis'][0]
            if tmp < -0.45 or tmp > 0.25:  # forward/backward
                penalty += -1.0
                penalty_dict['head_rel_pelv_0'] = 1

            # it the head is far left or right
            penalty_dict['head_rel_pelv_2'] = 0
            if abs(state['body_pos']['head'][2] - state['body_pos']['pelvis'][2]) > 0.25:  # left/right
                penalty += -1.0
                penalty_dict['head_rel_pelv_2'] = 1

            # if the left foot is far left or right
            penalty_dict['foot_l_rel_pelv_2'] = 0
            foot_l0 = state['body_pos']['talus_l'][0]
            foot_l2 = state['body_pos']['talus_l'][2]
            reward_foot_l_vel_0 = state['body_vel']['talus_l'][0]
            if abs(state['body_pos']['talus_l'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                penalty += -1.0
                penalty_dict['foot_l_rel_pelv_2'] = 1

            # right foot is far left or right
            penalty_dict['foot_r_rel_pelv_2'] = 0
            if self.args.prosthetic:  # True = prosthetic
                foot_r0 = state['body_pos']['pros_foot_r'][0]
                foot_r2 = state['body_pos']['pros_foot_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['pros_foot_r'][0]
                if abs(state['body_pos']['pros_foot_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1
            else:
                foot_r0 = state['body_pos']['talus_r'][0]
                foot_r2 = state['body_pos']['talus_r'][2]
                reward_foot_r_vel_0 = state['body_vel']['talus_r'][0]
                if abs(state['body_pos']['talus_r'][2] - state['body_pos']['pelvis'][2]) > lim_foot_rell_pelv:  # left/right - absolute values
                    penalty += -1.0
                    penalty_dict['foot_r_rel_pelv_2'] = 1

            feet_diff_2 = foot_r2 - foot_l2
            penalty_dict['feet_diff_2'] = 0
            if feet_diff_2 > 0.6 or feet_diff_2 < 0.0:
                penalty += -1.0
                penalty_dict['feet_diff_2'] = 1

            # no reward_foot if both feet are ahead
            if foot_l0 > state['misc']['mass_center_pos'][0] and foot_r0 > state['misc']['mass_center_pos'][0]:
                reward_foot = 0.0
            else:
                reward_foot = reward_foot_l_vel_0/4. + reward_foot_r_vel_0/4.

            self.penalty_dict = penalty_dict
            # return penalty + reward_mass_vel  # reward = X velocity - forward/backward
            r = penalty + reward_mass_vel + reward_foot
            return r


    # @staticmethod
    def dict_to_vec(self, dict_):
        """Project a dictionary to a vector.
        Filters fiber forces in muscle dictionary as they are already
        contained in the forces dictionary.
        """
        # length without prosthesis: 443 (+ 22 redundant values)
        # length with prosthesis: 390 (+ 19 redundant values)

        if self.args.proj == '2D35':
            pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
            projection = [dict_['body_pos']['pelvis'][1]]  # pelvis up
            for dict_name in ['body_pos', 'body_vel']:
                for dict_name_2 in ['head', 'pelvis', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'toes_r',
                                    'toes_l']:  # dict_[dict_name]
                    if dict_name_2 == 'pelvis' and dict_name == 'body_pos':
                        continue
                    lll = dict_[dict_name][dict_name_2]
                    if len(lll) > 0:
                        for i in [0, 1]:
                            l = lll[i]
                            if dict_name == 'body_pos' and i == 0:
                                projection += [l - pelvis_X]
                            else:
                                projection += [l]
            projection += [dict_['misc']['mass_center_pos'][
                               0] - pelvis_X]  # [0] - X forward/backward, [1] - Y up/down, [2] - Z left/right
            projection += [dict_['misc']['mass_center_pos'][1]]
            projection += [dict_['misc']['mass_center_vel'][0]]
            projection += [dict_['misc']['mass_center_vel'][1]]

            assert len(projection) == 36
            projection = np.array(projection)
            return projection


        elif self.args.proj == '2Dpos':
            pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
            projection = dict_['body_pos']['pelvis'][:2]  # pelvis X+Y
            for dict_name in ['body_pos']:
                for dict_name_2 in ['head', 'pelvis', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'toes_r', 'toes_l']:  # dict_[dict_name]
                    if dict_name_2 == 'pelvis' and dict_name == 'body_pos':
                        continue
                    lll = dict_[dict_name][dict_name_2]
                    if len(lll) > 0:
                        for i in [0, 1]:
                            l = lll[i]
                            if dict_name == 'body_pos' and i == 0:
                                projection += [l-pelvis_X]
                            else:
                                projection += [l]
            projection += [dict_['misc']['mass_center_pos'][0] - pelvis_X]
            projection += [dict_['misc']['mass_center_pos'][1]]
            # projection += dict_['misc']['mass_center_vel']
            # print(len(projection))
            assert len(projection) == 18
            projection = np.array(projection)
            return projection


        elif self.args.proj == '2DPro31':
            pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
            projection = [dict_['body_pos']['pelvis'][1]]  # pelvis up
            for dict_name in ['body_pos', 'body_vel']:
                for dict_name_2 in ['head', 'pelvis', 'pros_tibia_r', 'tibia_l', 'pros_foot_r', 'talus_l', 'toes_l']:  # dict_[dict_name]
                    if dict_name_2 == 'pelvis' and dict_name == 'body_pos':
                        continue
                    lll = dict_[dict_name][dict_name_2]
                    if len(lll) > 0:
                        for i in [0, 1]:
                            l = lll[i]
                            if dict_name == 'body_pos' and i == 0:
                                projection += [l-pelvis_X]
                            else:
                                projection += [l]
            projection += [dict_['misc']['mass_center_pos'][0] - pelvis_X]
            projection += [dict_['misc']['mass_center_pos'][1]]
            projection += dict_['misc']['mass_center_vel']

            assert len(projection) == 31
            projection = np.array(projection)
            return projection

        elif self.args.proj in ['3DPro35', '3D38']:
            # [0] = X - backward/forward
            # [1] = Y - down/up
            # [2] = Z - left/right
            projection_dict = {}
            pelvis_X = dict_['body_pos']['pelvis'][0]  # X - forward, Y - up, Z - left/right
            pelvis_Y = dict_['body_pos']['pelvis'][1]  # X - forward, Y - up, Z - left/right
            pelvis_Z = dict_['body_pos']['pelvis'][2]  # X - forward, Y - up, Z - left/right
            for dict_name in ['body_pos', 'body_vel']:
                for dict_name_2 in ['head', 'pelvis', 'pros_tibia_r', 'tibia_r', 'tibia_l', 'talus_r', 'talus_l', 'pros_foot_r', 'toes_r', 'toes_l']:  # dict_[dict_name]
                    if dict_name_2 in dict_[dict_name]:  # e.g. prosthetic exceptions: ['pros_tibia_r', 'pros_foot_r']
                        lll = dict_[dict_name][dict_name_2]
                        if len(lll) > 0:
                            for i in [0, 1, 2]:
                                l = lll[i]
                                if i == 0:  # forward/backward
                                    if dict_name == 'body_pos' and dict_name_2 != 'pelvis':
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i) + '_relX'] = l-pelvis_X
                                    elif dict_name == 'body_vel' and (dict_name_2 not in ['pros_tibia_r', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']):
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
                                elif i == 1:  # up/down
                                    if dict_name == 'body_pos':
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
                                    elif dict_name == 'body_vel' and (dict_name_2 not in ['pros_tibia_r', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']):
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
                                elif i == 2:  # left/right
                                    # if dict_name_2 in ['head', 'pelvis', 'talus_r', 'talus_l', 'pros_tibia_r', 'pros_foot_r']:
                                    if dict_name == 'body_pos' and dict_name_2 != 'pelvis':
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i) + '_relZ'] = l - pelvis_Z
                                    elif dict_name == 'body_vel' and (dict_name_2 not in ['pros_tibia_r', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']):
                                        projection_dict[dict_name + '-' + dict_name_2 + '-' + str(i)] = l
            projection_dict['misc' + '-' + 'mass_center_pos' + '-' + str(0) + '_relX'] = dict_['misc']['mass_center_pos'][0] - pelvis_X
            projection_dict['misc' + '-' + 'mass_center_pos' + '-' + str(1) + '_relY'] = dict_['misc']['mass_center_pos'][1] - pelvis_Y
            projection_dict['misc' + '-' + 'mass_center_vel' + '-' + str(0)] = dict_['misc']['mass_center_vel'][0]
            projection_dict['misc' + '-' + 'mass_center_vel' + '-' + str(1)] = dict_['misc']['mass_center_vel'][1]
            if self.args[1]:
                assert len(projection_dict) == 35  # prosthetic
            else:
                assert len(projection_dict) == 38  # normal
            self.projection_dict = projection_dict
            projection = np.array(list(projection_dict.values()))
            return projection

        elif self.args.proj in ['3DPro37', '3D39']:
            # [0] = X - backward/forward
            # [1] = Y - down/up
            # [2] = Z - left/right
            projection_dict = {}
            projection_dict['body_pos-pelvis-1'] = dict_['body_pos']['pelvis'][1]
            # projection_dict['body_pos-pelvis-2'] = dict_['body_pos']['pelvis'][2]
            projection_dict['body_vel-pelvis-0'] = dict_['body_vel']['pelvis'][0]
            projection_dict['body_vel-pelvis-1'] = dict_['body_vel']['pelvis'][1]
            projection_dict['body_vel-pelvis-2'] = dict_['body_vel']['pelvis'][2]

            projection_dict['body_pos-head_Rel_pelv-0'] = dict_['body_pos']['head'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-head_Rel_pelv-1'] = dict_['body_pos']['head'][1] - dict_['body_pos']['pelvis'][1]
            projection_dict['body_pos-head_Rel_pelv-2'] = dict_['body_pos']['head'][2] - dict_['body_pos']['pelvis'][2]
            projection_dict['body_vel-head_Rel_pelv-0'] = dict_['body_vel']['head'][0] - dict_['body_vel']['pelvis'][0]
            projection_dict['body_vel-head_Rel_pelv-1'] = dict_['body_vel']['head'][1] - dict_['body_vel']['pelvis'][1]
            projection_dict['body_vel-head_Rel_pelv-2'] = dict_['body_vel']['head'][2] - dict_['body_vel']['pelvis'][2]

            if self.args.prosthetic:
                projection_dict['body_pos-pros_tibia_r_Rel_pelv-0'] = dict_['body_pos']['pros_tibia_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-pros_tibia_r_Rel_pelv-1'] = dict_['body_pos']['pros_tibia_r'][1] - dict_['body_pos']['pelvis'][1]
                projection_dict['body_vel-pros_tibia_r_Rel_pelv-0'] = dict_['body_vel']['pros_tibia_r'][0] - dict_['body_vel']['pelvis'][0]
                projection_dict['body_vel-pros_tibia_r_Rel_pelv-1'] = dict_['body_vel']['pros_tibia_r'][1] - dict_['body_vel']['pelvis'][1]
            else:
                projection_dict['body_pos-tibia_r_Rel_pelv-0'] = dict_['body_pos']['tibia_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-tibia_r_Rel_pelv-1'] = dict_['body_pos']['tibia_r'][1] - dict_['body_pos']['pelvis'][1]
                projection_dict['body_vel-tibia_r_Rel_pelv-0'] = dict_['body_vel']['tibia_r'][0] - dict_['body_vel']['pelvis'][0]
                projection_dict['body_vel-tibia_r_Rel_pelv-1'] = dict_['body_vel']['tibia_r'][1] - dict_['body_vel']['pelvis'][1]
            projection_dict['body_pos-tibia_l_Rel_pelv-0'] = dict_['body_pos']['tibia_l'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-tibia_l_Rel_pelv-1'] = dict_['body_pos']['tibia_l'][1] - dict_['body_pos']['pelvis'][1]
            projection_dict['body_vel-tibia_l_Rel_pelv-0'] = dict_['body_vel']['tibia_l'][0] - dict_['body_vel']['pelvis'][0]
            projection_dict['body_vel-tibia_l_Rel_pelv-1'] = dict_['body_vel']['tibia_l'][1] - dict_['body_vel']['pelvis'][1]

            if self.args.prosthetic:
                projection_dict['body_pos-pros_foot_r_Rel_pelv-0'] = dict_['body_pos']['pros_foot_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-pros_foot_r-1'] = dict_['body_pos']['pros_foot_r'][1]
                projection_dict['body_pos-pros_foot_r_Rel_pelv-2'] = dict_['body_pos']['pros_foot_r'][2] - dict_['body_pos']['pelvis'][2]
                projection_dict['body_vel-pros_foot_r-0'] = dict_['body_vel']['pros_foot_r'][0]
                projection_dict['body_vel-pros_foot_r-1'] = dict_['body_vel']['pros_foot_r'][1]
                projection_dict['body_vel-pros_foot_r-2'] = dict_['body_vel']['pros_foot_r'][2]
            else:
                projection_dict['body_pos-talus_r_Rel_pelv-0'] = dict_['body_pos']['talus_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-talus_r-1'] = dict_['body_pos']['talus_r'][1]
                projection_dict['body_pos-talus_r_Rel_pelv-2'] = dict_['body_pos']['talus_r'][2] - dict_['body_pos']['pelvis'][2]
                projection_dict['body_vel-talus_r-0'] = dict_['body_vel']['talus_r'][0]
                projection_dict['body_vel-talus_r-1'] = dict_['body_vel']['talus_r'][1]
                projection_dict['body_vel-talus_r-2'] = dict_['body_vel']['talus_r'][2]
            projection_dict['body_pos-talus_l_Rel_pelv-0'] = dict_['body_pos']['talus_l'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-talus_l-1'] = dict_['body_pos']['talus_l'][1]
            projection_dict['body_pos-talus_l_Rel_pelv-2'] = dict_['body_pos']['talus_l'][2] - dict_['body_pos']['pelvis'][2]
            projection_dict['body_vel-talus_l-0'] = dict_['body_vel']['talus_l'][0]
            projection_dict['body_vel-talus_l-1'] = dict_['body_vel']['talus_l'][1]
            projection_dict['body_vel-talus_l-2'] = dict_['body_vel']['talus_l'][2]

            projection_dict['body_pos-toes_l-1'] = dict_['body_pos']['toes_l'][1]
            projection_dict['body_pos-toes_l_Rel_talus_l-1'] = dict_['body_pos']['toes_l'][1] - dict_['body_pos']['talus_l'][1]
            if not self.args.prosthetic:
                projection_dict['body_pos-toes_r-1'] = dict_['body_pos']['toes_r'][1]
                projection_dict['body_pos-toes_r_Rel_talus_r-1'] = dict_['body_pos']['toes_r'][1] - dict_['body_pos']['talus_r'][1]

            projection_dict['mass_center_pos-Rel_pelv-0'] = dict_['misc']['mass_center_pos'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['mass_center_pos-Rel_pelv-1'] = dict_['misc']['mass_center_pos'][1] - dict_['body_pos']['pelvis'][1]
            projection_dict['mass_center_vel-Rel_pelv-0'] = dict_['misc']['mass_center_vel'][0] - dict_['body_vel']['pelvis'][0]
            projection_dict['mass_center_vel-Rel_pelv-1'] = dict_['misc']['mass_center_vel'][1] - dict_['body_vel']['pelvis'][1]

            projection_dict['body_pos-Rel_femur_l_femur_r-0'] = dict_['body_pos']['femur_l'][0] - dict_['body_pos']['femur_r'][0]  # turn left/right value (rotation around Y) - 0 = strait forward

            if self.args.prosthetic:
                assert len(projection_dict) == 37  # prosthetic
            else:
                assert len(projection_dict) == 39  # normal
            self.projection_dict = projection_dict
            projection = np.array(list(projection_dict.values()))
            return projection

        elif self.args.proj in ['3DPro37_2', '3D39_2']:
            # [0] = X - backward/forward
            # [1] = Y - down/up
            # [2] = Z - left/right
            projection_dict = {}
            projection_dict['body_pos-pelvis-1'] = dict_['body_pos']['pelvis'][1]
            # projection_dict['body_pos-pelvis-2'] = dict_['body_pos']['pelvis'][2]
            projection_dict['body_vel-pelvis-0'] = dict_['body_vel']['pelvis'][0]
            projection_dict['body_vel-pelvis-1'] = dict_['body_vel']['pelvis'][1]
            projection_dict['body_vel-pelvis-2'] = dict_['body_vel']['pelvis'][2]

            projection_dict['body_pos-head_Rel_pelv-0'] = dict_['body_pos']['head'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-head-1'] = dict_['body_pos']['head'][1]
            projection_dict['body_pos-head_Rel_pelv-2'] = dict_['body_pos']['head'][2] - dict_['body_pos']['pelvis'][2]
            projection_dict['body_vel-head-0'] = dict_['body_vel']['head'][0]
            projection_dict['body_vel-head-1'] = dict_['body_vel']['head'][1]
            projection_dict['body_vel-head-2'] = dict_['body_vel']['head'][2]

            if self.args.prosthetic:
                projection_dict['body_pos-pros_tibia_r_Rel_pelv-0'] = dict_['body_pos']['pros_tibia_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-pros_tibia_r_Rel_pelv-1'] = dict_['body_pos']['pros_tibia_r'][1] - dict_['body_pos']['pelvis'][1]
                projection_dict['body_vel-pros_tibia_r-0'] = dict_['body_vel']['pros_tibia_r'][0]
                projection_dict['body_vel-pros_tibia_r-1'] = dict_['body_vel']['pros_tibia_r'][1]
            else:
                projection_dict['body_pos-tibia_r_Rel_pelv-0'] = dict_['body_pos']['tibia_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-tibia_r_Rel_pelv-1'] = dict_['body_pos']['tibia_r'][1] - dict_['body_pos']['pelvis'][1]
                projection_dict['body_vel-tibia_r-0'] = dict_['body_vel']['tibia_r'][0]
                projection_dict['body_vel-tibia_r-1'] = dict_['body_vel']['tibia_r'][1]
            projection_dict['body_pos-tibia_l_Rel_pelv-0'] = dict_['body_pos']['tibia_l'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-tibia_l_Rel_pelv-1'] = dict_['body_pos']['tibia_l'][1] - dict_['body_pos']['pelvis'][1]
            projection_dict['body_vel-tibia_l-0'] = dict_['body_vel']['tibia_l'][0]
            projection_dict['body_vel-tibia_l-1'] = dict_['body_vel']['tibia_l'][1]

            if self.args.prosthetic:
                projection_dict['body_pos-pros_foot_r_Rel_pelv-0'] = dict_['body_pos']['pros_foot_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-pros_foot_r-1'] = dict_['body_pos']['pros_foot_r'][1]
                projection_dict['body_pos-pros_foot_r_Rel_pelv-2'] = dict_['body_pos']['pros_foot_r'][2] - dict_['body_pos']['pelvis'][2]
                projection_dict['body_vel-pros_foot_r-0'] = dict_['body_vel']['pros_foot_r'][0]
                projection_dict['body_vel-pros_foot_r-1'] = dict_['body_vel']['pros_foot_r'][1]
                projection_dict['body_vel-pros_foot_r-2'] = dict_['body_vel']['pros_foot_r'][2]
            else:
                projection_dict['body_pos-talus_r_Rel_pelv-0'] = dict_['body_pos']['talus_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-talus_r-1'] = dict_['body_pos']['talus_r'][1]
                projection_dict['body_pos-talus_r_Rel_pelv-2'] = dict_['body_pos']['talus_r'][2] - dict_['body_pos']['pelvis'][2]
                projection_dict['body_vel-talus_r-0'] = dict_['body_vel']['talus_r'][0]
                projection_dict['body_vel-talus_r-1'] = dict_['body_vel']['talus_r'][1]
                projection_dict['body_vel-talus_r-2'] = dict_['body_vel']['talus_r'][2]
            projection_dict['body_pos-talus_l_Rel_pelv-0'] = dict_['body_pos']['talus_l'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-talus_l-1'] = dict_['body_pos']['talus_l'][1]
            projection_dict['body_pos-talus_l_Rel_pelv-2'] = dict_['body_pos']['talus_l'][2] - dict_['body_pos']['pelvis'][2]
            projection_dict['body_vel-talus_l-0'] = dict_['body_vel']['talus_l'][0]
            projection_dict['body_vel-talus_l-1'] = dict_['body_vel']['talus_l'][1]
            projection_dict['body_vel-talus_l-2'] = dict_['body_vel']['talus_l'][2]

            projection_dict['body_pos-toes_l-0_Rel_pelv-0'] = dict_['body_pos']['toes_l'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['body_pos-toes_l-1'] = dict_['body_pos']['toes_l'][1]
            if not self.args.prosthetic:
                projection_dict['body_pos-toes_r-0_Rel_pelv-0'] = dict_['body_pos']['toes_r'][0] - dict_['body_pos']['pelvis'][0]
                projection_dict['body_pos-toes_r-1'] = dict_['body_pos']['toes_r'][1]

            projection_dict['mass_center_pos-Rel_pelv-0'] = dict_['misc']['mass_center_pos'][0] - dict_['body_pos']['pelvis'][0]
            projection_dict['mass_center_pos-Rel_pelv-1'] = dict_['misc']['mass_center_pos'][1] - dict_['body_pos']['pelvis'][1]
            projection_dict['mass_center_vel-0'] = dict_['misc']['mass_center_vel'][0]
            projection_dict['mass_center_vel-1'] = dict_['misc']['mass_center_vel'][1]

            projection_dict['body_pos-Rel_femur_l_femur_r-0'] = dict_['body_pos']['femur_l'][0] - dict_['body_pos']['femur_r'][0]  # turn left/right value (rotation around Y) - 0 = strait forward

            if self.args.prosthetic:
                assert len(projection_dict) == 37  # prosthetic
            else:
                assert len(projection_dict) == 39  # normal
            self.projection_dict = projection_dict
            projection = np.array(list(projection_dict.values()))
            return projection
        else:
            raise ValueError("{}: dict_to_vec conversion failed, unrecognized option '{}'".format(self.__class__.__name__,
                                                                                                   self.args.proj))

class RunEnv2HER(RunEnv2):  # semueller: Converts class RunEnv2 to baselines.her compatible env
    def __init__(self, goaltype='', tolerance=None, **runenv2_args):
        super(RunEnv2HER, self).__init__(**runenv2_args)
        self.goal = None

        self.tolerance = tolerance # TODO tune

    def dict_to_vec(self, dict_):
        if self.args.proj != '2Dpos':
            for _ in range(10):
                print("IS PELVIS_X ON THE CORRECT POSITION OF THE GOAL VECTOR?")
        proj = super(RunEnv2HER, self).dict_to_vec(dict_)

        # asummption that pelvis-X is the first entry of projection
        pelvis_diff = 0 if self.goal is None else proj[0] - self.goal[0]
        proj = np.concatenate((proj, [pelvis_diff]), 0)  # put diff at the end of the observation

        return proj

    @staticmethod
    def metric(p1, p2):
        ''':return: squared euclidean of np.arrays p1 and p2'''
        assert(p1.shape == p2.shape)
        if len(p1.shape) == 1:
            p1 = p1.reshape((1,)+p1.shape)
            p2 = p2.reshape((1,)+p2.shape)
        return np.sum(np.square((p1-p2)), axis=1)
        # return np.sum(np.square((p1-p2)))

    def _is_success(self, achieved_goal, desired_goal):
        if achieved_goal is None:
            raise ValueError("{}: achieved_goal was None".format(self.__class__))
        if desired_goal is None:
            return np.float32(0)
        d = self.metric(achieved_goal, desired_goal)
        # return np.float32(1.0)
        return np.float32(d <= self.tolerance)

    def step(self, action, project = True):
        obs, r, d, _ = super(RunEnv2HER, self).step(action, project)
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal)
        }
        return obs, r, d, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = self.metric(achieved_goal, desired_goal)
        return np.float32(dist <= self.tolerance)

    def get_observation(self):
        obs = super(RunEnv2HER, self).get_observation()
        achieved_goal = self.dict_to_vec(super(RunEnv2HER, self).get_state_desc())
        if self.goal is None:
            desired_goal = [0]*len(achieved_goal)
        else:
            desired_goal = list(self.goal)
        if self.tolerance is None:
            self.tolerance = 0.075*len(desired_goal)
            print('\t\t TOLERANCE SET TO {}'.format(self.tolerance))
        return {
            # the ddpg from baselines.her concatenates observations and achieved_goals itself
            # observation is quite large, |obs+goal| ~ 200 entries, might this be problematic?
            'observation': np.array(obs),
            'desired_goal': np.array(desired_goal),
            'achieved_goal': np.array(achieved_goal)
        }

    def reset(self, difficulty=0, seed=None):
        super(RunEnv2HER, self).reset(difficulty=difficulty, seed=seed)
        return self.get_observation()