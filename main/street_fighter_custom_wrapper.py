# Copyright 2023 LIN Yi. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import time
import collections

import gym
import numpy as np

# --> bonzo add
def downsample(input, rate=2):
    return input[::rate, ::rate, :]
# <--

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176 # 總血量
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
    
    def _stack_observation(self):
        # 拿 frame 2 的 R + frame 5 的 G + frame 8 的 B 串在一起
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            # 放了 9 個一樣的畫面
            # --> bonzo replace
            # self.frame_stack.append(observation[::2, ::2, :])
            self.frame_stack.append(downsample(observation)) # down sample
            # <--

        # --> bonzo replace
        # return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        return self._stack_observation()
        # <--

    def step(self, action):
        custom_done = False

        obs, _reward, _done, info = self.env.step(action)
        # --> bonzo replace
        # self.frame_stack.append(obs[::2, ::2, :])
        self.frame_stack.append(downsample(obs))
        # <--

        # 做 action 之前先渲染畫面
        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render(mode='rgb_array')
            time.sleep(0.01) # 1 個 frame 是 0.0166 秒


        # 同個 action 持續 6 個 frame
        for _ in range(self.num_step_frames - 1):
            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, info = self.env.step(action)
            # --> bonzo replace
            #self.frame_stack.append(obs[::2, ::2, :])
            self.frame_stack.append(downsample(obs)) # 記錄相同 action 之下的 6 個 frame
            # <--

            # 渲染這 6 個 frame
            if self.rendering:
                self.env.render(mode='rgb_array')
                time.sleep(0.01)

        curr_player_health = info['agent_hp']
        curr_oppont_health = info['enemy_hp']
        round_countdown = info['round_countdown']
        
        self.total_timesteps += self.num_step_frames

        # custom_done: 用於訓練最後一關第一局
        
        # Game is over and player loses.
        if curr_player_health < 0:
            custom_reward = -math.pow(self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1))    # Use the remaining health points of opponent as penalty. 
                                                   # If the opponent also has negative health points, it's a even game and the reward is +1.
            custom_done = True

        # Game is over and player wins.
        elif curr_oppont_health < 0:
            # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
                                                                   # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            custom_reward = math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (self.prev_oppont_health - curr_oppont_health) - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False
        
        # end when round_countdown <= 0
        # this info does not match with the time on the screen
        if round_countdown <= 0:
            custom_done = True

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False
             
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), 0.001 * custom_reward, custom_done, info # reward normalization
    