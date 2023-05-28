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
from rewarder import CustomRewarder

import gym
import numpy as np
import cv2

NUM_STAGES = 12
WIN_GAME, LOSE_GAME = 1, -1
BONUS_END = 2

# --> bonzo add
def downsample(input, rate=2):
    return input[::rate, ::rate, :]
# <--

# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, rd_type="default", reset_round=1, rendering=False):
        assert reset_round in [0, 1, 3] # [no_reset, 1/1, 3/2]
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env
        self.prev_obs = self.preprocess(self.env.reset())
        self.init_info = self.get_curr_all_info()

        # Use a deque to store the last 5 frame difference (current frame - previous frame)
        # that equals 6 frams information. 
        self.num_frames = 5
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # the shortes RYU's move has frame counts == 3
        self.num_step_frames = 3 

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = self.init_info["agent_hp"] # 總血量

        # rd_info 為了計算 reward 用的資訊
        self.rd_info = {}
        self.init_rd_info()

        # CustomRewarder.get_available_rd_types()
        # ["default", "time", "score", "time+score"]
        self.rewarder = CustomRewarder(rd_type=rd_type, rd_coeff=self.reward_coeff, full_hp=self.full_hp, init_info_dict=self.rd_info)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering

        ## 算勝場(需處理過場動畫)
        self.win_games, self.lose_games = 0, 0
        self.win_rounds, self.lose_rounds = 0, 0
        self.is_in_ending = False
        self.tmp_win_rounds, self.tmp_lose_rounds = 0, 0
        self.reset_condition = self.reset_round/2 if reset_round != 0 else 3/2 # noreset時，三戰兩勝輸的時候還是要Reset

        ## 記錄過幾關
        self.stages = 0

    
    def _stack_observation(self):
        # 拿 frame 2 的 R + frame 5 的 G + frame 8 的 B 串在一起
        # return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

        # 全部拿來用
        return self.frame_stack

    def preprocess(self, observation): 
        # Grayscaling 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize 
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84,84,1))
        return channels 


    # 根據記錄的最後一個 process 過的畫面，跟現在process過的畫面算 diff
    def diff_with_prev_obs(self, curr_obs):
        diff_obs = curr_obs - self.prev_obs
        self.prev_obs = curr_obs
        return diff_obs

    def reset(self):
        #obs = self.env.reset()
        self.prev_obs = self.preprocess(self.env.reset())

        # 關卡重置
        self.init_info = self.get_curr_all_info()
        self.init_rd_info()
        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()

        #self.frame_stack.append(self.preprocess(prev_obs))
        no_action =  [0,0,0,0,0,0,0,0,0,0,0,0]
        for _ in range(self.num_frames):
            obs, dummy_reward, dummy_done, info  = self.env.step(no_action)
            curr_obs = self.preprocess(obs)
            diff_obs = self.diff_with_prev_obs(curr_obs)

            # 初始放了 5 個前後 frame 的差，等價 6 個 frame
            self.frame_stack.append(diff_obs) # down sample

        # --> bonzo replace
        # return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
        return self._stack_observation()
        # <--

    def step(self, action):
        custom_done = False
        is_reward_stage = False
        done_status = 0 # 用於回傳給外部計算勝率用的 done_status

        obs, _reward, _done, info = self.env.step(action)
        curr_obs = self.preprocess(obs)
        ## "永不 reset" 情況下，但是全通關，就要 reset
        if not self.reset_round and self.stages >= NUM_STAGES:
            info["done_status"] = 1
            return self._stack_observation(), 0, True, info
        
        # 時間沒在動 = 跑動畫，所以跳過 # "永不 reset" 情況下可能全通關造成無窮迴圈，但是上面排除了全通關的情況
        while self.rd_info["curr_countdown"] - self.rd_info["prev_countdown"] == 0:
            obs, _, _, info = self.env.step(action)
            self.rd_info["prev_countdown"] = self.rd_info["curr_countdown"]
            self.rd_info["curr_countdown"] = info['round_countdown']
            #print("SKIP")
        # --> bonzo replace
        # self.frame_stack.append(obs[::2, ::2, :])
        self.frame_stack.append(self.diff_with_prev_obs(curr_obs))
        # <--

        # 做 action 之前先渲染畫面
        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render(mode='rgb_array')
            time.sleep(0.01) # 1 個 frame 是 0.0166 秒


        # 同個 action 持續 3 個 frame
        for _ in range(self.num_step_frames - 1):
            
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, info = self.env.step(action)
            curr_obs = self.preprocess(obs)
            self.frame_stack.append(self.diff_with_prev_obs(curr_obs))


            # 渲染這 6 個 frame
            if self.rendering:
                self.env.render(mode='rgb_array')
                time.sleep(0.01)
        
        # 如果是獎勵關卡，時間分母要改。
        if info['enemy_status'] == 0: is_reward_stage = True
        self.rd_info["init_countdown"] = 16424 if is_reward_stage else 39208
        self.rd_info["curr_player_health"] = info['agent_hp']
        self.rd_info["curr_oppont_health"] = info['enemy_hp']
        self.rd_info["curr_countdown"] = info['round_countdown']
        self.rd_info["curr_score"] = info["score"]

        self.total_timesteps += self.num_step_frames

        # 算 reward 之前，先更新 Rewarder 的 rd_info
        self.rewarder.update(self.rd_info)

        #####  計算 Reward  #####
        # custom_done: 用於訓練最後一關第一局
        # Round is over and player loses.
        ## 時間到了，但是還沒結束，所以要結束 Round
        timeup, timeup_win = False, False
        if self.rd_info["curr_countdown"] <= 0 and not is_reward_stage:
            timeup = True
            if self.rd_info["curr_player_health"] > self.rd_info["curr_oppont_health"]:
                timeup_win = True

        # 時間到不算輸贏，只算時間到的Reward並重置變數rd_info
        if timeup:
            custom_reward = self.rewarder.fight()
            if timeup_win: self.tmp_win_rounds += 1
            else: self.tmp_lose_rounds += 1
            self.init_rd_info()
        # Round結束，玩家輸了
        if self.rd_info["curr_player_health"] < 0:
            custom_reward = self.rewarder.lose()    # Use the remaining health points of opponent as penalty. 
                                                   # If the opponent also has negative health points, it's a even game and the reward is +1.
            self.lose_rounds += 1
            self.tmp_lose_rounds += 1
            self.init_rd_info()

        # Round結束，玩家贏了
        elif self.rd_info["curr_oppont_health"] < 0:
            custom_reward = self.rewarder.win()
            self.win_rounds += 1
            self.tmp_win_rounds += 1
            self.init_rd_info()

        # While the fighting is still going on
        else:
            custom_reward = self.rewarder.fight()
            self.rd_info["prev_player_health"] = self.rd_info["curr_player_health"]
            self.rd_info["prev_oppont_health"] = self.rd_info["curr_oppont_health"]
            self.rd_info["prev_countdown"] = self.rd_info["curr_countdown"]
            self.rd_info["prev_score"] = self.rd_info["curr_score"]


        # end when round_countdown <= 0
        ## 時間到了，但是還沒結束，所以要結束局 (但是算玩家輸嗎？算的話要移到輸了計算Loss的地方)
        if self.rd_info["curr_countdown"] <= 0:
            # custom_done = True
            self.tmp_lose_rounds += 1

        ## 需不需要結算遊戲，看看是不是需要 reset 了
        result = self.is_end_of_game(is_reward_stage=is_reward_stage)
        if result == WIN_GAME: ## 贏GAME
            self.win_games += 1
            self.stages += 1
            custom_done = True
            done_status = 1
        elif result == LOSE_GAME: ## 輸GAME
            self.lose_games += 1
            self.stages = 0
            custom_done = True
            done_status = -1
        elif result == BONUS_END: # BONUS關贏了就是一Round一關結束，但是不算關卡數，且需要game_end_process並繼續
            pass
        
        if result:
            self.game_end_process()

        if result == WIN_GAME and not self.reset_round:
            custom_done = False # reset == 0 代表 "贏的情況永不 reset"
            done_status = 0
        
        info["done_status"] = done_status
        # Max reward is 3 * full_hp = 528, norm_coefficient = 0.001, MAX_REWARD = 0.528
        return self._stack_observation(), custom_reward, custom_done, info # reward normalization
    
    # Tools for Rewarder
    def init_rd_info(self):
        # 初始化血量
        self.rd_info["prev_player_health"] = self.full_hp
        self.rd_info["prev_oppont_health"] = self.full_hp
        self.rd_info["curr_player_health"] = self.full_hp
        self.rd_info["curr_oppont_health"] = self.full_hp
        # 初始化倒數計時
        self.rd_info["prev_countdown"] = self.init_info["round_countdown"]
        self.rd_info["curr_countdown"] = self.init_info["round_countdown"]
        # 初始化分數
        self.rd_info["prev_score"] = 0
        self.rd_info["curr_score"] = 0



    # Tools
    def get_curr_all_info(self):
        return self.env.data.lookup_all()
    def game_end_process(self):
        self.tmp_lose_rounds, self.tmp_win_rounds = 0, 0
        self.init_rd_info()
    def is_end_of_game(self, is_reward_stage=False):
        if is_reward_stage and self.rd_info["curr_countdown"] <= 0: return BONUS_END
        if self.tmp_lose_rounds > self.reset_condition: return LOSE_GAME
        elif self.tmp_win_rounds > self.reset_condition: return WIN_GAME
        else: return 0
    def winrate(self, how_count="round"):
        if how_count == "round":
            return self.win_rounds / (self.win_rounds + self.lose_rounds)
        elif how_count == "game":
            return self.win_games / (self.win_games + self.lose_games)
        else:
            return -1