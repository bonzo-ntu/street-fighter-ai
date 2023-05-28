
import os
import time

import retro
from stable_baselines3 import PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper
RESET_ROUND = True
RENDERING = False

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env
    return _init


stage_map = {
    1:'Champion.Level01.RyuVsRyu', #0
    2:'Champion.Level02.RyuVsHonda', #1
    3:'Champion.Level03.RyuVsBlanka',#2
    3.5:'Champion.Level03.5Bonus1',#3
    4:'Champion.Level04.RyuVsGuile',#3
    5:'Champion.Level05.RyuVsKen',#4
    6:'Champion.Level06.RyuVsChunLi',#5
    6.5:'Champion.Level06.5Bonus2',#6
    7:'Champion.Level07.RyuVsZangief',#6
    8:'Champion.Level08.RyuVsDalsim',#7
    9:'Champion.Level09.RyuVsBalrog',#8
    9.5:'Champion.Level09.5Bonus3',#9
    10:'Champion.Level10.RyuVsVega',#9
    11:'Champion.Level11.RyuVsSagat',#10
    12:'Champion.Level12.RyuVsBison',#11
    }
game = "StreetFighterIISpecialChampionEdition-Genesis"

for i in [1,2,3,3.5,4,5,6,6.5,7,8,9,9.5,10,11,12]:
    env = make_env(game, state=stage_map[i])()
    env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    print(f"{stage_map[i]}:{info['level']}")
    env.close()
