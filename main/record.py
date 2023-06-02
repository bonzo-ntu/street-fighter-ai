import os
import time 

import retro
from stable_baselines3 import PPO

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            record='.'
        )
        env = StreetFighterCustomWrapper(env, reset_round=game_type, rendering=RENDERING)
        return env
    return _init


MODEL_DIR = "trained_models_PPO-eval-2"
model_name = "PPO-eval-2"
checkpoint = 7500000 # list of checkpoints to evaluate

game_type = 3 # in [0, 1, 3] # [no_reset(1~12關), 一局, 三局兩勝]
RENDERING = True    # Whether to render the game screen.
RANDOM_ACTION = False

game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsBison")()

model_checkpoints = model_name + "_" + str(checkpoint) + "_steps" # Speicify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.

if not RANDOM_ACTION:
    model = PPO.load(os.path.join(MODEL_DIR, model_checkpoints), env=env)

experiment_reward_sum = 0
num_victory = 0

if RANDOM_ACTION:
    print("Random action")
else:
    print(model_checkpoints)
print("\nFighting Begins!\n")

done = False
obs = env.reset()

# env.render(mode='rgb_array')

while not done:
    timestamp = time.time()

    if RANDOM_ACTION:
        obs, reward, done, info = env.step(env.action_space.sample())
    else:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

    # env.render(mode='rgb_array')

print("\nFighting Ends!\n")


env.close()