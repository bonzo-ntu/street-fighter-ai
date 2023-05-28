# %%
import os
os.chdir('../main')

# %%
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

import os
import sys

import retro
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

from street_fighter_custom_wrapper import StreetFighterCustomWrapper

NUM_ENV = 8
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# used for saving checkpoints and logs
model_name = "DQN"

# %%
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.DISCRETE, 
            obs_type=retro.Observations.IMAGE    
        )
        print("button_combos", env.button_combos)
        print("buttons", env.buttons)
        env = StreetFighterCustomWrapper(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

# %%
import wandb
from wandb.integration.sb3 import WandbCallback
import os
if 'WANDB_BASE_URL' in os.environ:
    del os.environ['WANDB_BASE_URL']
# 先 login, 如果要切換帳號可以在 command line 打 `wandb login --relogin`
# 這行會要求你去 https://wandb.ai/authorize 拿取屬於自己的 api key
# 複製 api key 貼上 (即使是在同個 team 不同人會拿到不同的 key)
wandb.login()

# 我們的 team 是 'ntuai2023'，這個要設在 `wandb.init()` 的 `entity` 參數
# team 底下的 project 是`sf2`
config = {
    'algo':'DQN',
    'total_timesteps':int(7.5e6),
    'stage':'Level12.RyuVsBison',

}
WANDB_PROJECT='sf2'
run = wandb.init(project=WANDB_PROJECT, 
                 entity='ntuai2023', 
                 sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                 monitor_gym=True,  # auto-upload the videos of agents playing the game
                 save_code=True,  # optional
                 settings=wandb.Settings(start_method="fork"))

# %%
#def main():
if __name__=='__main__':
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = SubprocVecEnv([make_env(game, state="Champion.Level12.RyuVsBison", seed=i) for i in range(NUM_ENV)])
    #env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % (2000*3000) == 0, video_length=200*3000)

    # Set linear schedule for learning rate
    # Start
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    # Start
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    model = DQN(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        buffer_size = 20_000,
        # n_steps=512,
        #train_freq=(128, "step"),
        batch_size=512,
        # n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        #clip_range=clip_range_schedule,
        #tensorboard_log="logs"
        tensorboard_log=f"runs/{run.id}"
    )


    # Set the save directory
    save_dir = "trained_models_" + model_name
    os.makedirs(save_dir, exist_ok=True)

    # Load the model from file
    # model_path = "trained_models/ppo_ryu_7000000_steps.zip"
    
    # Load model and modify the learning rate and entropy coefficient
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule,
    #     "n_steps": 512
    # }
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set up callbacks
    # Note that 1 timesetp = 6 frame
    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=f"{save_dir}/{run.id}", name_prefix=model_name)
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_freq = checkpoint_interval,
        model_save_path=f"{save_dir}/{run.id}",
        #verbose=2,
    )

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(f"{save_dir}/{run.id}", "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=config['total_timesteps'], # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[wandb_callback, checkpoint_callback],#, stage_increase_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, model_name+".zip"))
    print(f'model saved to {os.path.join(save_dir, model_name+".zip")}')

    print('end of learning')

# %%



