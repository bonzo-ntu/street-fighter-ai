# PPO algo for RL
from stable_baselines3 import PPO
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy
# Import the sb3 monitor for logging 
from stable_baselines3.common.monitor import Monitor
# Import the vec wrappers to vectorize and frame stack
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import SubprocVecEnv
# Import os to deal with filepaths
import os
import sys

import retro
from street_fighter_custom_wrapper import StreetFighterCustomWrapper

model_name = "PPO-delta"

LOG_DIR = './logs/'
OPT_DIR = "opt_" + model_name
MODEL_DIR = "trained_models_" + model_name

NUM_ENV = 4


# Import base callback 
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )
        env = StreetFighterCustomWrapper(env)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():
    # Set up the environment and model
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = SubprocVecEnv([make_env(game, state="Champion.Level12.RyuVsBison", seed=i) for i in range(NUM_ENV)])
    # Create environment 
    # env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis',
    #                 state="Champion.Level12.RyuVsBison", 
    #                 use_restricted_actions=retro.Actions.FILTERED)
    # env = StreetFighterCustomWrapper(env)
    # env = Monitor(env, LOG_DIR)
    # env = DummyVecEnv([lambda: env])
    # env = VecFrameStack(env, 4, channels_order='last')

    # model_params = study.best_params
    model_params = {'n_steps': 5568,
                    'gamma': 0.841615894986323,
                    'learning_rate': 2.4877289885597835e-05,
                    'clip_range': 0.2622852669044159,
                    'gae_lambda': 0.8517353230777983}
    # model_params['learning_rate'] = 5e-7

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)

    # Reload previous weights from HPO
    model.load(os.path.join(OPT_DIR, 'trial_6_best_model.zip'))

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=MODEL_DIR)

    # Train the model
    original_stdout = sys.stdout
    log_file_path = os.path.join(MODEL_DIR, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(total_timesteps=100000, callback=callback)
        # model.learn(total_timestep=5000000)
        
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

if __name__ == "__main__":
    main()