import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Street Fighter AI args')
    # define game and state
    parser.add_argument('--game', type=str, default='StreetFighterIISpecialChampionEdition-Genesis', help='Game to play')
    parser.add_argument('--state', type=str, default='Champion.Level12.RyuVsBison', help='Game state to play')
    # for training
    parser.add_argument('--num_env', type=int, default=16, help='Number of environments')
    parser.add_argument('--n_steps', type=int, default=512, help='Number of steps')
    parser.add_argument('--total_timesteps', '-tt', type=int, default=10000000, help='Number of total timesteps')
    parser.add_argument('--check_timesteps', '-ct', type=int, default=500000, help='Number of timesteps to check')
    # for store results
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--train_name', type=str, default='PPO', help='Name to be added to the path that saves models')

    return parser.parse_args()
