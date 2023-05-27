import math

def keys_are_in_dict(keys, dict):
    for key in keys:
        if key not in dict:
            return False
    return True

class CustomRewarder:
    def __init__(self, rd_type="default", rd_coeff=3, full_hp=176,  init_info_dict=None):
        self.rd_type = rd_type
        self.rd_coeff = rd_coeff

        self.full_hp = full_hp
        self.curr_info_dict = init_info_dict

        # need to determine the rd_type
        self.essential_infokeys = self.get_essential_infokeys()
        self.lose_rd, self.win_rd, self.fight_rd = self.get_rd_func()

    def get_essential_infokeys(self):
        default_keys = ["prev_oppont_health", "curr_oppont_health", "prev_player_health", "curr_player_health"]

        if self.rd_type == "default":
            return default_keys
        elif self.rd_type == "custom":
            return default_keys + ["init_countdown", "curr_countdown"]
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
    
    def get_rd_func(self):
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
        default_lose = -math.pow(self.full_hp, (self.curr_info_dict["curr_oppont_health"] + 1) / (self.full_hp + 1))
        default_win = math.pow(self.full_hp, (self.curr_info_dict["curr_player_health"] + 1) / (self.full_hp + 1)) * self.rd_coeff
        default_fight = self.rd_coeff * (self.curr_info_dict["prev_oppont_health"] - self.curr_info_dict["curr_oppont_health"]) - \
                        (self.curr_info_dict["prev_player_health"] - self.curr_info_dict["curr_player_health"])
        if self.rd_type == "default": # 林亦原本的 Reward
            lose = default_lose
            win = default_win
            fight = default_fight
        elif self.rd_type == "custom": # 自訂的 Reward 
            countdown_coeff = self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"]
            lose = default_lose * countdown_coeff
            win = default_win * countdown_coeff
            fight = default_fight * countdown_coeff
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
        
        return lose, win, fight
    
    def norm(self, rd):
        default_norm = 0.001
        if self.rd_type == "default":
            norm = default_norm
        elif self.rd_type == "custom":
            norm = default_norm # TBD
        return rd * norm

    ## Update curr_info_dict
    def update(self, info_dict):
        self.curr_info_dict = info_dict
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
    
    ## Get normalized rewards
    def lose(self):
        return self.norm(self.lose_rd)
    def win(self):
        return self.norm(self.win_rd)
    def fight(self):
        return self.norm(self.fight_rd)
    
    ## Tools
    @staticmethod
    def get_available_rd_types():
        return ["default", "custom"]