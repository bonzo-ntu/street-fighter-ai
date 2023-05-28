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
        elif self.rd_type == "time":
            return default_keys + ["init_countdown", "curr_countdown"]
        elif self.rd_type == "score":
            return ["prev_score", "curr_score"]
        elif self.rd_type == "time+score":
            return ["init_countdown", "curr_countdown"] + ["prev_score", "curr_score"]
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
        elif self.rd_type == "time": # 自訂的 Reward 
            countdown_coeff = self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"]
            lose = default_lose * countdown_coeff
            win = default_win * countdown_coeff
            fight = default_fight * countdown_coeff
        elif self.rd_type == "score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            lose = score
            win = score
            fight = score
        elif self.rd_type == "time+score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            countdown_coeff = self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"]
            lose = score * countdown_coeff
            win = score * countdown_coeff
            fight = score * countdown_coeff
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
        
        return lose, win, fight
    
    def norm(self, rd):
        # 因為已經跑實驗了，所以就把 Normalized 的 Reward 上界定成 0.5
        default_norm = 0.001
        if self.rd_type == "default":
            norm = default_norm
        elif self.rd_type == "time":
            norm = default_norm
        elif self.rd_type == "score":
            norm = 1 # 分數沒有明顯上下界，所以就不 normalize 了
        elif self.rd_type == "time+score":
            norm = 1 # 分數沒有明顯上下界，所以就不 normalize 了
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
        return ["default", "time", "score", "time+score"]