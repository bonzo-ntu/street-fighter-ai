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

    def get_essential_infokeys(self):
        default_keys = ["prev_oppont_health", "curr_oppont_health", "prev_player_health", "curr_player_health"]

        if self.rd_type == "default":
            return default_keys
        elif self.rd_type == "time" or self.rd_type == "time2":
            return default_keys + ["init_countdown", "curr_countdown"]
        elif self.rd_type == "score":
            return ["prev_score", "curr_score"]
        elif self.rd_type == "time+score":
            return ["init_countdown", "curr_countdown"] + ["prev_score", "curr_score"]
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
    
    def get_rd_lose(self):
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
        default_lose = -math.pow(self.full_hp, (self.curr_info_dict["curr_oppont_health"] + 1) / (self.full_hp + 1))

        if self.rd_type == "default": # 林亦原本的 Reward
            lose = default_lose
        elif self.rd_type == "time" or self.rd_type == "time2": # 自訂的 Reward 
            # countdown_coeff = (1+self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"])
            lose = default_lose #* countdown_coeff
        elif self.rd_type == "score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            lose = score
        elif self.rd_type == "time+score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            countdown_coeff = self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"]
            lose = score * countdown_coeff
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
        
        return lose

    def get_rd_win(self):
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
        default_win = math.pow(self.full_hp, (self.curr_info_dict["curr_player_health"] + 1) / (self.full_hp + 1)) * self.rd_coeff
        
        if self.rd_type == "default": # 林亦原本的 Reward
            win = default_win
        elif self.rd_type == "time" or self.rd_type == "time2": # 自訂的 Reward 
            # countdown_coeff = (1+self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"])
            win = default_win #* countdown_coeff
        elif self.rd_type == "score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            win = score
        elif self.rd_type == "time+score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            countdown_coeff = self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"]
            win = score * countdown_coeff
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
        
        return win
    
    def get_rd_fight(self):
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
        default_fight = self.rd_coeff * (self.curr_info_dict["prev_oppont_health"] - self.curr_info_dict["curr_oppont_health"]) - \
                        (self.curr_info_dict["prev_player_health"] - self.curr_info_dict["curr_player_health"])
        if self.rd_type == "default": # 林亦原本的 Reward
            fight = default_fight
        elif self.rd_type == "time": # 自訂的 Reward 
            countdown_coeff = (1+self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"])
            countdown_punish = 1 - countdown_coeff
            # fight = self.rd_coeff * (self.curr_info_dict["prev_oppont_health"] - self.curr_info_dict["curr_oppont_health"]) * countdown_coeff - \
                        # (self.curr_info_dict["prev_player_health"] - self.curr_info_dict["curr_player_health"]) - \
                        # 176/10 #* countdown_punish
            fight = default_fight * countdown_coeff - 176/10
        elif self.rd_type == "time2":
            countdown_coeff = (1+self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"])
            countdown_punish = 1 - countdown_coeff
            fight = self.rd_coeff * (self.curr_info_dict["prev_oppont_health"] - self.curr_info_dict["curr_oppont_health"]) * countdown_coeff - \
                        (self.curr_info_dict["prev_player_health"] - self.curr_info_dict["curr_player_health"]) - \
                        176/5 * countdown_punish
        elif self.rd_type == "score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            fight = score
        elif self.rd_type == "time+score":
            score = self.curr_info_dict["curr_score"] - self.curr_info_dict["prev_score"]
            countdown_coeff = self.curr_info_dict["curr_countdown"] / self.curr_info_dict["init_countdown"]
            fight = score * countdown_coeff
        else:
            raise ValueError(f"rd_type should be {self.get_available_rd_types()}")
        
        return fight
    
    def norm(self, rd):
        # 因為已經跑實驗了，所以就把 Normalized 的 Reward 上界定成 0.5
        default_norm = 0.001
        if self.rd_type == "default":
            norm = default_norm
        elif self.rd_type == "time" or self.rd_type == "time2":
            norm = default_norm/2
        elif self.rd_type == "score":
            norm = 1 # 分數沒有明顯上下界，所以就不 normalize 了
        elif self.rd_type == "time+score":
            norm = 1 # 分數沒有明顯上下界，所以就不 normalize 了
        return rd * norm

    ## Update curr_info_dict
    def update(self, info_dict):
        self.curr_info_dict = info_dict
        # 只更新除 'level' 之外的 data
        ##old_level = self.curr_info_dict['level']
        #self.curr_info_dict = {k:v for k, v in info_dict.items() if k != 'level'}
        #self.curr_info_dict['level'] = old_level
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
    
    ## Get normalized rewards
    def lose(self):
        return self.norm(self.get_rd_lose())
    def win(self):
        return self.norm(self.get_rd_win())
    def fight(self):
        return self.norm(self.get_rd_fight())
    
    ## Tools
    @staticmethod
    def get_available_rd_types():
        return ["default", "time", "time2", "score", "time+score"]