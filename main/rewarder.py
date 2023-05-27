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
        if self.rd_type == "default":
            return ["prev_oppont_health", "curr_oppont_health", "prev_player_health", "curr_player_health"]
        elif self.rd_type == "custom":
            return ["prev_oppont_health", "curr_oppont_health", "prev_player_health", "curr_player_health"]
        else:
            raise ValueError("rd_type should be default or custom")
    def get_rd_func(self):
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
        if self.rd_type == "default":
            lose = -math.pow(self.full_hp, (self.curr_info_dict["curr_oppont_health"] + 1) / (self.full_hp + 1))
            win = math.pow(self.full_hp, (self.curr_info_dict["curr_player_health"] + 1) / (self.full_hp + 1)) * self.rd_coeff
            keep_fight = self.rd_coeff * (self.curr_info_dict["prev_oppont_health"] - self.curr_info_dict["curr_oppont_health"]) - \
                        (self.curr_info_dict["prev_player_health"] - self.curr_info_dict["curr_player_health"])
            return lose, win, keep_fight
        elif self.rd_type == "custom":
            return self.custom_lose, self.custom_win, self.custom_keep_fight
        else:
            raise ValueError("rd_type should be default or custom")
    def update(self, info_dict):
        self.curr_info_dict = info_dict
        assert keys_are_in_dict(self.essential_infokeys, self.curr_info_dict)
    
    ## Get rewards
    def lose(self):
        return self.lose_rd
    def win(self):
        return self.win_rd
    def fight(self):
        return self.fight_rd