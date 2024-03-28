from enum import Enum


class RewardPolicy(Enum):
    TARGET_IS_FOUND = 0.5
    TARGET_ROUTER_SWITCHED = 0.5
    SHORTEN_DISTANCE_TO_BOX = 0.3
    IN_TARGET_AREA_BOXES_FOUND = 0.05
    PASSIVE_REWARD = -0.025
    IN_TARGET_AREA_NO_BOXES_FOUND = -0.05
    INCREASE_DISTANCE_TO_BOX = -0.3
    HIT_OBJECT = -0.5

def _test():
    reward = RewardPolicy.PASSIVE_REWARD.value
    print(reward)

if __name__ == '__main__':
    _test()
