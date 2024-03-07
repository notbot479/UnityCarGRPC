from enum import Enum


class RewardPolicy(Enum):
    TARGET_IS_FOUND = 0.3
    SHORTEN_DISTANCE_TO_BOX = 0.2
    SHORTEN_DISTANCE_TO_ROUTER = 0.1
    IN_TARGET_AREA_BOXES_FOUND = 0.05
    PASSIVE_REWARD = -0.01
    IN_TARGET_AREA_NO_BOXES_FOUND = -0.05
    INCREASE_DISTANCE_TO_ROUTER = -0.1
    INCREASE_DISTANCE_TO_BOX = -0.2
    HIT_WALL = -0.3

def _test():
    reward = RewardPolicy.SHORTEN_DISTANCE_TO_ROUTER.value
    print(reward)

if __name__ == '__main__':
    _test()
