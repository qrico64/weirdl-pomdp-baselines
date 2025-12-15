from .td3 import TD3
from .sac import SAC
from .sacd import SACD
from .iql import IQL

RL_ALGORITHMS = {
    TD3.name: TD3,
    SAC.name: SAC,
    SACD.name: SACD,
    IQL.name: IQL,
}


RL_ALGORITHM_PROPERTIES = {
    'use_value_fn': {
        TD3.name: False,
        SAC.name: False,
        SACD.name: False,
        IQL.name: True,
    }
}
