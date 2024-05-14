from gymnasium.envs.registration import register

register(
    id='MotionPlanning-v0',
    entry_point='gym_auv.envs:MPv0',
)

register(
    id='ObstacleAvoidance-v0',
    entry_point='gym_auv.envs:OAv0',
)

register(
    id='Docking-v2',
    entry_point='gym_auv.envs:Dockingv2',
)