from flax import struct
import chex
from config import env_config

# @struct.dataclass
# class TeamState:

@struct.dataclass
class UnitState:
    unit_id: int
    health_current: float
    health_max: float
    location_x: int
    location_y: int
    melee_attack_base_damage: float
    ranged_attack_base_damage: float
    melee_attack_range: float
    ranged_attack_range: float
    movement_points_current: float
    movement_points_max: float
    action_points_current: float
    action_points_max: float
    available_actions: chex.Array

@struct.dataclass
class EnvState:
    # flatten this?
    # board: chex.Array
    player: UnitState
    enemy: UnitState
    distance_to_enemy: float
    steps: int
    turn_count: int
    previous_closest_distance: float
    initial_distance: float
    # enemy_network_params: XX

@struct.dataclass
class EnvParams:
    # Configuration for the environment
    BOARD_SIZE: int = env_config['BOARD_SIZE']
    MAX_HEALTH: float = env_config['MAX_HEALTH']
    MELEE_DAMAGE: float = env_config['MELEE_DAMAGE']
    RANGED_DAMAGE: float = env_config['RANGED_DAMAGE']
    MELEE_RANGE: float = env_config['MELEE_RANGE']
    RANGED_RANGE: float = env_config['RANGED_RANGE']
    MOVEMENT_POINTS: float = env_config['MOVEMENT_POINTS']
    ACTION_POINTS: float = env_config['ACTION_POINTS']
    MAX_STEPS: int = env_config['MAX_STEPS']