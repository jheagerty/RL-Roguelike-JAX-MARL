# data_classes.py
from flax import struct
import chex
from config import env_config
import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Tuple, Callable
from enum import Enum, auto

class AttackType(Enum):
    MELEE = auto()
    RANGED = auto()

class DamageType(Enum):
    PHYSICAL = auto()
    MAGICAL = auto()
    PURE = auto()

float_max = jnp.finfo(jnp.float32).max

schema = { # TODO: make function ? so ability index can be dynamic
  "UnitsState": {
    "unit_id": {
        "type": jnp.ndarray,
        "default": jnp.arange(env_config['HEROES_PER_TEAM'] * 2, dtype=jnp.int32),
        "obs": False
    },
    "location": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 2), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (x, y)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 2), dtype=jnp.int32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 2), 20, dtype=jnp.int32),
    },
    "action_points": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([3., 3., 5.], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (base, current, max)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 3), dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 3), 20, dtype=jnp.float32),
    },
    "movement_points": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([10., 10., 10., 1., 1.], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (base, current, max, percentage, multiplier)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 5), dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 5), 50, dtype=jnp.float32),
    },
    "health": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([100., 100., 1., 0., 0.], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (current, max, percentage, regeneration, regeneration_rate)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 5), dtype=jnp.float32), # TODO fix to tile, regen can be negative, rate between -1 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 5), 1000, dtype=jnp.float32), # TODO fix to tile, rate between -1 and 1
    },
    "mana": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([100., 100., 1., 0., 0.], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (current, max, percentage, regeneration, regeneration_rate)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 5), dtype=jnp.float32), # TODO fix to tile, regen can be negative, rate between -1 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 5), 1000, dtype=jnp.float32), # TODO fix to tile, rate between -1 and 1
    },
    "barrier": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([10., 100., 0.1, 0., 0., 0.], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (current, max, percentage, regeneration, regeneration_rate, layers)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 6), dtype=jnp.float32), # TODO fix to tile, regen can be negative, rate between -1 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), 1000, dtype=jnp.float32), # TODO fix to tile, rate between -1 and 1
    },
    "physical_defence": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 6), dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (block, resist, immunity, evasion, damage_return, damage_return_rate)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), -1000, dtype=jnp.float32), # TODO fix to tile, eg evasion between 0 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), 1000, dtype=jnp.float32), # TODO fix to tile, eg evasion between 0 and 1
    },
    "magical_defence": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 6), dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (block, resist, immunity, evasion, damage_return, damage_return_rate)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), -1000, dtype=jnp.float32), # TODO fix to tile, eg evasion between 0 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), 1000, dtype=jnp.float32), # TODO fix to tile, eg evasion between 0 and 1
    },
    "pure_defence": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 6), dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (block, resist, immunity, evasion, damage_return, damage_return_rate)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), -1000, dtype=jnp.float32), # TODO fix to tile, eg evasion between 0 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), 1000, dtype=jnp.float32), # TODO fix to tile, eg evasion between 0 and 1
    },
    "strength": {
      "type": jnp.ndarray,
      "default": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 10, dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (base, current, soft_cap, max)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 1000, dtype=jnp.float32),
    },
    "agility": {
      "type": jnp.ndarray,
      "default": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 10, dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (base, current, soft_cap, max)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 1000, dtype=jnp.float32),
    },
    "intelligence": {
      "type": jnp.ndarray,
      "default": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 10, dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (base, current, soft_cap, max)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 1000, dtype=jnp.float32),
    },
    "resolve": {
      "type": jnp.ndarray,
      "default": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 10, dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (base, current, soft_cap, max)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 1000, dtype=jnp.float32),
    },
    "damage_amplification": {
      "type": jnp.ndarray,
      "default": jnp.ones((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (universal, physical, magical, pure)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), -10, dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 4), 10, dtype=jnp.float32),
    },
    "melee_attack": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([20., 2.1, 0, 1.5], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (base_damage, attack_range, critical_chance, critical_damage_modifier)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), -1000, dtype=jnp.float32), # TODO fix to tile, eg critical_chance between 0 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), 1000, dtype=jnp.float32), # TODO fix to tile, eg critical_chance between 0 and 1
    },
    "ranged_attack": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([15., 5, 0, 1.5], dtype=jnp.float32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ), # (HEROES_PER_TEAM * 2) by (base_damage, attack_range, critical_chance, critical_damage_modifier)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), -1000, dtype=jnp.float32), # TODO fix to tile, eg critical_chance between 0 and 1
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 6), 1000, dtype=jnp.float32), # TODO fix to tile, eg critical_chance between 0 and 1
    },
    "lifesteal": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 3), dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (physical, magical, pure)
      "obs": True,
      "low": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 3), -10, dtype=jnp.float32),
      "high": jnp.full((env_config['HEROES_PER_TEAM'] * 2, 3), 10, dtype=jnp.float32),
    },
    "silenced": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "broken": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "stunned": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "feared": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "taunted": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "invisible": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "sleepingg": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "ethereal": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "untargetable": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "hidden": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "phased": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "blind": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "disarmed": {
      "type": jnp.ndarray,
      "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by (flag, duration, dispelable, needs_greater_dispel)
      "obs": True,
      "low": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 4), dtype=jnp.int32),
      "high": jnp.tile(
          jnp.array([1, 100, 1, 1], dtype=jnp.int32), 
          (env_config['HEROES_PER_TEAM'] * 2, 1)
          ),
    },
    "abilities": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([-1, 0, 0, 0, 0, 0], dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (ABILITIES_PER_HERO) by (ability_index, base_cooldown, current_cooldown, parameter_1, parameter_2, parameter_3)
          (env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
      "obs": True,
      "low": jnp.tile(
          jnp.array([-1, 0, 0, -1000, -1000, -1000], dtype=jnp.float32),
          (env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
      "high": jnp.tile(
          jnp.array([100, 100, 100, 1000, 1000, 1000], dtype=jnp.float32), # TODO ability_index up to number of abilities
          (env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
    },
    "ability_status": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([-1, -1, 0, 1000, 1000], dtype=jnp.float32), # (HEROES_PER_TEAM * 2) by (HEROES_PER_TEAM * 2) by (ABILITIES_PER_HERO) by (ability_index, source_player_idx, duration, parameter_1, parameter_2)
          (env_config['HEROES_PER_TEAM'] * 2, env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
      "obs": True,
      "low": jnp.tile(
          jnp.array([-1, -1, 0, -1000, -1000, -1000], dtype=jnp.float32),
          (env_config['HEROES_PER_TEAM'] * 2, env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
      "high": jnp.tile(
          jnp.array([100, 100, 100, 1000, 1000], dtype=jnp.float32), # TODO up to number of abilities, number of heroes
          (env_config['HEROES_PER_TEAM'] * 2, env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
    },
    "available_actions": {
        "type": jnp.ndarray,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 12), dtype=jnp.int32), # (HEROES_PER_TEAM * 2) by 12 actions # TODO: make this dynamic instead of 12 actions
        "obs": False
    },
    "suicide_ability_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "steal_strength_ability_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "multi_attack_ability_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "return_ability_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "strength_regen_ability_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "add_barrier_ability_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "base_melee_attack_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "base_ranged_attack_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
    "end_turn_count": {
        "type": int,
        "default": jnp.zeros((env_config['HEROES_PER_TEAM'] * 2, 1), dtype=jnp.int32),
        "obs": False
    },
  },
  # TODO: "MapState": {
  # },
  "GameState": {
    "units": {
        "type": "UnitsState",
        "obs": True
        },
    "distance_to_enemy": {
        "type": float,
        "default": jnp.float32(0),
        "obs": True,
        "low": 0,
        "high": 30
        },
    "steps": {
        "type": int,
        "default": jnp.int32(0),
        "obs": True,
        "low": 0,
        "high": 1000
        },
    "turn_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": True,
        "low": 0,
        "high": 100
        },
    "previous_closest_distance": {
        "type": float,
        "default": jnp.float32(0),
        "obs": False
        },
    "initial_distance": {
        "type": float,
        "default": jnp.float32(0),
        "obs": False
        },
    "cur_player_idx": {
        "type": jnp.ndarray,
        "default": jnp.zeros(2).at[0].set(1),
        "obs": False
        },
    "terminal": {
        "type": bool,
        "default": False,
        "obs": False
        },
    "ability_pool": {
      "type": jnp.ndarray,
      "default": jnp.tile(
          jnp.array([-1, 0, 0, 0, 0, 0], dtype=jnp.float32), # (ABILITY_POOL_SIZE) by (ability_index, base_cooldown, current_cooldown, parameter_1, parameter_2, parameter_3)
          (env_config['ABILITY_POOL_SIZE'], 1)
          ),
      "obs": True,
      "low": jnp.tile(
          jnp.array([-1, 0, 0, -1000, -1000, -1000], dtype=jnp.float32),
          (env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
      "high": jnp.tile(
          jnp.array([100, 100, 100, 1000, 1000, 1000], dtype=jnp.float32), # TODO ability_index up to number of abilities
          (env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
          ),
    },
    "pick_mode": {
        "type": int,
        "default": jnp.int32(1),
        "obs": True
        },
    "ability_pool_picked": {
        "type": jnp.ndarray,
        "default": jnp.zeros((env_config['ABILITY_POOL_SIZE']), dtype=jnp.int32),
        "obs": True,
        "low": jnp.zeros((env_config['ABILITY_POOL_SIZE']), dtype=jnp.int32),
        "high": jnp.ones((env_config['ABILITY_POOL_SIZE']), dtype=jnp.int32),
        },
    "pick_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": True,
        "low": 0,
        "high": 100,
        },
  },
}


# Define the function that creates UnitsState, TeamState, MapState, and GameState dataclasses from the schema
def create_struct_dataclass(schema):
    classes = {}
    for key, value in schema.items():
        annotations = {}
        for field, attrs in value.items():
            dtype = attrs["type"]#eval(attrs["type"]) if attrs["type"] in dir(jnp) else attrs["type"]
            annotations[field] = dtype
        classes[key] = struct.dataclass(type(key, (object,), {'__annotations__': annotations}))
    return classes

# Create classes from schema first
schema_classes = create_struct_dataclass(schema)

schema_classes = create_struct_dataclass(schema)
UnitsState = schema_classes['UnitsState']
GameState = schema_classes['GameState']

# # Define the function that initialises game state
# def initialise_game_state(UnitsState, GameState):
#     # Construct the default UnitsState by evaluating the provided defaults
#     default_unit_state = UnitsState(**{
#         k: eval(v['default']) if isinstance(v['default'], str) and 'jnp' in v['default'] else v['default']
#         for k, v in schema['UnitsState'].items()
#     })
    
#     # Create player and enemy unit states
#     player = default_unit_state
#     enemy = default_unit_state.replace(unit_id=-1)  # Assuming 'unit_id' differentiates players and enemies
    
#     # Construct the GameState by providing defaults for non-unit state properties
#     game_state = GameState(
#         player=player,
#         enemy=enemy,
#         **{k: eval(v['default']) if isinstance(v['default'], str) and 'jnp' in v['default'] else v['default']
#            for k, v in schema['GameState'].items() if k not in ['player', 'enemy']}
#     )
#     return game_state

# Define the function that gets the observation bounds
def get_observation_bounds(state_schema):
    def get_bounds_from_schema(schema_part):
        # Handle nested UnitsState fields
        nested_bounds = [(schema[v["type"]], v["obs"]) 
                        for k, v in schema_part.items() 
                        if isinstance(v.get("type"), str) and v["type"] in schema and v.get("obs")]
        
        # Get bounds from nested schemas
        nested_values = [bound for schema_part, _ in nested_bounds 
                        for bound in get_bounds_from_schema(schema_part)]
        
        # Get bounds from direct fields
        direct_bounds = [(v["low"], v["high"]) 
                        for k, v in schema_part.items()
                        if v.get("obs") and "low" in v and "high" in v]
        
        return nested_values + direct_bounds

    # Start from GameState schema and unzip results
    bounds = get_bounds_from_schema(state_schema["GameState"])
    low, high = zip(*bounds) if bounds else ([], [])
    
    return list(low), list(high)

# Define the function that creates a UnitsState instance
def create_unit_state(UnitsState, custom_values=None):
    # Get defaults from schema
    defaults = {}
    
    for field, attrs in schema['UnitsState'].items():
        if isinstance(attrs['type'], str) and attrs['type'] in schema:
            # Handle nested types like AbilityState and AbilityStatusState
            nested_class = schema_classes[attrs['type']]
            nested_defaults = {
                k: (eval(v['default']) if isinstance(v['default'], str) and 'jnp' in v['default'] 
                    else v['default'])
                for k, v in schema[attrs['type']].items()
            }
            defaults[field] = nested_class(**nested_defaults)
        else:
            defaults[field] = (eval(attrs['default']) if isinstance(attrs['default'], str) and 'jnp' in attrs['default'] 
                             else attrs['default'])
    
    # Update with custom values if provided
    if custom_values:
        # Remove the item() conversion and just use values directly
        defaults.update(custom_values)
        
    # Create UnitsState instance
    return UnitsState(**defaults)

# Define the function that takes in the schema and a GameState object and returns the observation array # TODO: this flattens arrays, might need to unflatten for CNN etc later
def get_observation_values(state_obj: Any, schema: Dict, parent_key: str = '') -> List[float]:
    """Get observation values from state object according to schema.
    
    Args:
        state_obj: State object containing game data
        schema: Schema dictionary defining state structure
        parent_key: Key path for nested objects
        
    Returns:
        List of flattened observation values
    """
    schema_section = schema.get(state_obj.__class__.__name__, schema)
    
    nested_values = [
        val for field_name, field_def in schema_section.items()
        if field_def.get('obs', False) and isinstance(field_def.get('type'), str) and field_def['type'] in schema
        for val in get_observation_values(getattr(state_obj, field_name), schema, f"{parent_key}{field_name}.")
    ]
    
    direct_values = []
    for field_name, field_def in schema_section.items():
        if field_def.get('obs', False) and not (isinstance(field_def.get('type'), str) and field_def['type'] in schema):
            field_value = getattr(state_obj, field_name)
            # Handle array values by flattening them
            if isinstance(field_value, (jnp.ndarray, chex.Array)):
                direct_values.extend(field_value.flatten())
            else:
                direct_values.append(jnp.float32(field_value) if isinstance(field_value, bool) else field_value)
    
    return nested_values + direct_values

def create_get_obs_functions(schema: Dict) -> Tuple[Callable, Callable]:
    """Creates functions to get observations for player and enemy perspectives.
    
    Args:
        schema: Dictionary defining state structure
        
    Returns:
        (get_player_obs, get_enemy_obs): Functions that return observations
    """
    
    def get_player_obs(state: GameState) -> chex.Array:
        return jnp.array(get_observation_values(state, schema))
    
    def get_enemy_obs(state: GameState) -> chex.Array:
        # Swap team order in UnitsState arrays
        swapped_units = state.units.replace(
            **{field: _swap_team_order(getattr(state.units, field)) 
               for field in state.units.__dict__.keys()}
        )
        swapped_state = state.replace(units=swapped_units)
        return jnp.array(get_observation_values(swapped_state, schema))
    
    return get_player_obs, get_enemy_obs

def _swap_team_order(array: chex.Array) -> chex.Array:
    """Swaps player and enemy sections of array along first dimension.
    
    Args:
        array: Input array with shape (units, ...)
        
    Returns:
        Array with team order swapped
    """
    heroes_per_team = env_config['HEROES_PER_TEAM']
    
    # Handle 1D arrays or higher dimensions uniformly
    if len(array.shape) <= 1:
        return array  # Skip swapping for non-unit arrays
        
    # For all other arrays, swap team sections
    return jnp.concatenate([
        array[heroes_per_team:],
        array[:heroes_per_team]
    ])

def make_schema_based_observation(state, aidx, schema):
    get_player_obs, get_enemy_obs = create_get_obs_functions(schema)
    
    return jax.lax.cond(
        aidx == 0,
        lambda _: get_player_obs(state),
        lambda _: get_enemy_obs(state),
        operand=None
    )

def get_observation_size(schema: Dict) -> int:
    def count_observable_fields(schema_part):
        # Count array fields by their size
        array_counts = []
        for k, v in schema_part.items():
            if v.get('obs', False) and not (isinstance(v.get('type'), str) and v['type'] in schema):
                if isinstance(v.get('default'), (jnp.ndarray, chex.Array)):
                    try:
                        array_counts.append(v['default'].size)  # Use size instead of len
                    except:
                        array_counts.append(1)  # Fallback for scalar arrays
                else:
                    array_counts.append(1)
        
        # Count nested fields
        nested_counts = [
            count_observable_fields(schema[v['type']])
            for k, v in schema_part.items()
            if v.get('obs', False) and isinstance(v.get('type'), str) and v['type'] in schema
        ]
        
        return sum(array_counts) + sum(nested_counts)
    
    return count_observable_fields(schema['GameState'])

# def get_low_high(schema, class_name):
#     low = jnp.array([info["low"] for field, info in schema[class_name].items() if info.get("obs", False)])
#     high = jnp.array([info["high"] for field, info in schema[class_name].items() if info.get("obs", False)])
#     return low, high

# @struct.dataclass
# class TeamState:

# @struct.dataclass
# class UnitState:
#     unit_id: int
#     health_current: float
#     health_max: float
#     location_x: int
#     location_y: int
#     melee_base_attack_damage: float
#     ranged_base_attack_damage: float
#     melee_attack_range: float
#     ranged_attack_range: float
#     movement_points_current: float
#     movement_points_max: float
#     action_points_current: float
#     action_points_max: float
#     available_actions: chex.Array

# @struct.dataclass
# class EnvState:
#     # flatten this?
#     # board: chex.Array
#     player: UnitState
#     enemy: UnitState
#     distance_to_enemy: float
#     steps: int
#     turn_count: int
#     previous_closest_distance: float
#     initial_distance: float
#     # enemy_network_params: XX

# @struct.dataclass
# class EnvParams:
#     # Configuration for the environment
#     BOARD_SIZE: int = env_config['BOARD_SIZE']
#     MAX_HEALTH: float = env_config['MAX_HEALTH']
#     MELEE_DAMAGE: float = env_config['MELEE_DAMAGE']
#     RANGED_DAMAGE: float = env_config['RANGED_DAMAGE']
#     MELEE_RANGE: float = env_config['MELEE_RANGE']
#     RANGED_RANGE: float = env_config['RANGED_RANGE']
#     MOVEMENT_POINTS: float = env_config['MOVEMENT_POINTS']
#     ACTION_POINTS: float = env_config['ACTION_POINTS']
#     MAX_STEPS: int = env_config['MAX_STEPS']