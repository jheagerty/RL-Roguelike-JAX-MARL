# data_classes.py
from flax import struct
import chex
from config import env_config
import jax
import jax.numpy as jnp
from typing import Dict, List, Any
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
  "AbilityState": {
    "ability_index": {
      "type": int,
      "default": jnp.int32(-1),
      "obs": True,
      "low": -1,
      "high": 100
    },
    "base_cooldown": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 100
    },
    "current_cooldown": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 100
    },
    "parameter_1": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": -1000,
      "high": -1000
    },
    "parameter_2": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": -1000,
      "high": 1000
    },
    "parameter_3": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": -1000,
      "high": 1000
    }
  },
  "AbilityStatusState": {
    "ability_index": {
      "type": int,
      "default": jnp.int32(-1),
      "obs": True,
      "low": -1,
      "high": 100
    },
    "source_player_idx": {
      "type": chex.Array,
      "default": jnp.zeros(2),
      "obs": True, #TODO this might take a bit of work
      "low": jnp.zeros(2),
      "high": jnp.ones(2)
    },
    "duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 100
    },
    "parameter_1": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": -1000,
      "high": -1000
    },
    "parameter_2": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": -1000,
      "high": 1000
    }
  },
  "UnitState": {
    "unit_id": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "location_x": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 20,
    },
    "location_y": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 20,
    },
    "action_points_base": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 20,
    },
    "action_points_current": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 20,
    },
    "action_points_max": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 20,
    },
    "movement_points_base": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 50,
    },
    "movement_points_current": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 50,
    },
    "movement_points_max": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 50,
    },
    "movement_points_percentage": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "movement_points_multiplier": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": 0,
      "high": 10,
    },
    "health_current": {
      "type": float,
      "default": jnp.float32(100),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "health_max": {
      "type": float,
      "default": jnp.float32(100),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "health_percentage": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "health_regeneration": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "health_regeneration_rate": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "mana_current": {
      "type": float,
      "default": jnp.float32(100),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "mana_max": {
      "type": float,
      "default": jnp.float32(100),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "mana_percentage": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "mana_regeneration": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "mana_regeneration_rate": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "barrier_current": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "barrier_status_reduction": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": 0,
      "high": 100,
    },
    "barrier_max": {
      "type": float,
      "default": jnp.float32(100),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "barrier_percentage": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "barrier_regeneration": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "barrier_regeneration_rate": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "physical_block": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "magical_block": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "physical_resist": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "magical_resist": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "physical_immunity": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "magical_immunity": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "physical_evasion": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "magical_evasion": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "physical_damage_return": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "physical_damage_return_rate": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "magical_damage_return": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "magical_damage_return_rate": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "pure_damage_return": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "pure_damage_return_rate": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -1,
      "high": 1,
    },
    "base_strength": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "strength_current": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "base_agility": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "agility_current": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "base_intelligence": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "intelligence_current": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "base_resolve": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "resolve_current": {
      "type": float,
      "default": jnp.float32(10),
      "obs": True,
      "low": 0,
      "high": 1000,
    },
    "attack_damage_amplification": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": -1,
      "high": 10,
    },
    "melee_base_attack_damage": {
      "type": float,
      "default": jnp.float32(25),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "melee_attack_range": {
      "type": float,
      "default": jnp.float32(2.6),
      "obs": True,
      "low": 0,
      "high": 10,
    },
    "melee_crit_chance": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "melee_crit_modifier": {
      "type": float,
      "default": jnp.float32(1.5),
      "obs": True,
      "low": 1,
      "high": 10,
    },
    "ranged_base_attack_damage": {
      "type": float,
      "default": jnp.float32(15),
      "obs": True,
      "low": -1000,
      "high": 1000,
    },
    "ranged_attack_range": {
      "type": float,
      "default": jnp.float32(5),
      "obs": True,
      "low": 0,
      "high": 20,
    },
    "ranged_crit_chance": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": 0,
      "high": 1,
    },
    "ranged_crit_modifier": {
      "type": float,
      "default": jnp.float32(1.5),
      "obs": True,
      "low": 1,
      "high": 3,
    },
    "damage_amplification": {
      "type": float,
      "default": jnp.float32(1),
      "obs": True,
      "low": 0,
      "high": 10,
    },
    "physical_lifesteal": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -10,
      "high": 10,
    },
    "magical_lifesteal": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -10,
      "high": 10,
    },
    "pure_lifesteal": {
      "type": float,
      "default": jnp.float32(0),
      "obs": True,
      "low": -10,
      "high": 10,
    },
    "silenced_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "silenced_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "silenced_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "silenced_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "silenced_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "broken_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "broken_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "broken_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "broken_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "broken_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "stunned_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "stunned_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "stunned_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "stunned_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "stunned_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "feared_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "feared_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "feared_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "feared_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "feared_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "taunted_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "taunted_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "taunted_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "taunted_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "taunted_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "invisible_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "invisible_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "invisible_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "invisible_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "invisible_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "sleeping_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "sleeping_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "sleeping_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "sleeping_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "sleeping_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "ethereal_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "ethereal_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "ethereal_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "ethereal_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "ethereal_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "untargetable_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "untargetable_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "untargetable_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "untargetable_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "untargetable_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "hidden_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "hidden_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "hidden_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "hidden_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "hidden_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "phased_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "phased_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "phased_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "phased_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "phased_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "blind_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "blind_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "blind_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "blind_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "blind_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "disarmed_flag": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "disarmed_duration": {
      "type": int,
      "default": jnp.int32(0),
      "obs": True,
      "low": 0,
      "high": 30,
    },
    "disarmed_permanent": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "disarmed_dispelable": {
      "type": bool,
      "default": True,
      "obs": True,
      "low": False,
      "high": True,
    },
    "disarmed_needs_greater_dispel": {
      "type": bool,
      "default": False,
      "obs": True,
      "low": False,
      "high": True,
    },
    "ability_state_1": {
      "type": "AbilityState",
      "obs": True
    },
    "ability_status_state_1": {
      "type": "AbilityStatusState",
      "obs": True
    },
    "ability_status_state_2": {
      "type": "AbilityStatusState",
      "obs": True
    },
    "available_actions": {
        "type": chex.Array,
        "default": jnp.zeros(12), #TODO: make this dynamic
        "obs": False
    },
    "suicide_ability_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "steal_strength_ability_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "multi_attack_ability_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "return_ability_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "strength_regen_ability_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "add_barrier_ability_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "base_melee_attack_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "base_ranged_attack_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
    "end_turn_count": {
        "type": int,
        "default": jnp.int32(0),
        "obs": False
    },
  },
  # TODO: "TeamState": {
  # },
  # TODO: "MapState": {
  # },
  "GameState": {
    "player": {
        "type": "UnitState",
        "obs": True
        },
    "enemy": {
        "type": "UnitState",
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
        "type": chex.Array,
        "default": jnp.zeros(2).at[0].set(1),
        "obs": False
        },
    "terminal": {
        "type": bool,
        "default": False,
        "obs": False
        },
  },
}

# Define the function that creates UnitState, TeamState, MapState, and GameState dataclasses from the schema
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

# # Define the function that initialises game state
# def initialise_game_state(UnitState, GameState):
#     # Construct the default UnitState by evaluating the provided defaults
#     default_unit_state = UnitState(**{
#         k: eval(v['default']) if isinstance(v['default'], str) and 'jnp' in v['default'] else v['default']
#         for k, v in schema['UnitState'].items()
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
        # Handle nested UnitState fields
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

# Define the function that creates a UnitState instance
def create_unit_state(UnitState, custom_values=None):
    # Get defaults from schema
    defaults = {}
    
    for field, attrs in schema['UnitState'].items():
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
        
    # Create UnitState instance
    return UnitState(**defaults)

# Define the function that takes in the schema and a GameState object and returns the observation array # TODO: this flattens arrays, might need to unflatten for CNN etc later
def get_observation_values(state_obj: Any, schema: Dict, parent_key: str = '') -> List[float]:
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

def create_get_obs_functions(schema: Dict):
    def get_player_obs(state):
        return jnp.array(get_observation_values(state, schema))
    
    def get_enemy_obs(state):
        swapped_state = state.replace(
            player=state.enemy,
            enemy=state.player
        )
        return jnp.array(get_observation_values(swapped_state, schema))
    
    return get_player_obs, get_enemy_obs

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