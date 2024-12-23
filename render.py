import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from prettytable import PrettyTable
from imojify import imojify
from data_classes import GameState
from config import env_config

def render_game_state(state: GameState):
    """Renders the current game state in a human-readable format.
    
    Args:
        state: Current GameState instance
    """
    # Create board visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # Grid setup
    tick_positions = [i + 0.5 for i in range(20)]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(i) for i in range(20)])
    ax.set_yticklabels([str(i) for i in range(20)])
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticks(range(1, 20), minor=True)
    ax.set_yticks(range(1, 20), minor=True)
    ax.grid(True, which='minor', linestyle='-', linewidth=1)

    # Plot units
    for i, location in enumerate(state.units.location):
        coords = (location[0] + 0.5, location[1] + 0.5)
        emoji = '🤗' if i < env_config['HEROES_PER_TEAM'] else '😈'
        img = plt.imread(imojify.get_img_path(emoji))
        im = OffsetImage(img, zoom=0.04)
        im.image.axes = ax
        ab = AnnotationBbox(im, coords, frameon=False, pad=0)
        ax.add_artist(ab)

    plt.show()

    # Print unit status tables
    table = PrettyTable()
    table.field_names = ["Attribute", "Player", "Enemy"]

    # Add key stats
    player_idx = 0
    enemy_idx = env_config['HEROES_PER_TEAM']
    
    stats = [
        ("Health", state.units.health[player_idx][0], state.units.health[enemy_idx][0]),
        ("Action Points", state.units.action_points[player_idx][1], state.units.action_points[enemy_idx][1]),
        ("Movement Points", state.units.movement_points[player_idx][1], state.units.movement_points[enemy_idx][1]),
        ("Location", f"({state.units.location[player_idx][0]}, {state.units.location[player_idx][1]})", 
                    f"({state.units.location[enemy_idx][0]}, {state.units.location[enemy_idx][1]})"),
        ("Distance", state.distance_to_enemy, ""),
        ("Turn", state.turn_count, ""),
        ("Steps", state.steps, ""),
    ]

    for name, p_val, e_val in stats:
        table.add_row([name, p_val, e_val])

    print("\nGame State:")
    print(table)

    # Print ability status
    ability_table = PrettyTable()
    ability_table.field_names = ["Slot", "Player Ability", "Enemy Ability"]
    
    for slot in range(env_config['ABILITIES_PER_HERO']):
        p_ability = state.units.abilities[player_idx][slot]
        e_ability = state.units.abilities[enemy_idx][slot]
        ability_table.add_row([
            slot,
            f"ID: {int(p_ability[0])} (CD: {p_ability[2]}/{p_ability[1]})" if p_ability[0] >= 0 else "Empty",
            f"ID: {int(e_ability[0])} (CD: {e_ability[2]}/{e_ability[1]})" if e_ability[0] >= 0 else "Empty"
        ])

    print("\nAbilities:")
    print(ability_table)

    # Print pick mode status if relevant
    if state.pick_mode:
        print("\nPick Mode - Available Abilities:")
        pick_table = PrettyTable()
        pick_table.field_names = ["Index", "Ability ID", "Status"]
        
        for i, ability in enumerate(state.ability_pool):
            status = "Picked" if state.ability_pool_picked[i] else "Available"
            if ability[0] >= 0:
                pick_table.add_row([i, int(ability[0]), status])
        
        print(pick_table)