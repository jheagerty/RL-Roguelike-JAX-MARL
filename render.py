import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from prettytable import PrettyTable
from imojify import imojify

def offset_image(coords, emoji, ax, zoom=0.04):
    img = plt.imread(imojify.get_img_path(emoji))
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    ab = AnnotationBbox(im, coords, frameon=False, pad=0)
    ax.add_artist(ab)

def render_game_state(state):
    # Plotting the game board
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # Setting ticks at positions 0.5, 1.5, ..., 19.5 for labels
    tick_positions = [i + 0.5 for i in range(20)]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    # Setting tick labels
    ax.set_xticklabels([str(i) for i in range(20)])
    ax.set_yticklabels([str(i) for i in range(20)])

    # Removing the visible ticks
    ax.tick_params(axis='both', which='both', length=0)

    # Adding gridlines at positions 1, 2, ..., 19
    ax.set_xticks(range(1, 20), minor=True)
    ax.set_yticks(range(1, 20), minor=True)
    ax.grid(True, which='minor', linestyle='-', linewidth=1)

    # Placing emojis
    player_coords = (state.player.location_x + 0.5, state.player.location_y + 0.5)  # Adjusted for offset
    enemy_coords = (state.enemy.location_x + 0.5, state.enemy.location_y + 0.5)  # Adjusted for offset
    offset_image(player_coords, '🤗', ax)  # Hugging face emoji for player
    offset_image(enemy_coords, '😈', ax)  # Devil emoji for enemy

    plt.show()

    # Printing tables
    # Function to fill table rows for a unit
    def fill_table_rows(entity, max_rows):
        rows = []
        for attr, value in entity.__dict__.items():
            if attr != 'available_actions':
                rows.append([attr, value])

        # Filling in empty rows if necessary
        while len(rows) < max_rows:
            rows.append(["", ""])
        return rows

    # Counting max number of rows needed for the table
    max_rows = max(len(state.player.__dict__) - 1,  # -1 to exclude 'available_actions'
                   len(state.enemy.__dict__) - 1)

    # Filling rows for each unit/entity
    player_rows = fill_table_rows(state.player, max_rows)
    enemy_rows = fill_table_rows(state.enemy, max_rows)

    # Creating a combined table
    combined_table = PrettyTable()
    # combined_table.title = "Game State Overview"
    combined_table.field_names = ["Player Attribute", "Player Value", "Enemy Attribute", "Enemy Value"]

    for i in range(max_rows):
        combined_row = player_rows[i] + enemy_rows[i]
        combined_table.add_row(combined_row)

    print(combined_table)

    # Player Actions Table
    actions_table = PrettyTable()
    # actions_table.title = "Player Available Actions"
    actions_table.field_names = ["Action ID", "Action Name"]

    action_names = [
        "move left down",
        "move left",
        "move left up",
        "move down",
        "move up",
        "move right down",
        "move right",
        "move right up",
        "melee attack",
        "ranged attack",
        "end turn"
    ]

    for i, action in enumerate(state.player.available_actions):
        action_name = action_names[i] if i < len(action_names) else ""
        actions_table.add_row([action, action_name])

    print(actions_table)