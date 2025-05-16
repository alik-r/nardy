from long_nardy import LongNardy

game = LongNardy()
print(game.state.dice_remaining)
states = game.get_states_after_dice()
for state in states:
    state.pretty_print()
