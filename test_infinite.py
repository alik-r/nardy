from long_nardy import LongNardy

while True:
    game = LongNardy()
    i = 0
    while not game.is_finished():
        states = game.get_states_after_dice()
        game.step(states[0])
        i += 1
        if i > 1000:
            print("Infinite loop")
            game.state.pretty_print()
            exit(0)
