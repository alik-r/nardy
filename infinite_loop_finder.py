from long_nardy import LongNardy

while True:
    game = LongNardy()

    moves = 0
    while not game.is_finished():
        move = game.get_states_after_dice()[0]
        game.step(move)
        moves += 1
        if moves > 1000:
            print("Infinite loop detected!")
            game.state.pretty_print()
            break

