from long_nardy import LongNardy

def play():
    game = LongNardy()
    while not game.is_finished():
        game.pretty_print_board()
        moves = game.get_valid_moves()
        if moves:
            print("\nAvailable moves:")
            for idx, move in enumerate(moves):
                print(f"{idx}: From board index {move[0] + 1}, move {move[1]} pips, dice info: {move[2]}")
            try:
                choice = int(input("Enter the index of the move to perform: "))
                if choice < 0 or choice >= len(moves):
                    print("Invalid move index. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            action = moves[choice]
            state, reward, done = game.step(action)
            print(f"Move executed. Reward: {reward}\n")
        else:
            print("No valid moves available. Turn passes automatically.")
            input("Press Enter to continue...")
            # Force turn change if no valid moves.
            game.current_player = 'black' if game.current_player == 'white' else 'white'
            game.roll_dice()

    game.pretty_print_board()
    if game.white_off == 15:
        print("Game over! White wins!")
    elif game.black_off == 15:
        print("Game over! Black wins!")

def example_one_action_play():
    game = LongNardy()
    game.print()
    
    valid_moves = game.get_valid_moves()
    print("Valid moves:", valid_moves)
    
    if valid_moves:
        action = valid_moves[0]
        print("Taking action:", action)
        state, reward, done = game.step(action)
        game.print()
        print("State:", state)
        print("Reward:", reward, "Done:", done)

if __name__ == "__main__":
    # example_one_action_play()
    play()