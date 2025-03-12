import cProfile
import io
import pstats
from long_nardy import LongNardy
from time import perf_counter

def play():
    game = LongNardy()
    while not game.is_finished():
        # game.state.pretty_print()
        moves = game.get_states_after_dice()
        if moves:
            game.step(moves[0])
        else:
            print("Error. No valid moves available.")
            raise Exception("No valid moves available.")
            

    # game.state.pretty_print()
    if game.state.white_off == 15:
        # print("Game over! White wins!")
        return True
    elif game.state.black_off == 15:
        # print("Game over! Black wins!")
        return False

# def example_one_action_play():
#     game = LongNardy()
#     game.print()
    
#     valid_moves = game.get_valid_moves()
#     print("Valid moves:", valid_moves)
    
#     if valid_moves:
#         action = valid_moves[0]
#         print("Taking action:", action)
#         state, reward, done = game.step(action)
#         game.print()
#         print("State:", state)
#         print("Reward:", reward, "Done:", done)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # example_one_action_play()
    white_wins = 0
    black_wins = 0
    start_time = perf_counter()
    for _ in range(1):
        if play():
            white_wins += 1
        else:
            black_wins += 1
    profiler.disable()
    
    end_time = perf_counter()
    print("Time taken:", end_time - start_time)
    print("White wins:", white_wins)
    print("Black wins:", black_wins)


    # Print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(10)  # Show top 10 slowest functions
    print(stream.getvalue())