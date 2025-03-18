import cProfile
import io
import pstats
import random
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
            game.state.roll_dice()
            game.state.change_turn()
            

    # game.state.pretty_print()
    if game.state.white_off == 15:
        # print("Game over! White wins!")
        return True
    elif game.state.black_off == 15:
        # print("Game over! Black wins!")
        return False

if __name__ == "__main__":
    random.seed(0)
    white_wins = 0
    black_wins = 0
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = perf_counter()
    for _ in range(1000):
        if play():
            white_wins += 1
        else:
            black_wins += 1
    profiler.disable()
    
    end_time = perf_counter()
    print("Time taken:", end_time - start_time)
    print("White wins:", white_wins)
    print("Black wins:", black_wins)


    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(10)  # Show top 10 slowest functions
    print(stream.getvalue())