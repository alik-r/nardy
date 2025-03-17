import threading
import concurrent.futures
from long_nardy import LongNardy

def play_game():
    while True:
        game = LongNardy()
        moves = 0
        while not game.is_finished():
            move = game.get_states_after_dice()[0]
            game.step(move)
            moves += 1
            if moves > 1000:
                print(f"Thread {threading.current_thread().name}: Infinite loop detected!")
                game.state.pretty_print()
                break

# Get the maximum number of threads supported by the system
max_threads = min(32, (threading.active_count() or 1) * 2)

# Run as many threads as possible
with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    futures = [executor.submit(play_game) for _ in range(max_threads)]
    
    # Keep the main thread alive
    for future in concurrent.futures.as_completed(futures):
        future.result()  # This will block and prevent main thread from exiting
