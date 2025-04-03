import random
import time
import csv
from pathlib import Path
from typing import List
from multiprocessing import Manager, cpu_count, Lock
import concurrent.futures
import numpy as np
import torch
from torch import nn
from long_nardy import LongNardy
from state import State

device = torch.device("cpu")

# Original ANN and Agent implementations preserved
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(98, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class Agent(nn.Module):
    def __init__(self, lr=0.1, epsilon=0.1):
        super().__init__()
        self.net = ANN().to(device)
        self.epsilon = epsilon
        self.lr = lr
        self.eligibility_traces = {name: torch.zeros_like(param) 
                                   for name, param in self.net.named_parameters()}
        
    def evaluate(self, state: State) -> float:
        state_tensor = torch.tensor(state.get_representation_for_current_player(),
                                  dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.net(state_tensor).item()

class SharedState:
    def __init__(self, manager):
        self.rating = manager.Value('i', 1500)
        self.uncertainty = manager.Value('i', 150)
        self.games_played = manager.Value('i', 0)
        self.lock = manager.Lock()

class Player:
    def __init__(self, name, model_path, manager):
        self.name = name
        self.agent = Agent().to(device)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        self.agent.load_state_dict(state_dict)
        self.agent.eval()
        self.shared = SharedState(manager)

    @property
    def rating(self):
        with self.shared.lock:
            return self.shared.rating.value

    @property
    def uncertainty(self):
        with self.shared.lock:
            return self.shared.uncertainty.value

    @property
    def games_played(self):
        with self.shared.lock:
            return self.shared.games_played.value

    def __str__(self):
        return f"{self.name}: {self.rating}±{self.uncertainty}"

def play_game(white: Player, black: Player) -> int:
    game = LongNardy()
    current_players = [white, black]
    
    while not game.is_finished():
        player = current_players[0] if game.state.is_white else current_players[1]
        candidate_states = game.get_states_after_dice()
        
        if not candidate_states:
            game.apply_dice(game.state)
            continue
            
        values = [player.agent.evaluate(state) for state in candidate_states]
        game.step(candidate_states[np.argmax(values)])
    
    return 1 if game.state.white_off == 15 else 0

def update_ratings(winner: Player, loser: Player):
    first, second = sorted([winner, loser], key=lambda p: p.name)
    with first.shared.lock, second.shared.lock:
        combined_uncertainty = (winner.uncertainty + loser.uncertainty) / 200
        rating_diff = (loser.rating - winner.rating) / max(1, combined_uncertainty)
        
        expected = 1 / (1 + 10 ** (rating_diff / 400))
        actual_k = 32 * min(winner.uncertainty, loser.uncertainty) / 100
        delta = int(actual_k * (1 - expected))
        
        winner.shared.rating.value += delta
        loser.shared.rating.value -= delta
        
        decay_rate = 0.98 if min(winner.games_played, loser.games_played) < 50 else 0.995
        for p in [winner, loser]:
            new_uncertainty = int(p.uncertainty * decay_rate)
            p.shared.uncertainty.value = max(30, new_uncertainty)
            p.shared.games_played.value += 1

def match_game(pair):
    white, black = pair
    result = play_game(white, black)
    
    if result == 1:
        update_ratings(white, black)
    else:
        update_ratings(black, white)
    return result

def schedule_matches(players, num_matches):
    buckets = {}
    for p in players:
        bucket = p.rating // 25
        buckets.setdefault(bucket, []).append(p)
    
    matches = []
    valid_buckets = {k: v for k, v in buckets.items() if len(v) >= 1}
    bucket_keys = sorted(valid_buckets.keys())
    
    for _ in range(num_matches):
        if not bucket_keys:
            break
        
        pair_options = []
        for k in bucket_keys:
            if len(valid_buckets[k]) >= 2:
                pair_options.append((k, k))
            if k+25 in bucket_keys:
                pair_options.append((k, k+25))
        
        if not pair_options:
            break
        
        b1, b2 = random.choice(pair_options)
        
        try:
            candidates = valid_buckets[b1]
            p1 = random.choice(candidates)
            
            if b1 == b2:
                candidates = [p for p in candidates if p != p1]
                if not candidates:
                    continue
                p2 = random.choice(candidates)
            else:
                p2 = random.choice(valid_buckets[b2])
                
            matches.append((p1, p2))
        except (KeyError, IndexError):
            continue
    
    return matches

def save_results(players: List[Player], filename="ratings.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Name", "Rating", "Uncertainty", "Games"])
        sorted_players = sorted(players, key=lambda p: -p.rating)
        for i, p in enumerate(sorted_players, 1):
            writer.writerow([i, p.name, p.rating, p.uncertainty, p.games_played])

def main():
    with Manager() as manager:
        model_dir = Path(__file__).parent / "v2"
        players = [
            Player(f.stem, str(f), manager)
            for f in model_dir.glob("*.pth")
        ]
        
        num_matches = 300000
        start_time = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_count(),
            initializer=lambda: [p.agent.share_memory() for p in players]
        ) as executor:
            batch_size = 1000
            for _ in range(num_matches // batch_size):
                matches = schedule_matches(players, batch_size)
                executor.map(match_game, matches)
                print(f"Completed {batch_size} matches in {time.time()-start_time:.2f}s")
            
        print(f"Completed {num_matches} matches in {time.time()-start_time:.2f}s")
        save_results(players)

if __name__ == '__main__':
    main()