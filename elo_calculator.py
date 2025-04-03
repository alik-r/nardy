import random
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import cpu_count, Manager
import concurrent.futures
import numpy as np
import torch
from torch import nn
from long_nardy import LongNardy
from state import State

device = torch.device("cpu")

# Global cache for agents in worker processes
_worker_agents = {}

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
    def __init__(self, model_path):
        super().__init__()
        self.net = ANN().to(device)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def evaluate(self, state: State) -> float:
        state_tensor = torch.tensor(state.get_representation_for_current_player(),
                                  dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.net(state_tensor).item()

class RatingManager:
    def __init__(self, model_dir):
        self.models = {f.stem: f for f in model_dir.glob("*.pth")}
        self.manager = Manager()
        self.ratings = self.manager.dict({
            name: self.manager.dict({
                'rating': 1500,
                'uncertainty': 150,
                'games_played': 0,
                'lock': self.manager.Lock()
            })
            for name in self.models.keys()
        })

    def get_agent(self, name):
        """Get or load agent in current process"""
        if name not in _worker_agents:
            _worker_agents[name] = Agent(self.models[name])
        return _worker_agents[name]

def play_game(white_name: str, black_name: str, rating_manager: RatingManager) -> int:
    white_agent = rating_manager.get_agent(white_name)
    black_agent = rating_manager.get_agent(black_name)
    
    game = LongNardy()
    current_agents = [white_agent, black_agent]
    
    while not game.is_finished():
        agent = current_agents[0] if game.state.is_white else current_agents[1]
        candidate_states = game.get_states_after_dice()
        
        if not candidate_states:
            game.apply_dice(game.state)
            continue
            
        values = [agent.evaluate(state) for state in candidate_states]
        game.step(candidate_states[np.argmax(values)])
    
    return 1 if game.state.white_off == 15 else 0

def update_ratings(winner: str, loser: str, rating_manager: RatingManager):
    # Get both players' locks in alphabetical order to prevent deadlocks
    sorted_names = sorted([winner, loser])
    with rating_manager.ratings[sorted_names[0]]['lock'], \
         rating_manager.ratings[sorted_names[1]]['lock']:

        wr = rating_manager.ratings[winner]
        lr = rating_manager.ratings[loser]

        combined_uncertainty = (wr['uncertainty'] + lr['uncertainty']) / 200
        rating_diff = (lr['rating'] - wr['rating']) / max(1, combined_uncertainty)
        
        expected = 1 / (1 + 10 ** (rating_diff / 400))
        actual_k = 32 * min(wr['uncertainty'], lr['uncertainty']) / 100
        delta = int(actual_k * (1 - expected))
        
        wr['rating'] += delta
        lr['rating'] -= delta
        
        decay_rate = 0.98 if min(wr['games_played'], lr['games_played']) < 50 else 0.995
        for p in [wr, lr]:
            new_uncertainty = int(p['uncertainty'] * decay_rate)
            p['uncertainty'] = max(30, new_uncertainty)
            p['games_played'] += 1

def match_game(pair: Tuple[str, str], rating_manager: RatingManager) -> int:
    white, black = pair
    result = play_game(white, black, rating_manager)
    
    if result == 1:
        update_ratings(white, black, rating_manager)
    else:
        update_ratings(black, white, rating_manager)
    return result

def schedule_matches(rating_manager: RatingManager, num_matches: int) -> List[Tuple[str, str]]:
    """Generate matches using rating buckets with empty bucket protection"""
    buckets = {}
    # Create buckets with atomic rating reads
    for name in rating_manager.ratings:
        player_data = rating_manager.ratings[name]
        with player_data['lock']:
            bucket = player_data['rating'] // 25
            buckets.setdefault(bucket, []).append(name)
    
    matches = []
    valid_buckets = {k: v for k, v in buckets.items() if len(v) >= 1}
    bucket_keys = sorted(valid_buckets.keys())
    
    for _ in range(num_matches):
        if not bucket_keys:
            break  # No valid players
        
        # Generate possible pairs safely
        pair_options = []
        for k in bucket_keys:
            # Intra-bucket pairs (requires at least 2 players)
            if len(valid_buckets[k]) >= 2:
                pair_options.append((k, k))
            # Inter-bucket pairs with next bucket
            if k+25 in bucket_keys:
                pair_options.append((k, k+25))
        
        if not pair_options:
            break  # No valid match options
        
        b1, b2 = random.choice(pair_options)
        
        try:
            candidates = valid_buckets[b1]
            p1 = random.choice(candidates)
            
            if b1 == b2:  # Intra-bucket match
                candidates = [p for p in candidates if p != p1]
                if not candidates:
                    continue
                p2 = random.choice(candidates)
            else:  # Inter-bucket match
                p2 = random.choice(valid_buckets[b2])
                
            matches.append((p1, p2))
        except (KeyError, IndexError):
            continue
    
    return matches

def save_results(rating_manager: RatingManager, filename="ratings.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Name", "Rating", "Uncertainty", "Games"])
        sorted_players = sorted(
            rating_manager.ratings.items(),
            key=lambda x: -x[1]['rating']
        )
        for i, (name, data) in enumerate(sorted_players, 1):
            writer.writerow([i, name, data['rating'], data['uncertainty'], data['games_played']])

def init_worker(models: Dict[str, str]):
    """Initialize worker process with model paths"""
    global _worker_agents
    _worker_agents = {name: Agent(path) for name, path in models.items()}

def main():
    model_dir = Path(__file__).parent / "v2"
    rating_manager = RatingManager(model_dir)
    models = {f.stem: str(f) for f in model_dir.glob("*.pth")}
    
    num_matches = 10000
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cpu_count(),
        initializer=init_worker,
        initargs=(models,)
    ) as executor:
        batch_size = 1000
        for _ in range(num_matches // batch_size):
            matches = schedule_matches(rating_manager, batch_size)
            # Pass rating_manager to each match game
            list(executor.map(match_game, matches, [rating_manager]*len(matches)))
            print(f"Completed batch of {batch_size} matches in {time.time()-start_time:.2f}s")
    
    print(f"Completed {num_matches} matches in {time.time()-start_time:.2f}s")
    save_results(rating_manager)

if __name__ == '__main__':
    main()