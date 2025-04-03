import random
import time
import csv
from pathlib import Path
from typing import List
from multiprocessing import Manager, cpu_count
import concurrent.futures
import numpy as np
import torch
from torch import nn
import logging
from datetime import datetime
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'longnardy_tournament_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cpu")

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
    
    def acquire_lock(self, player_name, purpose, timeout=10):
        logger.debug(f"Player {player_name} waiting for lock ({purpose})")
        acquired = self.lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for {player_name} ({purpose}) after {timeout}s")
        logger.debug(f"Player {player_name} acquired lock ({purpose})")
    
    def release_lock(self, player_name, purpose):
        self.lock.release()
        logger.debug(f"Player {player_name} released lock ({purpose})")

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
        self.shared.acquire_lock(self.name, "get rating")
        try:
            return self.shared.rating.value
        finally:
            self.shared.release_lock(self.name, "get rating")

    @property
    def uncertainty(self):
        self.shared.acquire_lock(self.name, "get uncertainty")
        try:
            return self.shared.uncertainty.value
        finally:
            self.shared.release_lock(self.name, "get uncertainty")

    @property
    def games_played(self):
        self.shared.acquire_lock(self.name, "get games_played")
        try:
            return self.shared.games_played.value
        finally:
            self.shared.release_lock(self.name, "get games_played")

    def __str__(self):
        return f"{self.name}: {self.rating}±{self.uncertainty}"

def play_game(white: Player, black: Player) -> int:
    logger.info(f"Starting game: {white.name} (white) vs {black.name} (black)")
    game = LongNardy()
    current_players = [white, black]
    move_count = 0
    
    while not game.is_finished():
        move_count += 1
        player = current_players[0] if game.state.is_white else current_players[1]
        candidate_states = game.get_states_after_dice()
        
        if not candidate_states:
            game.apply_dice(game.state)
            continue
            
        values = [player.agent.evaluate(state) for state in candidate_states]
        game.step(candidate_states[np.argmax(values)])
        
        if move_count % 10 == 0:
            logger.debug(f"Game progress - move {move_count}: {player.name} playing")
    
    result = 1 if game.state.white_off == 15 else 0
    logger.info(f"Game finished: {white.name} {'won' if result == 1 else 'lost'} "
               f"against {black.name} in {move_count} moves")
    return result

def update_ratings(winner: Player, loser: Player):
    logger.info(f"Updating ratings: {winner.name} vs {loser.name}")
    
    first, second = sorted([winner, loser], key=lambda p: p.name)
    first.shared.acquire_lock(first.name, "update ratings (first)")
    second.shared.acquire_lock(second.name, "update ratings (second)")
    
    try:
        logger.debug(f"Pre-update - {winner.name}: {winner.rating}±{winner.uncertainty}, "
                    f"{loser.name}: {loser.rating}±{loser.uncertainty}")
        
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
        
        logger.debug(f"Post-update - {winner.name}: {winner.rating}±{winner.uncertainty}, "
                    f"{loser.name}: {loser.rating}±{loser.uncertainty}")
    finally:
        second.shared.release_lock(second.name, "update ratings (second)")
        first.shared.release_lock(first.name, "update ratings (first)")

def match_game(pair):
    try:
        white, black = pair
        logger.debug(f"Starting match between {white.name} and {black.name}")
        result = play_game(white, black)
        
        if result == 1:
            update_ratings(white, black)
        else:
            update_ratings(black, white)
        return result
    except Exception as e:
        logger.error(f"Error in match between {pair[0].name} and {pair[1].name}: {str(e)}")
        return None

def schedule_matches(players, num_matches):
    logger.info(f"Scheduling {num_matches} matches among {len(players)} players")
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
        except (KeyError, IndexError) as e:
            logger.warning(f"Error scheduling match: {str(e)}")
            continue
    
    logger.debug(f"Generated {len(matches)} matches for this batch")
    return matches

def log_system_status():
    mem = psutil.virtual_memory()
    logger.info(f"System status - CPU: {psutil.cpu_percent()}%, "
               f"Memory: {mem.used/1024/1024:.1f}MB used ({mem.percent}%)")

def save_results(players: List[Player], filename="ratings.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Name", "Rating", "Uncertainty", "Games"])
        sorted_players = sorted(players, key=lambda p: -p.rating)
        for i, p in enumerate(sorted_players, 1):
            writer.writerow([i, p.name, p.rating, p.uncertainty, p.games_played])
    logger.info(f"Saved results to {filename}")

def main():
    with Manager() as manager:
        model_dir = Path(__file__).parent / "v2"
        players = [
            Player(f.stem, str(f), manager)
            for f in model_dir.glob("*.pth")
        ]
        
        # Share model memory in the main process
        for p in players:
            p.agent.net.share_memory()
        
        num_matches = 300000
        start_time = time.time()
        logger.info(f"Starting tournament with {len(players)} players")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            batch_size = 1000
            total_completed = 0
            
            for batch_num in range(num_matches // batch_size):
                log_system_status()
                logger.info(f"Starting batch {batch_num + 1}")
                
                matches = schedule_matches(players, batch_size)
                results = list(executor.map(match_game, matches))
                
                completed = len([r for r in results if r is not None])
                total_completed += completed
                logger.info(f"Batch {batch_num + 1} completed: {completed} matches "
                           f"(total: {total_completed}, elapsed: {time.time()-start_time:.2f}s)")
                
                # Save intermediate results every 10 batches
                if (batch_num + 1) % 10 == 0:
                    save_results(players)
                    logger.info("Saved intermediate results")
        
        logger.info(f"Tournament completed: {total_completed} matches played in "
                   f"{time.time()-start_time:.2f} seconds")
        save_results(players)

if __name__ == '__main__':
    main()