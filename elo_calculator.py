import threading
import random
import concurrent.futures
import time
from long_nardy import LongNardy
import torch
from torch import nn
import numpy as np
from state import State
from typing import Tuple, List
from pathlib import Path
import os
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

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
        
    def get_value(self, state: State, grad=False) -> torch.Tensor:
        """Get V(s) with optional gradient tracking"""
        with torch.set_grad_enabled(grad):
            state_tensor = torch.tensor(state.get_representation_for_current_player(), 
                                      dtype=torch.float32).to(device)
            return self.net(state_tensor)
        
    def update_eligibility_traces(self):
        """Update traces with current gradients (lambda=1, gamma=1)"""
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                self.eligibility_traces[name] = param.grad + self.eligibility_traces[name]

    def reset_eligibility_traces(self):
        for name in self.eligibility_traces:
            self.eligibility_traces[name].zero_()
            
    def epsilon_greedy(self, candidate_states: List[State]) -> State:
        """Epsilon-greedy selection with perspective flip for opponent"""
        if np.random.rand() < self.epsilon:
            chosen_idx = np.random.randint(len(candidate_states))
            chosen_state = candidate_states[chosen_idx]
        else:
            with torch.no_grad():
                values = [self.get_value(state) for state in candidate_states]
            
            chosen_idx = np.argmax([v.item() for v in values])
            chosen_state = candidate_states[chosen_idx]
            
        return chosen_state

class Player:
    def __init__(self, name, path):
        self.name = name
        self.rating = 1500.0
        self.uncertainty = 350.0  # initial uncertainty
        self.lock = threading.Lock()  # for thread-safe updates
        self.agent = Agent()
        self.agent.load_state_dict(torch.load(path, map_location=device))
        self.agent.eval()
        self.agent.epsilon = 0.0  # Disable exploration for the agent

    def __str__(self):
        return f"{self.name}: Rating={self.rating:.2f}, Uncertainty={self.uncertainty:.2f}"

def play_game(white: Player, black: Player) -> int:
    """
    Simulate a game between two players.
    The win probability is computed using the Elo expected score formula.
    Returns:
      1 if white wins,
      0 if black wins.
    """
    game = LongNardy()
    while not game.is_finished():
        if game.state.is_white:
            player = white
        else:
            player = black
        candidate_states = game.get_states_after_dice()

        if not candidate_states:
            # Handle no valid moves by passing turn
            game.apply_dice(game.state)
            continue

        chosen_state = player.agent.epsilon_greedy(candidate_states)
        game.step(chosen_state)

        if game.is_finished():
            if game.state.white_off == 15:
                return 1
            else:
                return 0
            

def update_rating(player, opponent, score, k_factor):
    """
    Update a player's rating based on the result of a match.
    
    player: The player whose rating is updated.
    opponent: The opponent player.
    score: Actual score (1 for win, 0 for loss).
    k_factor: The dynamic K factor based on player's uncertainty.
    """
    expected = 1.0 / (1.0 + 10 ** ((opponent.rating - player.rating) / 400))
    delta = k_factor * (score - expected)
    player.rating += delta
    # Decrease uncertainty (but not below 50)
    player.uncertainty = max(50, player.uncertainty * 0.95)

def match_game(white, black):
    """
    Simulate a match between two players (white and black), update their ratings, and return the result.
    """
    result = play_game(white, black)

    # Determine K factor based on current uncertainty:
    def k_factor(player):
        if player.uncertainty > 100:
            return 80
        elif player.uncertainty > 50:
            return 40
        else:
            return 20

    k_white = k_factor(white)
    k_black = k_factor(black)

    # Acquire both locks to safely update ratings concurrently.
    with white.lock, black.lock:
        if result == 1:
            # White wins, black loses.
            update_rating(white, black, 1, k_white)
            update_rating(black, white, 0, k_black)
        else:
            # Black wins, white loses.
            update_rating(white, black, 0, k_white)
            update_rating(black, white, 1, k_black)
    return result

def schedule_matches(players, num_matches):
    """
    Create a list of random pairings for matches.
    Each match is a tuple (white, black), randomly selected from the players list.
    """
    matches = []
    for _ in range(num_matches):
        white, black = random.sample(players, 2)
        matches.append((white, black))
    return matches

def save_results_to_csv(players, filename="final_ratings.csv"):
    """Save player ratings to a CSV file."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Player Name", "Elo Rating", "Uncertainty"])
        for player in players:
            writer.writerow([player.name, round(player.rating, 2), round(player.uncertainty, 2)])
    print(f"Results saved to {filename}")

def main():
    players = []
    current_directory = Path(__file__).parent
    for path in os.listdir(current_directory / "v2"):
        if path.endswith(".pth"):
            player_name = path.split(".")[0]
            player_path = current_directory / "v2" / path
            players.append(Player(player_name, player_path))
    
    # Determine how many matches to run.
    # For instance, within 2 days (172800 seconds) and 0.5 seconds per game,
    # the maximum number of matches is 345,600.
    # Here, we simulate a smaller number (e.g., 10000 matches) as a demo.
    num_matches = 100

    # Using a ThreadPoolExecutor to run matches concurrently.
    max_workers = os.cpu_count()
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for white, black in schedule_matches(players, num_matches):
            futures.append(executor.submit(match_game, white, black))
        # Optionally, process results as they complete.
        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()
            except Exception as exc:
                print(f"A game generated an exception: {exc}")

    elapsed = time.time() - start_time
    print(f"Completed {num_matches} matches in {elapsed:.2f} seconds.\nFinal player ratings:")
    for player in players:
        print(player)

    save_results_to_csv(players)
    print("All matches completed.")

if __name__ == '__main__':
    main()
