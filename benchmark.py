from long_nardy import LongNardy
import torch
from torch import nn
import numpy as np
from state import State
from typing import Tuple, List
from pathlib import Path
import csv
import concurrent.futures
from multiprocessing import cpu_count

MATCH_COUNT = 10000
MAX_WORKERS = cpu_count()

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
        
    def epsilon_greedy(self, candidate_states: List[State]) -> State:
        best_value = float('-inf')
        best_state = None
        for state in candidate_states:
            value = self.evaluate(state)
            if value > best_value:
                best_value = value
                best_state = state
        return best_state

class RandomAgent:
    def epsilon_greedy(self, candidate_states: List[State]) -> State:
        chosen_idx = np.random.randint(len(candidate_states))
        return candidate_states[chosen_idx]

def test_battle(agent: Agent, match_count: int = MATCH_COUNT) -> Tuple[int, int]:
    random_agent = RandomAgent()
    agent_wins = 0
    random_wins = 0
    
    for i in range(match_count):
        side = True if i % 2 == 1 else False
        game = LongNardy()
        moves = 0
        while not game.is_finished():
            moves += 1
            if moves > 10000:
                print("Game took too long, breaking out.")
                break
            if game.state.is_white == side:
                current_agent = random_agent
            else:
                current_agent = agent
            candidate_states = game.get_states_after_dice()
            if not candidate_states:
                # Handle no valid moves by passing turn
                game.apply_dice(game.state)
                continue
            chosen_state = current_agent.epsilon_greedy(candidate_states)
            game.step(chosen_state)

            if game.is_finished():
                if game.state.is_white == side:
                    random_wins += 1
                else:
                    agent_wins += 1
                break
    return agent_wins, random_wins

def evaluate_model(model_path: Path) -> list:
    print(f"Evaluating model: {model_path.name}")
    agent = Agent(epsilon=0.0)
    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    wins, losses = test_battle(agent, MATCH_COUNT)
    win_rate = wins / MATCH_COUNT
    print(f"Model: {model_path.name} - Win rate: {win_rate:.2%}")
    return [model_path.name, wins, losses, win_rate]

def main():
    current_directory = Path(__file__).parent
    models_folder = current_directory / "v2"
    model_paths = list(models_folder.glob("*.pth"))

    already_evaluated = 995000
    new_models = [model_path for model_path in model_paths if int(model_path.stem.split("_")[-1]) > already_evaluated]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_model = {executor.submit(evaluate_model, model_path): model_path for model_path in new_models}
        for future in concurrent.futures.as_completed(future_to_model):
            result = future.result()
            results.append(result)

    output_csv = current_directory / "model_win_rates.csv"
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write header only if the file is empty
        if csv_file.tell() == 0:
            writer.writerow(["Model", "AgentWins", "RandomWins", "WinRate"])

        for row in results:
            writer.writerow(row)

    print(f"\nResults in {output_csv}")

if __name__ == "__main__":
    main()
