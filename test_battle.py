from long_nardy import LongNardy
import torch
from torch import nn
import numpy as np
from state import State
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(100, 80),
            nn.Sigmoid(),
            nn.Linear(80, 1),
            nn.Sigmoid()
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
            state_tensor = torch.tensor(state.get_tensor_representation(), 
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
            
    def epsilon_greedy(self, candidate_states: List[State]) -> Tuple[State, torch.Tensor]:
        """Epsilon-greedy selection with perspective flip for opponent"""
        if np.random.rand() < self.epsilon:
            chosen_idx = np.random.randint(len(candidate_states))
            chosen_state = candidate_states[chosen_idx]
            value = self.get_value(chosen_state)
        else:
            with torch.no_grad():
                values = [self.get_value(state) for state in candidate_states]
            
            chosen_idx = np.argmin([v.item() for v in values])
            chosen_state = candidate_states[chosen_idx]
            value = values[chosen_idx]
            
        return chosen_state, value
    
weak = Agent()
strong = Agent()

weak.load_state_dict(torch.load("td_gammon_selfplay_10000.pth", map_location=device))
strong.load_state_dict(torch.load("td_gammon_selfplay_200000.pth", map_location=torch.device('cpu')))

weak.epsilon = 0
strong.epsilon = 0

class RandomAgent:
    def epsilon_greedy(self, candidate_states: List[State]) -> Tuple[State, torch.Tensor]:
        chosen_idx = np.random.randint(len(candidate_states))
        chosen_state = candidate_states[chosen_idx]
        return chosen_state, torch.tensor(0.0)
    
random = RandomAgent()

def test_battle():
    weak_wins = 0
    strong_wins = 0
    for i in range(1000):
        print(f"Game {i}, Weak wins: {weak_wins}, Strong wins: {strong_wins}")
        side = True if i % 2 == 0 else False
        game = LongNardy()
        while not game.is_finished():
            if game.state.is_white == side:
                agent = weak
                first = True
            else:
                agent = strong
                first = False
            candidate_states = game.get_states_after_dice()

            if not candidate_states:
                # Handle no valid moves by passing turn
                game.apply_dice(game.state)
                continue

            chosen_state = agent.epsilon_greedy(candidate_states)[0]
            game.step(chosen_state)

            if game.is_finished():
                if game.state.is_white == side:
                    strong_wins += 1
                else:
                    weak_wins += 1
                break

            
    print(f"Weak wins: {weak_wins}, Strong wins: {strong_wins}")

test_battle()