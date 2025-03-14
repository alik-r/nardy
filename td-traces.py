import torch
from torch import nn
import numpy as np
from long_nardy import LongNardy
from state import State
from typing import Tuple, List
import time

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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

agent = Agent(lr=0.001, epsilon=0.05)

num_episodes = 1000000
save_interval = 100
total_start_time = time.time()

for episode in range(num_episodes):
    game = LongNardy()
    agent.reset_eligibility_traces()
    done = False
    while not done:
        current_value = agent.get_value(game.state, grad=True)
        
        # Get legal moves for current player
        candidate_states = game.get_states_after_dice()
        
        if not candidate_states:
            # Handle no valid moves by passing turn
            game.apply_dice(game.state)
            continue
            
        # Select move using epsilon-greedy
        chosen_state, next_value = agent.epsilon_greedy(candidate_states)
            
        # Make the move
        game.step(chosen_state)

        # Check terminal state
        if game.is_finished():
            reward = 1
            done = True
        else:
            reward = 0

        # Calculate TD error
        td_error = reward + next_value - current_value.detach()
        
        # Update network
        agent.net.zero_grad()
        current_value.backward()
        agent.update_eligibility_traces()
        
        with torch.no_grad():
            for name, param in agent.net.named_parameters():
                param += agent.lr * td_error * agent.eligibility_traces[name]

    # Periodic saving and logging
    if episode % save_interval == 0:
        torch.save(agent.state_dict(), f"td_gammon_selfplay_{episode}.pth")
        total_elapsed_time = time.time() - total_start_time
        print(f"Episode {episode} | Avg TD Error: {td_error.item():.4f} | Time: {total_elapsed_time:.2f}s")
