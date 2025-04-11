import torch
from torch import nn
import numpy as np
from long_nardy import LongNardy
from state import State
from typing import Tuple, List
import time
import os

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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
        with torch.set_grad_enabled(grad):
            state_tensor = torch.tensor(state.get_representation_for_current_player(), 
                                      dtype=torch.float32).to(device)
            return self.net(state_tensor)
        
    def update_eligibility_traces(self):
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                self.eligibility_traces[name] = torch.maximum(param.grad, self.eligibility_traces[name])

    def reset_eligibility_traces(self):
        for name in self.eligibility_traces:
            self.eligibility_traces[name].zero_()
            
    def epsilon_greedy(self, candidate_states: List[State]) -> Tuple[State]:
        if np.random.rand() < self.epsilon:
            chosen_idx = np.random.randint(len(candidate_states))
            chosen_state = candidate_states[chosen_idx]
        else:
            with torch.no_grad():
                values = [self.get_value(state) for state in candidate_states]
            chosen_idx = np.argmax([v.item() for v in values])
            chosen_state = candidate_states[chosen_idx]
        return chosen_state

# Initialize agent and load pretrained weights
agent = Agent(lr=0.001, epsilon=0.05)
pretrained_path = "/v2/td_gammon_selfplay_995000.pth"
agent.load_state_dict(torch.load(pretrained_path, map_location=device))

# Extract starting episode from filename
filename = os.path.basename(pretrained_path)
base_name = os.path.splitext(filename)[0]
starting_episode = int(base_name.split('_')[-1])

# Configure training parameters
num_additional_episodes = 2000000
start_episode = starting_episode + 1
end_episode = start_episode + num_additional_episodes
save_interval = 5000
total_start_time = time.time()

# Modified training loop
for episode in range(start_episode, end_episode):
    game = LongNardy()
    agent.reset_eligibility_traces()
    done = False

    td_errors = [] if episode % save_interval == 0 else None

    while not done:
        current_value = agent.get_value(game.state, grad=True)
        candidate_states = game.get_states_after_dice()
        
        if not candidate_states:
            game.apply_dice(game.state)
            continue
            
        chosen_state = agent.epsilon_greedy(candidate_states)
        game.step(chosen_state)

        reward = 1 if game.is_finished() else 0
        done = game.is_finished()
        next_value = agent.get_value(game.state, grad=False)

        td_error = reward + (1 - next_value) - current_value.detach()
        if td_errors is not None:
            td_errors.append(td_error.item())

        agent.net.zero_grad()
        current_value.backward()
        agent.update_eligibility_traces()
        
        with torch.no_grad():
            for name, param in agent.net.named_parameters():
                param += agent.lr * td_error * agent.eligibility_traces[name]

    # Save checkpoint and log progress
    if episode % save_interval == 0:
        mean_td_error = sum(td_errors)/len(td_errors) if td_errors else 0
        max_td_error = max(td_errors) if td_errors else 0
        min_td_error = min(td_errors) if td_errors else 0

        torch.save(agent.state_dict(), f"saves/td_gammon_selfplay_{episode}.pth")
        total_elapsed_time = time.time() - total_start_time
        print(f"Episode {episode} | Mean TD: {mean_td_error:.4f} | "
              f"Max TD: {max_td_error:.4f} | Min TD: {min_td_error:.4f} | "
              f"Time: {total_elapsed_time:.2f}s")