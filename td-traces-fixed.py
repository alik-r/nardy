#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import numpy as np
from pathlib import Path
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

# Set up paths
current_dir = Path.cwd()
save_dir = current_dir / "v2"
save_dir.mkdir(parents=True, exist_ok=True)

# Initialize agent and load pretrained weights
agent = Agent(lr=0.002, epsilon=0.05)
pretrained_path = current_dir / "v2" / "td_gammon_selfplay_1170000.pth"

if pretrained_path.exists():
    agent.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"Loaded pretrained weights from {pretrained_path}")
else:
    print(f"No pretrained weights found at {pretrained_path}, starting from scratch")

# Extract starting episode from filename
start_episode = int(pretrained_path.stem.split("_")[-1]) if pretrained_path.exists() else 0
num_additional_episodes = 2000000
end_episode = start_episode + num_additional_episodes
save_interval = 5000

# Training loop
total_start_time = time.time()
for episode in range(start_episode + 1, end_episode + 1):
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
        checkpoint_path = save_dir / f"td_gammon_selfplay_{episode}.pth"
        torch.save(agent.state_dict(), checkpoint_path)
        
        stats = {
            'mean_td': sum(td_errors)/len(td_errors) if td_errors else 0,
            'max_td': max(td_errors) if td_errors else 0,
            'min_td': min(td_errors) if td_errors else 0,
            'time': time.time() - total_start_time
        }
        
        print((f"Episode {episode} | "
               f"Mean TD: {stats['mean_td']:.4f} | "
               f"Max TD: {stats['max_td']:.4f} | "
               f"Min TD: {stats['min_td']:.4f} | "
               f"Time: {stats['time']:.2f}s"))