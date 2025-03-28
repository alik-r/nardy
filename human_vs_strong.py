import numpy as np
import torch
from typing import List
from long_nardy import LongNardy
from state import State
from pathlib import Path
from torch import nn

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
    def __init__(self):
        super().__init__()
        self.net = ANN().to(device)
        self.epsilon = 0.0  # We want the strong agent to make its best moves

    def get_value(self, state: State) -> torch.Tensor:
        with torch.no_grad():
            state_tensor = torch.tensor(state.get_representation_for_current_player(), dtype=torch.float32).to(device)
            return self.net(state_tensor)

    def epsilon_greedy(self, candidate_states: List[State]) -> State:
        # Pick the state with highest value
        values = [self.get_value(s) for s in candidate_states]
        best_index = np.argmax([v.item() for v in values])
        return candidate_states[best_index]

strong_agent = Agent()
current_directory = Path(__file__).parent
checkpoint_path = current_directory / "v2" / "td_gammon_selfplay_535000.pth"
strong_agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
strong_agent.eval()

def describe_move(old_state: State, new_state: State) -> str:
    """
    Describe how old_state transitioned to new_state using short "to <- from" arrows.
    Example: "14 <- 13" means a piece moved from point 13 to point 14.
    If bearing off, shows: "Off(white) <- 6".
    """
    desc_moves = []

    # If white_off changed, pieces were borne off from some position(s).
    white_diff = new_state.white_off - old_state.white_off
    black_diff = new_state.black_off - old_state.black_off

    # We'll look at the difference on the board:
    diff = new_state.board - old_state.board

    # For each board point, see if the count changed:
    # A negative difference in old_state.is_white turn usually means "moved from i".
    # If black's turn, the sign is reversed. We'll just detect the from->to by comparing.
    if old_state.is_white:
        # White’s move: white pieces are positive. 
        # "From" points will have a negative diff (lost white piece),
        # "To" points will have a positive diff (gained white piece).
        
        # If any white pieces got borne off:
        # we won't know the exact from-point just by new_state.white_off alone.
        # So we find a point that lost exactly 1 white piece (diff = -1).
        # Then we know it went to "Off(white)".
        if white_diff > 0:
            # We expect exactly 'white_diff' from-points with diff = -1
            # if there were multiple pieces borne off this turn.
            from_points = np.where(diff == -1)[0]  # all positions that lost exactly 1 piece
            for pos in from_points:
                desc_moves.append(f"Off(white) <- {pos}")

        # For normal from->to moves on the board:
        # each from-point has diff < 0, each to-point has diff > 0.
        # We'll pair them up by absolute value so we handle multiple moves if we had a double.
        from_points = []
        to_points = []
        for i in range(24):
            if diff[i] < 0:
                from_points.append((i, abs(diff[i])))  # store count
            elif diff[i] > 0:
                to_points.append((i, diff[i]))         # store count

        # We'll match them up in any order just for a textual summary
        fp_idx = 0
        tp_idx = 0
        while fp_idx < len(from_points) and tp_idx < len(to_points):
            from_i, from_count = from_points[fp_idx]
            to_i, to_count = to_points[tp_idx]
            
            # The number of pieces to move is the minimum of these two counts
            move_count = min(from_count, to_count)
            for _ in range(move_count):
                desc_moves.append(f"{to_i} <- {from_i}")
            # Update the leftover pieces at from_i or to_i
            from_points[fp_idx] = (from_i, from_count - move_count)
            to_points[tp_idx] = (to_i, to_count - move_count)
            # If from_count is now 0, move on
            if from_points[fp_idx][1] == 0:
                fp_idx += 1
            # If to_count is now 0, move on
            if to_points[tp_idx][1] == 0:
                tp_idx += 1

    else:
        # Black’s move: black pieces are negative on the board.
        # If black bared off pieces:
        if black_diff > 0:
            from_points = np.where(diff == +1)[0]  # black is negative, so "from" is +1
            for pos in from_points:
                desc_moves.append(f"Off(black) <- {pos}")

        # For on-board moves, "from" points have diff>0 (lost negative piece),
        # "to" points have diff<0 (gained negative piece).
        from_points = []
        to_points = []
        for i in range(24):
            if diff[i] > 0:
                from_points.append((i, diff[i]))  # black "from" count
            elif diff[i] < 0:
                to_points.append((i, abs(diff[i])))

        fp_idx = 0
        tp_idx = 0
        while fp_idx < len(from_points) and tp_idx < len(to_points):
            from_i, from_count = from_points[fp_idx]
            to_i, to_count = to_points[tp_idx]
            
            move_count = min(from_count, to_count)
            for _ in range(move_count):
                desc_moves.append(f"{to_i} <- {from_i}")
            from_points[fp_idx] = (from_i, from_count - move_count)
            to_points[tp_idx] = (to_i, to_count - move_count)
            if from_points[fp_idx][1] == 0:
                fp_idx += 1
            if to_points[tp_idx][1] == 0:
                tp_idx += 1

    if not desc_moves:
        return "Pass / No Change"
    else:
        # Return each move on one line, or separate with commas
        return ", ".join(desc_moves)

def human_choose_move(game: LongNardy) -> State:
    """
    Let the human pick one of the candidate moves. 
    If there is exactly one legal move, we take it automatically.
    If there are no moves, returns None.
    """
    candidates = game.get_states_after_dice()
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    old_state = game.state
    # Show each candidate move with a small textual difference
    print("\nPossible moves:")
    for idx, st in enumerate(candidates):
        move_desc = describe_move(old_state, st)
        print(f"  [{idx}] {move_desc}")

    while True:
        choice_str = input("Choose move index: ")
        if not choice_str.isdigit():
            print("Please enter a valid index.")
            continue
        choice = int(choice_str)
        if 0 <= choice < len(candidates):
            return candidates[choice]
        print("Invalid index. Try again.")

def play_vs_strong():
    """
    Runs a single game in which the user (human) plays against the strong agent.
    """
    # Ask which side the user wants
    side_input = input("Do you want to play as White (w) or Black (b)? ").lower().strip()
    user_is_white = (side_input == 'w')

    # Initialize game
    game = LongNardy()

    # Loop until game finished
    while not game.is_finished():
        game.state.pretty_print()

        if game.state.is_white == user_is_white:
            print("\nYour turn!")
            chosen_state = human_choose_move(game)
            if chosen_state is None:
                # No valid moves => pass
                print("No moves. Passing.")
                # artificially create a pass move: dice are cleared, turn changes
                pass_state = game.state.copy()
                pass_state.dice_remaining = []
                pass_state.change_turn()
                game.step(pass_state)
            else:
                game.step(chosen_state)
        else:
            print("\nStrong agent's turn...")
            candidates = game.get_states_after_dice()
            if not candidates:
                print("Strong agent has no moves. Passing.")
                pass_state = game.state.copy()
                pass_state.dice_remaining = []
                pass_state.change_turn()
                game.step(pass_state)
            else:
                chosen_state = strong_agent.epsilon_greedy(candidates)
                game.step(chosen_state)

    # Game finished, print results
    game.state.pretty_print()
    if game.state.white_off == 15:
        winner = "White"
    else:
        winner = "Black"

    print(f"Game over! The winner is: {winner}")
    if (winner == "White" and user_is_white) or (winner == "Black" and not user_is_white):
        print("You won! Congratulations!")
    else:
        print("The strong agent won!")

if __name__ == "__main__":
    play_vs_strong()