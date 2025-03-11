import random
import copy
import torch
import numpy as np
from itertools import combinations
from numba import njit

class LongNardy:
    """
    Board representation:
      - The game is played on a common board of 24 points.
      - Each player has pieces placed on the board as follows:
        For White:
            * Points 1 to 6: home board (represented by indices 0 to 5)
            * Points 19 to 24: table (represented by indices 6 to 11)
        For Black:
            * Points 13 to 18: home board (represented by indices 12 to 17)
            * Points 7 to 12: table (represented by indices 18 to 23)
      - Movement is anticlockwise - a move subtracts the dice pips from the current point.
      - The head for White is at **point 24** (index 11), and the head for Black is at **point 12** (index 23).
      
    Locking (prime) rules:
      - A piece is considered locked if the six consecutive absolute points (ahead in the moving direction)
        are all occupied by the opponent.
      - A move that would result in a destination point having six pieces (i.e. forming a block)
        is legal only if at least one opponent checker is in his home.
      - Moreover, it is forbidden to build a block that would lock all 15 of the opponent's checkers.
    """

    def __init__(self):
        # Initialize a 2x12 matrix where:
        # - Row 0 represents white pieces
        # - Row 1 represents black pieces
        self.board = np.zeros(24, dtype=np.int32)

        # Initial setup: all 15 pieces on the head.
        self.board[11] = 15
        self.board[23] = -15

        # Counters for borne-off pieces.
        self.white_off = 0
        self.black_off = 0

        # 'white' always starts.
        self.is_white = True
        
        # Dice roll for current turn.
        self.dice = []            # full dice roll for the turn
        self.dice_remaining = []  # dice moves remaining this turn
        
        # Flag to enforce that only one move from the head is allowed per turn.
        self.head_moved = False

        # Roll dice to begin the first turn.
        self.roll_dice()

    @njit
    def get_tensor_representation(self):
        """
        Returns a tensor representation of the game state as described:
        - 2 inputs for the number of borne-off pieces for white and black.
        - 2 inputs for whether it is white or black move order.
        - For each of the 24 fields:
            - 1 input indicating if there is at least one white piece (1 if yes, 0 if no).
            - 1 input indicating the number of white pieces minus 1 (0 if no white pieces).
            - 1 input indicating if there is at least one black piece (1 if yes, 0 if no).
            - 1 input indicating the number of black pieces minus 1 (0 if no black pieces).
        Total tensor size: 2 (borne-off) + 2 (move order) + 24 * 4 (fields) = 100.
        """
        # Initialize the tensor to hold the game state
        state_tensor = np.zeros(100, dtype=np.float32)

        # Set borne-off pieces
        state_tensor[0] = self.white_off
        state_tensor[1] = self.black_off

        # Set move order (white's turn or black's turn)
        state_tensor[2] = 1 if self.is_white else 0
        state_tensor[3] = 1 if not self.is_white else 0

        # Access the board and use it directly for calculations
        board = self.board
        
        # Vectorized calculation for presence of white and black pieces at each position
        white_presence = (board > 0).astype(np.float32)  # 1 if white piece is present, else 0
        black_presence = (board < 0).astype(np.float32)  # 1 if black piece is present, else 0

        # Vectorized calculation for number of pieces (white and black)
        num_white = np.maximum(0, board)  # White piece count (0 for no white pieces)
        num_black = np.maximum(0, -board)  # Black piece count (0 for no black pieces)

        # Directly populate the tensor for each position
        for i in range(24):
            base_index = 4 + i * 4
            state_tensor[base_index] = white_presence[i]  # White presence (0 or 1)
            state_tensor[base_index + 1] = num_white[i] - 1  # White pieces - 1 (0 if no white pieces)
            state_tensor[base_index + 2] = black_presence[i]  # Black presence (0 or 1)
            state_tensor[base_index + 3] = num_black[i] - 1  # Black pieces - 1 (0 if no black pieces)

        return state_tensor

    def roll_dice(self):
        """
        Roll dice for the turn. For a double, four moves are available.
        """
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        if d1 == d2:
            self.dice = [d1] * 4
        else:
            self.dice = [d1, d2]
        self.dice_remaining = self.dice.copy()
        self.head_moved = False

    def reset(self):
        """
        Reset the game state.
        """
        self.board = np.zeros(24, dtype=np.int32)
        self.board[11] = 15
        self.board[23] = -15
        self.white_off = 0
        self.black_off = 0
        self.is_white = True
        self.roll_dice()
        
    @njit
    def _is_locked(self, pos):
        """
        Check if the piece at board index pos (for the current player) is locked.
        A piece is locked if, in the six absolute points immediately ahead (in anticlockwise order for white,
        clockwise for black), every such point that exists on the board is occupied by the opponent.
        """
        # Define opponent's piece condition: negative values are for black, positive values are for white
        opponent_sign = -1 if self.is_white else 1

        # White moves anticlockwise (higher to lower indices), Black moves clockwise (lower to higher indices)
        step = -1 if self.is_white else 1

        # Check 6 consecutive positions ahead in the moving direction
        for offset in range(1, 7):
            check_pos = pos + step * offset
            
            # If the position is out of bounds (borne-off area), stop the check
            if check_pos < 0 or check_pos >= 24:
                break
            
            # If the opponent does not occupy the position, the piece is not locked
            if (self.board[check_pos] * opponent_sign) <= 0:
                return False

        return True  # All positions are occupied by the opponent, so the piece is locked

    @njit
    def _would_block_all_opponent(self, candidate_move):
        """
        Simulate applying candidate_move and check if it would block (lock) all 15 of the opponent's pieces.
        Returns True if after the move all opponent pieces are locked.
        """
        # Clone the current game state
        clone_board = np.copy(self.board)
        opp_sign = -1 if self.is_white else 1  # Opponent's sign: -1 for black, 1 for white
        
        # Decompose the candidate_move into its components
        from_pos, move_distance = candidate_move
        new_pos = from_pos - move_distance  # Compute the new position after the move

        # Apply the move on the cloned board
        clone_board[from_pos] += opp_sign  # Remove the piece from the original position
        clone_board[new_pos] -= opp_sign  # Place the piece at the new position

        # Initialize locked pieces counter
        sim_locked = 0

        # Get the positions where the opponent has pieces 
        opponent_positions = np.where((clone_board * opp_sign > 0))[0]

        # Check opponent pieces on the cloned board
        for pos in opponent_positions:
            if self._is_locked(pos):
                    sim_locked += 1  # Increment locked pieces count
        
        # Return True if all opponent pieces are locked, otherwise False
        return sim_locked == 15

    def get_valid_moves(self):
        """
        Generate and return a list of all valid moves for the current player given the current dice_remaining.
        Moves that violate the locking rules or would illegally form a block (prime) are filtered out.
        """
        valid_moves = []
        board = self.white if self.current_player == 'white' else self.black
        opponent_home = self.black[:6] if self.current_player == 'white' else self.white[:6]
        
        # Check if bearing off is allowed
        pieces_in_home = board[:6].sum()
        can_bear_off = (pieces_in_home == 15)

        # Precompute opponent home sum
        opponent_home_count = opponent_home.sum()

        # Collect single dice values and possible combined values
        available_dice = self.dice_remaining
        single_moves = available_dice
        combined_moves = {sum(pair) for pair in combinations(available_dice, 2)} if len(available_dice) > 1 else set()

        # Iterate through board positions
        for pos in range(12):
            if board[pos] == 0:  # Skip empty positions early
                continue
            if self._is_locked(self.current_player, pos):  # Skip locked pieces
                continue
            if pos == 11 and self.head_moved:  # Enforce head move restriction
                continue

            # Check single dice moves
            for d in single_moves:
                new_pos = pos - d

                if new_pos >= 0:  # Normal move
                    if board[new_pos] == 5 and opponent_home_count == 0:  # Prime block rule
                        continue
                    if board[new_pos] == 5 and self._would_block_all_opponent((pos, d, (d,))):
                        continue
                    valid_moves.append((pos, d))

                elif can_bear_off:  # Bearing off move
                    if pos == d or (d > pos and board[pos+1:6].sum() == 0):
                        valid_moves.append((pos, d))

            # Check combined dice moves
            for d in combined_moves:
                new_pos = pos - d

                if new_pos >= 0:
                    if board[new_pos] == 5 and opponent_home_count == 0:
                        continue
                    if board[new_pos] == 5 and self._would_block_all_opponent((pos, d, 'combined')):
                        continue
                    valid_moves.append((pos, d))

                elif can_bear_off:
                    if pos == d or (d > pos and board[pos+1:6].sum() == 0):
                        valid_moves.append((pos, d))

        return valid_moves

    def step(self, action):
        """
        Execute the given action and update the game state.
        
        The action is a tuple (from_pos, move_distance, dice_info).
        This method updates the board, removes the dice used,
        and if no moves remain it switches the turn (with a new dice roll).
        
        Returns:
            state: current game state.
            done: whether the game is finished.
        """
        current = self.current_player
        board = self.white if current == 'white' else self.black

        from_pos, move_distance, dice_info = action
        new_pos = from_pos - move_distance

        # Remove one piece from starting position.
        board[from_pos] -= 1

        # Move piece or bear off.
        if new_pos >= 0:
            board[new_pos] += 1
        else:
            if current == 'white':
                self.white_off += 1
            else:
                self.black_off += 1

        # Mark head move if applicable.
        if from_pos == 11:
            self.head_moved = True

        # Remove used dice.
        if dice_info == 'combined':
            found = False
            for i in range(len(self.dice_remaining)):
                for j in range(i + 1, len(self.dice_remaining)):
                    if self.dice_remaining[i] + self.dice_remaining[j] == move_distance:
                        self.dice_remaining.pop(j)
                        self.dice_remaining.pop(i)
                        found = True
                        break
                if found:
                    break
        else:
            die_val = dice_info[0]
            self.dice_remaining.remove(die_val)

        # If no dice remain or no further moves are valid, end the turn.
        if len(self.dice_remaining) == 0 or len(self.get_valid_moves()) == 0:
            self.current_player = 'black' if current == 'white' else 'white'
            self.roll_dice()

        state = self.get_state()
        return state

    def is_finished(self):
        """
        The game ends when one player has borne off all 15 pieces.
        """
        return self.white_off == 15 or self.black_off == 15

    def print(self):
        """
        Print a human-readable view of the current state.
        """
        print("Current player:", self.current_player)
        print("White board (indices 0-5: home [points 1-6], 6-11: table [points 19-24]):")
        print(self.white)
        print("White borne off:", self.white_off)
        print("")
        print("Black board (indices 0-5: home [points 13-18], 6-11: table [points 7-12]):")
        print(self.black)
        print("Black borne off:", self.black_off)
        print("")
        print("Dice remaining:", self.dice_remaining)
        print("-" * 40)

    def get_state(self):
        """
        Return a dictionary representing the current state.
        """
        return {
            'current_player': self.current_player,
            'white_board': self.white.clone(),
            'black_board': self.black.clone(),
            'white_off': self.white_off,
            'black_off': self.black_off,
            'dice_remaining': self.dice_remaining.copy()
        }

    def pretty_print_board(self):
        """
        Print a visual representation of the board in two rows.
        
        The top row shows absolute points 24 to 13 and the bottom row shows points 1 to 12.
        For each point, if there are pieces present, the initial of the occupant (W or B)
        and the count are displayed.
        """
        abs_board = self.get_absolute_board()
        # Top row: points 24 to 13
        top_points = list(range(24, 12, -1))
        # Bottom row: points 1 to 12
        bottom_points = list(range(1, 13))
        
        def format_point(pt):
            occupant, count = abs_board[pt]
            if count > 0:
                return f"{pt:2}[{occupant[0].upper()}{count:2}]"
            else:
                return f"{pt:2}[    ]"
        
        top_row = "  ".join(format_point(pt) for pt in top_points)
        bottom_row = "  ".join(format_point(pt) for pt in bottom_points)
        
        print("=" * 60)
        print(f"Current Player: {self.current_player.upper()}")
        print(f"White borne off: {self.white_off}   Black borne off: {self.black_off}")
        print("Dice remaining:", self.dice_remaining)
        print("-" * 60)
        print("TOP ROW (Points 24 to 13):")
        print(top_row)
        print("-" * 60)
        print("BOTTOM ROW (Points 1 to 12):")
        print(bottom_row)
        print("=" * 60)