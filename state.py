import numpy as np
import random

random.seed(0)

class State:
    def __init__(self):
        """
        Initializes the Nardy game state.
        Board representation:
        - 24 points numbered from 1 to 24.

        - White's home board: points 1 to 6.
        - White's outer board: points 7 to 24.
        - Black's home board: points 19 to 24.
        - Black's outer board: points 1 - to 19
        """
        self.board = np.zeros(24, dtype=np.int32)

        # Initial setup: all 15 pieces on the head.
        self.board[11] = -15
        self.board[23] = 15

        # Counters for borne-off pieces.
        self.white_off = 0
        self.black_off = 0

        # Dice roll for current turn.
        self.dice_remaining = []  # dice moves remaining this turn

        # 'white' always starts.
        self.is_white = True
        
        # Flag to enforce that only one move from the head is allowed per turn.
        self.head_moved = False

    def reset(self):
        """
        Resets the state to initial state
        """
        self.board.fill(0)
        self.board[11] = -15
        self.board[23] = 15
        self.white_off = 0
        self.black_off = 0
        self.is_white = True

    def copy(self):
        """
        Creates the copy of the state
        """
        # Optimized deep copy of the state
        new_state = State()

        # Copy the numpy board (efficient copy)
        new_state.board = self.board.copy()

        # Copy the other primitive variables
        new_state.white_off = self.white_off
        new_state.black_off = self.black_off
        new_state.is_white = self.is_white
        new_state.head_moved = self.head_moved

        new_state.dice_remaining = self.dice_remaining.copy()

        return new_state
    
    def roll_dice(self):
        """
        Roll dice for the turn. For a double, four moves are available.
        """
        d1 = random.randint(1, 6)
        d2 = random.randint(1, 6)
        if d1 == d2:
            self.dice_remaining = [d1] * 4
        else:
            self.dice_remaining = [d1, d2]
        self.head_moved = False
    
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
        state_tensor = np.empty(100, dtype=np.float32)

        # Set borne-off pieces and move order.
        state_tensor[0:2] = [self.white_off, self.black_off]
        state_tensor[2:4] = [1.0 if self.is_white else 0.0,
                             0.0 if self.is_white else 1.0]
        
        # Vectorized encoding for board positions:
        board = self.board.astype(np.float32)
        white_presence = (board > 0).astype(np.float32)
        black_presence = (board < 0).astype(np.float32)

        # Subtract one from the count where pieces exist (else zero)
        white_count = np.maximum(board, 0) - 1
        black_count = np.maximum(-board, 0) - 1
        
        # Stack and flatten (each point contributes 4 numbers)
        field_tensor = np.column_stack((white_presence, white_count, black_presence, black_count)).flatten()
        state_tensor[4:] = field_tensor
        return state_tensor
    
    def pretty_print(self):
        """
        Prints the Nardy board state in a visually appealing format.
        Positive values represent White's pieces, and negative values represent Black's pieces.
        """
        print("\nNardy Board State:")
        print("-------------------------------------------------")
        
        # Upper half of the board (points 13 to 24)
        for i in range(12, 24):
            piece = self.board[i]
            piece_str = f"{piece:2d}" if piece != 0 else " ."
            print(f"{i:2d} [{piece_str}]", end="  ")
        
        print("\n-------------------------------------------------")

        # Lower half of the board (points 1 to 12)
        for i in range(11, -1, -1):
            piece = self.board[i]
            piece_str = f"{piece:2d}" if piece != 0 else " ."
            print(f"{i:2d} [{piece_str}]", end="  ")

        print("\n-------------------------------------------------\n")

        print("White Turn: ", self.is_white)
        print("Dice: ", self.dice_remaining)
        print("White off: ", self.white_off)
        print("Black off: ", self.black_off)
        
    def change_turn(self):
        """
        Changes the turn from white to black or vice versa.
        """
        self.is_white = not self.is_white

    def compare(self, other) -> bool:
        return (
            np.array_equal(self.board, other.board) and
            self.white_off == other.white_off and
            self.black_off == other.black_off and
            self.dice_remaining == other.dice_remaining and
            self.is_white == other.is_white and
            self.head_moved == other.head_moved
        )
