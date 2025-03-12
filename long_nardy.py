import random
import copy
import torch
import numpy as np
from itertools import permutations
from state import State
from concurrent.futures import ProcessPoolExecutor

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
      - The head for White is at point 24, and the head for Black is at point 12.
      
    Locking (prime) rules:
      - A piece is considered locked if the six consecutive absolute points (ahead in the moving direction)
        are all occupied by the opponent.
      - A move that would result in a destination point having six pieces (i.e. forming a block)
        is legal only if at least one opponent checker is in his home.
      - Moreover, it is forbidden to build a block that would lock all 15 of the opponent's checkers.
    """

    def __init__(self):
        self.state = State()

        # Roll dice to begin the first turn.
        self.state.roll_dice()
        
    def _is_locked(self, state: State, pos):
        """
        Check if the piece at board index pos is locked.
        A piece is locked if, in the six absolute points immediately ahead (in anticlockwise order for white,
        clockwise for black), every such point that exists on the board is occupied by the opponent.
        """
        # Define opponent's piece condition: negative values are for black, positive values are for white
        opponent_sign = -1 if state.is_white else 1

        # White moves anticlockwise (higher to lower indices), Black moves clockwise (lower to higher indices)
        step = -1 if state.is_white else 1

        # Check 6 consecutive positions ahead in the moving direction
        for offset in range(1, 7):
            check_pos = pos + step * offset

            if not state.is_white and check_pos < 0:
                check_pos += 24
            
            # If the position is out of bounds (borne-off area), stop the check
            if check_pos < 0 or check_pos >= 24:
                break
            
            # If the opponent does not occupy the position, the piece is not locked
            if (state.board[check_pos] * opponent_sign) <= 0:
                return False

        return True  # All positions are occupied by the opponent, so the piece is locked

    def _is_blocking_opponent(self, state: State):
        """
        Returns True if all opponent pieces are locked.
        """
        opp_sign = -1 if state.is_white else 1  # Opponent's sign: -1 for black, 1 for white

        # Initialize locked pieces counter
        sim_locked = 0

        # Get the positions where the opponent has pieces 
        opponent_positions = np.where((state.board * opp_sign > 0))[0]

        # Check opponent pieces on the cloned board
        for pos in opponent_positions:
            if self._is_locked(state, pos):
                    sim_locked += 1  # Increment locked pieces count
        
        # Return True if all opponent pieces are locked, otherwise False
        return sim_locked == 15

    def apply_dice(self, state: State) -> list[State]:
        print("Applying dice for ", len(state.dice_remaining), " dice")
        if len(state.dice_remaining) == 0:
            print("No dice remaining")
            return []
        
        results = []

        if state.is_white:
            pieces_in_home = state.board[:6].sum()
        else:
            pieces_in_home = state.board[12:18].sum()

        can_bear_off = (pieces_in_home == 15)

        opp_sign = -1 if state.is_white else 1  # Opponent's sign: -1 for black, 1 for white

        stack = [state.copy()]
    
        while stack:
            print("Stack size: ", len(stack))
            current_state = stack.pop()
            print("Dice remaining: ", current_state.dice_remaining)
            current_state.pretty_print()

            # If the dice are exhausted, continue to the next state
            if len(current_state.dice_remaining) == 0:
                print("Dice exhausted")
                continue
        
            dice = current_state.dice_remaining.pop()

            piece_positions = np.where((current_state.board * opp_sign < 0))[0]
            print("Board: ", current_state.board)
            print("Piece positions: ", piece_positions)

            if len(piece_positions) == 0:
                print("No pieces to move")            
                results.append(current_state)

            valid_move_found = False  # Flag to track if any valid move was applied.

            for pos in piece_positions:
                if self._is_locked(current_state, pos):  # Skip locked pieces
                    print("Locked piece")
                    continue
                if pos == 11 and current_state.head_moved:  # Enforce head move restriction
                    print("Head moved")
                    continue

                new_pos = pos - dice
                print("pos: ", pos, " new_pos: ", new_pos)

                if not current_state.is_white and new_pos < 0:
                    new_pos += 24

                if current_state.board[new_pos] * opp_sign > 1:
                    print("Opponent's piece at new position")
                    continue

                if (new_pos < 0 or (new_pos < 12 and pos >= 12 and current_state.is_white)) and can_bear_off:
                    # Handle bearing off
                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.white_off += 1
                    if len(new_state.dice_remaining) == 0:
                        results.append(new_state)
                    stack.append(new_state)
                    valid_move_found = True
                elif self._is_blocking_opponent(current_state):
                    print("Blocking opponent")
                    continue
                else:
                    # Regular move
                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.board[new_pos] -= opp_sign
                    if len(new_state.dice_remaining) == 0:
                        results.append(new_state)
                    stack.append(new_state)
                    valid_move_found = True
                    print("Added new state")
                    new_state.pretty_print()

                    if pos == 11 or pos == 23:
                        new_state.head_moved = True

            # If no valid move was found for this die, record the state.
            if not valid_move_found:
                print("No valid move found for dice:", dice, "Recording current state as final.")
                results.append(current_state)
        return results
    
    def get_states_after_dice(self) -> list[State]:
        return self.apply_dice(self.state)

    def step(self, state: State):
        """
        Apply the new state to the game.
        """
        self.state = state
        self.state.roll_dice()
        self.state.change_turn()

    def is_finished(self):
        """
        The game ends when one player has borne off all 15 pieces.
        """
        return self.state.white_off == 15 or self.state.black_off == 15
    
