import numpy as np
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
        self.white_positions = self._precompute_white_positions()
        self.black_positions = self._precompute_black_positions()

        # Roll dice to begin the first turn.
        self.state.roll_dice()

    def _precompute_white_positions(self):
        positions = []
        for pos in range(24):
            start = pos - 7 if pos > 5 else -1
            end = pos - 1
            pos_range = range(end, start, -1)
            valid = [p for p in pos_range if 0 <= p < 24]
            positions.append(tuple(valid))
        return positions
    
    def _precompute_black_positions(self):
        positions = []
        for pos in range(24):
            if pos > 11:
                start = pos - 7 if pos > 18 else 11
                end = pos - 1
                pos_range = range(end, start, -1)
            else:
                if pos > 6:
                    start = pos - 7
                    end = pos - 1
                    pos_range = range(end, start, -1)
                else:
                    start = pos - 7 if pos > 6 else -1
                    end = pos - 1
                    first = range(end, start, -1)
                    if pos < 7:
                        second = range(23, 18 + pos, -1)
                        pos_range = list(first) + list(second)
                    else:
                        pos_range = first
            valid = [p for p in pos_range if 0 <= p < 24]
            positions.append(tuple(valid))
        return positions
        
    # @profile
    def _is_locked(self, state: State, pos: int) -> bool:
        """
        Check if the piece at board index pos is locked.
        A piece is locked if, in the six absolute points immediately ahead (in anticlockwise order for white,
        clockwise for black), every such point that exists on the board is occupied by the opponent.
        """
        piece = state.board[pos]
        if piece > 0:
            positions = self.white_positions[pos]
            opponent_sign = -1
        else:
            positions = self.black_positions[pos]
            opponent_sign = 1
        
        if not positions:
            return False
        
        return all((state.board[p] * opponent_sign) > 0 for p in positions)

    # @profile
    def _is_blocking_opponent(self, state: State):
        """
        Returns True if all opponent pieces are locked.
        """
        if state.is_white:
            if state.black_off > 0:
                return False
            opponent_positions = np.flatnonzero(state.board < 0)
        else:
            if state.white_off > 0:
                return False
            opponent_positions = np.flatnonzero(state.board > 0)
        
        # Convert to list for faster iteration in Python
        opponent_positions = opponent_positions.tolist()
        
        return all(self._is_locked(state, pos) for pos in opponent_positions)

    # @profile
    def apply_dice(self, state: State) -> list[State]:
        # print("Applying dice for ", len(state.dice_remaining), " dice")
        results = []

        if state.is_white:
            pieces_in_home = state.board[:6].sum() + state.white_off
        else:
            pieces_in_home = -state.board[12:18].sum() + state.black_off

        can_bear_off = (pieces_in_home == 15)

        opp_sign = -1 if state.is_white else 1  # Opponent's sign: -1 for black, 1 for white

        stack = [state.copy()]
    
        while stack:
            # print("Stack size: ", len(stack))
            current_state = stack.pop()
            # print("Dice remaining: ", current_state.dice_remaining)
            # current_state.pretty_print()

            # If the dice are exhausted, continue to the next state
            if len(current_state.dice_remaining) == 0:
                # print("Dice exhausted")
                continue
        
            dice = current_state.dice_remaining.pop()

            piece_positions = np.where((current_state.board * opp_sign < 0))[0]
            # print("Board: ", current_state.board)
            # print("Piece positions: ", piece_positions)

            if len(piece_positions) == 0:
                # print("No pieces to move")            
                results.append(current_state)

            valid_move_found = False  # Flag to track if any valid move was applied.

            for pos in piece_positions:
                if self._is_locked(current_state, pos):  # Skip locked pieces
                    # print("Locked piece")
                    continue
                if pos == 11 and current_state.head_moved:  # Enforce head move restriction
                    # print("Head moved")
                    continue

                new_pos = pos - dice
                # print("pos: ", pos, " new_pos: ", new_pos)

                # Handle wrap-around
                if not current_state.is_white and new_pos < 0:
                    new_pos += 24

                # Check if opponent's piece is at the new position
                if current_state.board[new_pos] * opp_sign > 0:
                    # print("Opponent's piece at new position")
                    continue

                if (new_pos < 0 or (new_pos < 12 and pos >= 12 and not current_state.is_white)):
                    if can_bear_off:
                        # Handle bearing off
                        new_state = current_state.copy()
                        new_state.board[pos] += opp_sign
                        if current_state.is_white:
                            new_state.white_off += 1
                        else:
                            new_state.black_off += 1
                        if len(new_state.dice_remaining) == 0:
                            results.append(new_state)
                        stack.append(new_state)
                        valid_move_found = True
                    else:
                        continue
                elif self._is_blocking_opponent(current_state):
                    # print("Blocking opponent")
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
                    # print("Added new state")
                    # new_state.pretty_print()

                    if pos == 11 or pos == 23:
                        new_state.head_moved = True

            # If no valid move was found for this die, record the state.
            if not valid_move_found:
                # print("No valid move found for dice:", dice, "Recording current state as final.")
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
    
