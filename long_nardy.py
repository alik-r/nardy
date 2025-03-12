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
        results = []

        home_slice = slice(0, 6) if state.is_white else slice(18, 24)
        off_attr = 'white_off' if state.is_white else 'black_off'
        head_pos = 11 if state.is_white else 23

        opp_sign = -1 if state.is_white else 1  # Opponent's sign: -1 for black, 1 for white

        stack = [state.copy()]
    
        while stack:
            current_state = stack.pop()

            # Compute can_bear_off for the current state
            home_pieces = current_state.board[home_slice]
            if state.is_white:
                pieces_in_home = home_pieces[home_pieces > 0].sum()  # Only count white pieces
            else:
                pieces_in_home = -home_pieces[home_pieces < 0].sum()  # Only count black pieces
            pieces_in_home += getattr(current_state, off_attr)
            can_bear_off = (pieces_in_home == 15)

            # Precompute piece positions once per current_state
            piece_positions = np.where((current_state.board * opp_sign) < 0)[0]

            # If the dice are exhausted, continue to the next state
            if not current_state.dice_remaining:
                results.append(current_state)
                continue

            for i in reversed(range(len(current_state.dice_remaining))):
                dice = current_state.dice_remaining[i]
                remaining_dice = current_state.dice_remaining[:i] + current_state.dice_remaining[i+1:]

                if piece_positions.size == 0:
                    # No pieces to move; propagate state with remaining dice
                    new_state = current_state.copy()
                    new_state.dice_remaining = remaining_dice
                    if not remaining_dice:
                        results.append(new_state)
                    else:
                        stack.append(new_state)
                    continue

                valid_move_found = False

                for pos in piece_positions:
                    if self._is_locked(current_state, pos):  # Skip locked pieces
                        continue
                    if pos == head_pos and current_state.head_moved:  # Enforce head move restriction
                        continue

                    new_pos = pos - dice

                    # Handle wrap-around
                    if not current_state.is_white and new_pos < 0:
                        new_pos += 24

                    # Check if opponent's piece is at the new position
                    if current_state.board[new_pos] * opp_sign > 0:
                        continue

                    if (new_pos < 0 or (new_pos < 12 and pos >= 12 and not current_state.is_white)):
                        if can_bear_off:
                            # Handle bearing off
                            new_state = current_state.copy()
                            new_state.board[pos] += opp_sign    
                            setattr(new_state, off_attr, getattr(new_state, off_attr) + 1)
                            new_state.dice_remaining = remaining_dice
                            if not remaining_dice:
                                results.append(new_state)
                            else:
                                stack.append(new_state)
                            valid_move_found = True
                        continue
                    
                    if self._is_blocking_opponent(current_state):
                        continue

                    # Regular move
                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.board[new_pos] -= opp_sign
                    new_state.dice_remaining = remaining_dice
                    if not remaining_dice:
                        results.append(new_state)
                    else:
                        stack.append(new_state)
                    valid_move_found = True
                    if pos == head_pos:
                        new_state.head_moved = True

                    if not valid_move_found:
                        new_state = current_state.copy()
                        new_state.dice_remaining = remaining_dice
                        if not remaining_dice:
                            results.append(new_state)
                        else:
                            stack.append(new_state)
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
    
