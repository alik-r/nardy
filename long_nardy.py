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
        self._precalculated_white = self._precalculate_white()
        self._precalculated_black = self._precalculate_black()

        # Roll dice to begin the first turn.
        self.state.roll_dice()

    def _precalculate_white(self) -> np.ndarray:
        positions = []
        for pos in range(24):
            start = pos - 7 if pos > 5 else -1
            end = pos - 1
            pos_range = range(end, start, -1)
            valid = [p for p in pos_range if 0 <= p < 24]
            positions.append(tuple(valid))
        return positions
    
    def _precalculate_black(self) -> np.ndarray:
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

    def are_pieces_after_pos(self, state: State, pos: int) -> bool:
        sign = -1 if state.is_white else 1

        if state.is_white:
            positions = self._precalculated_black[pos]
        else:
            positions = self._precalculated_white[pos]

        print(f"Checking positions after {pos}: {positions}")
        for p in positions:
            print(f"Checking piece at {p} with value {state.board[p]}")
            if state.board[p] * sign > 0:
                print(f"Found piece at {p}")
                return True
        print(f"No pieces after {pos}") 
        return False

    def _is_illegal(self, state: State) -> bool:
        sign = 1 if state.is_white else -1

        consequitive_count = 0
        for index, num in np.ndenumerate(state.board):
            if num * sign > 0:
                consequitive_count += 1
            if consequitive_count > 5:
                if not self.are_pieces_after_pos(state, index[0]):
                    print(f"Locking rule violated at position {index[0]}")
                    return True
                consequitive_count = 0
        return False

    # @profile
    def apply_dice(self, state: State) -> list[State]:
        results = []
        seen_configs = set()

        home_slice = slice(0, 6) if state.is_white else slice(18, 24)
        off_attr = 'white_off' if state.is_white else 'black_off'
        head_pos = 11 if state.is_white else 23
        opp_sign = -1 if state.is_white else 1

        stack = [state.copy()]
        
        while stack:
            current_state = stack.pop()
            current_key = board_key(current_state)
            
            if current_key in seen_configs:
                continue
            seen_configs.add(current_key)

            # print("Trying this state:")
            # current_state.pretty_print()

            home_pieces = current_state.board[home_slice]
            if state.is_white:
                pieces_in_home = home_pieces[home_pieces > 0].sum()
            else:
                pieces_in_home = -home_pieces[home_pieces < 0].sum()
            pieces_in_home += getattr(current_state, off_attr)
            can_bear_off = (pieces_in_home == 15)

            if not current_state.dice_remaining:
                current_state.change_turn()
                results.append(current_state)
                continue

            dice_remaining = current_state.dice_remaining
            # Check if all dice are the same
            if len(dice_remaining) > 0 and dice_remaining[0] == dice_remaining[1]:
                # print("dice is same:", dice_remaining)
                dice_value = dice_remaining[0]
                num_dice = len(dice_remaining)
                new_remaining = [dice_value] * (num_dice - 1) if num_dice > 1 else []
                piece_positions = np.where((current_state.board * opp_sign) < 0)[0]
                valid_move_found = False

                for pos in piece_positions:
                    if pos == head_pos and current_state.head_moved:
                        # print(f"Head piece at {pos} already moved")
                        continue

                    new_pos = pos - dice_value
                    if not current_state.is_white and new_pos < 0:
                        new_pos += 24

                    if current_state.board[new_pos] * opp_sign > 0:
                        # print(f"Destination at {new_pos} is blocked for piece at {pos}")
                        continue

                    # Bearing off logic
                    if (new_pos < 0 or (new_pos < 12 and pos >= 12 and not current_state.is_white)):
                        if can_bear_off:
                            # print(f"Bearing off piece at {pos} to {new_pos}")
                            new_state = current_state.copy()
                            new_state.board[pos] += opp_sign
                            setattr(new_state, off_attr, getattr(new_state, off_attr) + 1)
                            new_state.dice_remaining = new_remaining
                            if not new_remaining:
                                new_state.change_turn()
                                results.append(new_state)
                            else:
                                stack.append(new_state)
                            valid_move_found = True
                        # print(f"Bearing off piece at {pos} is not allowed")
                        continue

                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.board[new_pos] -= opp_sign
                    new_state.dice_remaining = new_remaining

                    if self._is_illegal(new_state):
                        # print(f"Piece at {pos} to {new_pos} is blocking opponent")
                        continue

                    if pos == head_pos:
                        new_state.head_moved = True
                    if not new_remaining:
                        new_state.change_turn()
                        results.append(new_state)
                    else:
                        stack.append(new_state)
                    valid_move_found = True
                    # print(f"Moving piece at {pos} to {new_pos}")
                    # new_state.pretty_print()

                if not valid_move_found:
                    # print("No valid moves found for dice:", dice_value)
                    new_state = current_state.copy()
                    new_state.dice_remaining = new_remaining
                    if not new_remaining:
                        new_state.change_turn()
                        results.append(new_state)
                    else:
                        stack.append(new_state)
                continue  # Skip the original loop after processing grouped dice

            # Original processing for non-identical dice
            for i in reversed(range(len(current_state.dice_remaining))):
                # print("Distinct dice:", current_state.dice_remaining)
                dice = current_state.dice_remaining[i]
                remaining_dice = current_state.dice_remaining[:i] + current_state.dice_remaining[i+1:]
                piece_positions = np.where((current_state.board * opp_sign) < 0)[0]

                if piece_positions.size == 0:
                    # print("No pieces to move")
                    new_state = current_state.copy()
                    new_state.dice_remaining = remaining_dice
                    if not remaining_dice:
                        new_state.change_turn()
                        results.append(new_state)
                    else:
                        stack.append(new_state)
                    continue

                valid_move_found = False

                for pos in piece_positions:
                    if pos == head_pos and current_state.head_moved:
                        # print(f"Head piece at {pos} already moved")
                        continue

                    new_pos = pos - dice
                    if not current_state.is_white and new_pos < 0:
                        new_pos += 24

                    if current_state.board[new_pos] * opp_sign > 0:
                        # print(f"Destination at {new_pos} is blocked for piece at {pos}")
                        continue

                    if (new_pos < 0 or (new_pos < 12 and pos >= 12 and not current_state.is_white)):
                        if can_bear_off:
                            # print(f"Bearing off piece at {pos}")
                            new_state = current_state.copy()
                            new_state.board[pos] += opp_sign
                            setattr(new_state, off_attr, getattr(new_state, off_attr) + 1)
                            new_state.dice_remaining = remaining_dice
                            if not remaining_dice:
                                new_state.change_turn()
                                results.append(new_state)
                            else:
                                stack.append(new_state)
                            valid_move_found = True
                        # print(f"Bearing off piece at {pos} is not allowed")
                        continue

                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.board[new_pos] -= opp_sign
                    new_state.dice_remaining = remaining_dice

                    if self._is_illegal(new_state):
                        # print(f"Piece at {pos} to {new_pos} is blocking opponent")
                        continue

                    if pos == head_pos:
                        new_state.head_moved = True
                    if not remaining_dice:
                        # print("Reached end of dice")
                        new_state.change_turn()
                        results.append(new_state)
                    else:
                        # print("Dice remaining:", remaining_dice)    
                        stack.append(new_state)
                    valid_move_found = True
                    # print(f"Moving piece at {pos} to {new_pos}")
                    # new_state.pretty_print()

                if not valid_move_found:
                    # print("No valid moves found for dice:", dice)
                    new_state = current_state.copy()
                    new_state.dice_remaining = remaining_dice
                    if not remaining_dice:
                        new_state.change_turn()
                        results.append(new_state)
                    else:
                        stack.append(new_state)
        if not results:
            new_state = state.copy()
            new_state.dice_remaining = []
            new_state.change_turn()
            results.append(new_state)

        return results
        
    def get_states_after_dice(self) -> list[State]:
        return self.apply_dice(self.state)

    def step(self, state: State):
        """
        Apply the new state to the game.
        """
        self.state = state
        self.state.roll_dice()

    def is_finished(self):
        """
        The game ends when one player has borne off all 15 pieces.
        """
        return self.state.white_off == 15 or self.state.black_off == 15
    
def board_key(state: State) -> tuple:
    # Convert the board (assumed to be a NumPy array) to a tuple
    return tuple(state.board.tolist())