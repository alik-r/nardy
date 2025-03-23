import numpy as np
from state import State
from concurrent.futures import ProcessPoolExecutor
from numba import njit
from typing import List, Tuple

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

        # precalculated indices where a white piece can move from each position on the board
        self._precalculated_white = np.array([
            [-1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1],
            [1, 0, -1, -1, -1, -1],
            [2, 1, 0, -1, -1, -1],
            [3, 2, 1, 0, -1, -1],
            [4, 3, 2, 1, 0, -1],
            [5, 4, 3, 2, 1, 0],
            [6, 5, 4, 3, 2, 1],
            [7, 6, 5, 4, 3, 2],
            [8, 7, 6, 5, 4, 3],
            [9, 8, 7, 6, 5, 4],
            [10, 9, 8, 7, 6, 5],
            [11, 10, 9, 8, 7, 6],
            [12, 11, 10, 9, 8, 7],
            [13, 12, 11, 10, 9, 8],
            [14, 13, 12, 11, 10, 9],
            [15, 14, 13, 12, 11, 10],
            [16, 15, 14, 13, 12, 11],
            [17, 16, 15, 14, 13, 12],
            [18, 17, 16, 15, 14, 13],
            [19, 18, 17, 16, 15, 14],
            [20, 19, 18, 17, 16, 15],
            [21, 20, 19, 18, 17, 16],
            [22, 21, 20, 19, 18, 17]
        ], dtype=np.int64)

        self._lengths_white = np.array([
            0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
        ], dtype=np.int64)

        # Precalculated black positions as a NumPy array (padded with -1)
        self._precalculated_black = np.array([
            [23, 22, 21, 20, 19, -1],
            [0, 23, 22, 21, 20, -1],
            [1, 0, 23, 22, 21, -1],
            [2, 1, 0, 23, 22, -1],
            [3, 2, 1, 0, 23, -1],
            [4, 3, 2, 1, 0, -1],
            [5, 4, 3, 2, 1, 0],
            [6, 5, 4, 3, 2, 1],
            [7, 6, 5, 4, 3, 2],
            [8, 7, 6, 5, 4, 3],
            [9, 8, 7, 6, 5, 4],
            [10, 9, 8, 7, 6, 5],
            [-1, -1, -1, -1, -1, -1],
            [12, -1, -1, -1, -1, -1],
            [13, 12, -1, -1, -1, -1],
            [14, 13, 12, -1, -1, -1],
            [15, 14, 13, 12, -1, -1],
            [16, 15, 14, 13, 12, -1],
            [17, 16, 15, 14, 13, 12],
            [18, 17, 16, 15, 14, 13],
            [19, 18, 17, 16, 15, 14],
            [20, 19, 18, 17, 16, 15],
            [21, 20, 19, 18, 17, 16],
            [22, 21, 20, 19, 18, 17]
        ], dtype=np.int64)

        self._lengths_black = np.array([
            5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 
            0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6
        ], dtype=np.int64)

        # Roll dice to begin the first turn.
        self.state.roll_dice()

    # @profile
    def apply_dice(self, state: State) -> List[State]:
        results = []

        # states that are processed already
        seen_configs = set()
        result_configs = set()

        home_slice = slice(0, 6) if state.is_white else slice(12, 18)
        off_attr = 'white_off' if state.is_white else 'black_off'
        head_pos = 23 if state.is_white else 11
        opp_sign = -1 if state.is_white else 1

        if state.board[head_pos] * -opp_sign == 15 and state.dice_remaining in [[3,3,3,3], [4,4,4,4], [6,6,6,6]]:
            dice_value = state.dice_remaining[0]
                    
            if dice_value == 6:
                new_pos1 = head_pos - dice_value
                new_pos2 = head_pos - dice_value 
            
            elif dice_value == 4:
                if not state.is_white and state.board[3] > 0:
                    new_pos1 = head_pos - dice_value
                    new_pos2 = head_pos - dice_value
                else:
                    new_pos1 = head_pos - dice_value * 2
                    new_pos2 = head_pos - dice_value * 2
            
            elif dice_value == 3:
                new_pos1 = head_pos - dice_value * 3
                new_pos2 = head_pos - dice_value 
            
            resulting_State = state.copy()
            resulting_State.board[head_pos] -= 2 * -opp_sign
            resulting_State.board[new_pos1] += 1 * -opp_sign
            resulting_State.board[new_pos2] += 1 * -opp_sign
            resulting_State.change_turn()
            return [resulting_State]

        precalc = self._precalculated_white if state.is_white else self._precalculated_black
        lengths = self._lengths_white if state.is_white else self._lengths_black

        stack = [state.copy()]
        
        while stack:
            current_state = stack.pop()

            current_key = current_state.board.tobytes()
            
            # skip processed states
            if current_key in seen_configs:
                continue
            seen_configs.add(current_key)
            
            # if there is no dice left, terminate the turn
            if not current_state.dice_remaining:
                config = current_state.board.tobytes()
                if config not in result_configs:
                    current_state.change_turn()
                    results.append(current_state)
                    result_configs.add(config)
                continue

            # Check if the player can bear off
            if (current_state.white_off > 0 and current_state.is_white) or (current_state.black_off > 0 and not current_state.is_white):
                can_bear_off = True
            else:
                home_pieces = current_state.board[home_slice]
                if state.is_white:
                    can_bear_off = home_pieces[home_pieces > 0].sum() == 15
                else:
                    can_bear_off = -home_pieces[home_pieces < 0].sum() == 15

            dice_remaining = current_state.dice_remaining

            num_dice = len(dice_remaining)

            # approach for the case when all dice are the same
            if num_dice > 1 and dice_remaining[0] == dice_remaining[1]:
                new_remaining = dice_remaining.copy()
                dice_value = new_remaining.pop(0)
                piece_positions = np.where((current_state.board * opp_sign) < 0)[0]
                valid_move_found = False

                for pos in piece_positions:
                    if pos == head_pos and current_state.head_moved:
                        continue

                    new_pos = pos - dice_value
                    if not current_state.is_white and new_pos < 0:
                        new_pos += 24

                    if current_state.board[new_pos] * opp_sign > 0:
                        continue

                    # Bearing off logic
                    if (new_pos < 0 or (new_pos < 12 and pos >= 12 and not current_state.is_white)):
                        if can_bear_off:
                            new_state = current_state.copy()
                            new_state.board[pos] += opp_sign
                            setattr(new_state, off_attr, getattr(new_state, off_attr) + 1)
                            new_state.dice_remaining = new_remaining
                            if not new_remaining:
                                config = new_state.board.tobytes()
                                if config not in result_configs:
                                    new_state.change_turn()
                                    results.append(new_state)
                                    result_configs.add(config)
                                else:
                                    continue
                            else:
                                stack.append(new_state)
                            valid_move_found = True
                        continue

                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.board[new_pos] -= opp_sign
                    new_state.dice_remaining = new_remaining

                    if _is_illegal(new_state.is_white, new_state.board, precalc, lengths):
                        continue

                    if pos == head_pos:
                        new_state.head_moved = True
                    if not new_remaining:
                        config = new_state.board.tobytes()
                        if config not in result_configs:
                            new_state.change_turn()
                            results.append(new_state)
                            result_configs.add(config)
                        else:
                            continue
                    else:
                        stack.append(new_state)
                    valid_move_found = True

                if not valid_move_found:
                    config = current_state.board.tobytes()
                    if config not in result_configs:
                        new_state = current_state.copy()
                        new_state.dice_remaining = []
                        new_state.change_turn()
                        results.append(new_state)
                        result_configs.add(config)
                continue  # Skip the original loop after processing grouped dice

            # Original processing for non-identical dice
            piece_positions = np.where((current_state.board * opp_sign) < 0)[0]
            for i in range(len(current_state.dice_remaining)):
                dice = current_state.dice_remaining[i]
                remaining_dice = current_state.dice_remaining[:i] + current_state.dice_remaining[i+1:]

                valid_move_found = False

                for pos in piece_positions:
                    if pos == head_pos and current_state.head_moved:
                        continue

                    new_pos = pos - dice
                    if not current_state.is_white and new_pos < 0:
                        new_pos += 24

                    if current_state.board[new_pos] * opp_sign > 0:
                        continue

                    if (new_pos < 0 or (new_pos < 12 and pos >= 12 and not current_state.is_white)):
                        if can_bear_off:
                            new_state = current_state.copy()
                            new_state.board[pos] += opp_sign
                            setattr(new_state, off_attr, getattr(new_state, off_attr) + 1)
                            new_state.dice_remaining = remaining_dice
                            if not remaining_dice:
                                config = new_state.board.tobytes()
                                if config not in result_configs:
                                    new_state.change_turn()
                                    results.append(new_state)
                                    result_configs.add(config)
                                else:
                                    continue
                            else:
                                stack.append(new_state)
                            valid_move_found = True
                        continue

                    new_state = current_state.copy()
                    new_state.board[pos] += opp_sign
                    new_state.board[new_pos] -= opp_sign
                    new_state.dice_remaining = remaining_dice

                    if _is_illegal(new_state.is_white, new_state.board, precalc, lengths):
                        continue

                    if pos == head_pos:
                        new_state.head_moved = True
                    if not remaining_dice:
                        config = new_state.board.tobytes()
                        if config not in result_configs:
                            new_state.change_turn()
                            results.append(new_state)
                            result_configs.add(config)
                        else:
                            continue
                    else:
                        stack.append(new_state)
                    valid_move_found = True

                if not valid_move_found:
                    new_state = current_state.copy()
                    new_state.dice_remaining = remaining_dice
                    if not remaining_dice:
                        config = new_state.board.tobytes()
                        if config not in result_configs:
                            new_state.change_turn()
                            results.append(new_state)
                            result_configs.add(config)
                        else:
                            continue
                    else:
                        stack.append(new_state)
        if not results:
            new_state = state.copy()
            new_state.dice_remaining = []
            new_state.change_turn()
            results.append(new_state)

        return results
        
    def get_states_after_dice(self) -> List[State]:
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

@njit
def are_pieces_after_pos(is_white: bool, board: np.array, pos: int,
                           precalc, lengths) -> bool:
    sign = -1 if is_white else 1
    valid_length = lengths[pos]
    positions = precalc[pos]
    for i in range(valid_length):
        p = positions[i]
        if board[p] * sign > 0:
            return True
    return False


@njit
def _is_illegal(is_white: bool, board: np.array, precalc, lengths) -> bool:
    sign = 1 if is_white else -1

    consecutive_count = 0

    for i in range(0, 24):
        if board[i] * sign > 0:
            consecutive_count += 1
        else:
            consecutive_count = 0
        if consecutive_count > 5 and not are_pieces_after_pos(is_white, board, i, precalc, lengths):
            return True
    return False

