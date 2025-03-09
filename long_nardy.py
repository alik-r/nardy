import random
import copy

class LongNardy:
    """
    Board representation:
      - The game is played on a common board of 24 points.
      - Each player's route is "unfolded" into 12 positions.
         For White:
            * Indices 0-5: home board (absolute points 1 to 6)
            * Indices 6-11: table (absolute points 19 to 24, with index 11 = head, point 24)
         For Black:
            * Indices 0-5: home board (absolute points 13 to 18)
            * Indices 6-11: table (absolute points 7 to 12, with index 11 = head, point 12)
      - Movement is anticlockwise - a move subtracts the dice pips from the current index.
      
    Locking (prime) rules:
      - A piece is considered locked if the six consecutive absolute points (ahead in the moving direction)
        are all occupied by the opponent.
      - A move that would result in a destination point having six pieces (i.e. forming a block)
        is legal only if at least one opponent checker is in his home.
      - Moreover, it is forbidden to build a block that would lock all 15 of the opponent's checkers.
        
    Reward logic:
      - Every time a piece is borne off, a small reward (0.1) is given.
      - When a game ends, the winner obtains a bonus:
            - 1 point if the opponent has borne off at least one piece (Oyn),
            - 2 points (Mars) if the opponent has borne off none.
      - For simultaneous training, if the current agent is Black the total reward is multiplied by -1.
    """

    def __init__(self):
        # Each player has a 12-slot route.
        self.white = [0] * 12
        self.black = [0] * 12
        # Initial setup: all 15 pieces on the head.
        self.white[11] = 15   # White's head (absolute point 24)
        self.black[11] = 15   # Black's head (absolute point 12)

        # Counters for borne-off pieces.
        self.white_off = 0
        self.black_off = 0

        # 'white' always starts.
        self.current_player = 'white'
        
        # Dice roll for current turn.
        self.dice = []            # full dice roll for the turn
        self.dice_remaining = []  # dice moves remaining this turn
        
        # Flag to enforce that only one move from the head is allowed per turn.
        self.head_moved = False

        # Roll dice to begin the first turn.
        self.roll_dice()

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
        self.white = [0] * 12
        self.black = [0] * 12
        self.white[11] = 15
        self.black[11] = 15
        self.white_off = 0
        self.black_off = 0
        self.current_player = 'white'
        self.roll_dice()

    def _absolute_position(self, player, pos):
        """
        Map a player's board index (0-11) to the common absolute board point (1-24).
        For White:
            0-5   -> points 1 to 6   (home)
            6-11  -> points 19 to 24 (table; index 11 = head = 24)
        For Black:
            0-5   -> points 13 to 18 (home)
            6-11  -> points 7 to 12  (table; index 11 = head = 12)
        """
        if player == 'white':
            if pos < 6:
                return pos + 1
            else:
                return pos - 6 + 19
        else:  # black
            if pos < 6:
                return pos + 13
            else:
                return pos - 6 + 7

    def get_absolute_board(self):
        """
        Build a dictionary for the full board (points 1 to 24).
        Each point is assigned exclusively to one color:
          - Points 1-6: white home.
          - Points 7-12: black table.
          - Points 13-18: black home.
          - Points 19-24: white table.
        Returns a dict mapping point -> (occupant, count)
        """
        abs_board = {}
        # White home: indices 0-5 -> points 1-6.
        for i in range(6):
            abs_board[i+1] = ('white', self.white[i])
        # White table: indices 6-11 -> points 19-24.
        for i in range(6, 12):
            abs_board[i - 6 + 19] = ('white', self.white[i])
        # Black home: indices 0-5 -> points 13-18.
        for i in range(6):
            abs_board[i+13] = ('black', self.black[i])
        # Black table: indices 6-11 -> points 7-12.
        for i in range(6, 12):
            abs_board[i - 6 + 7] = ('black', self.black[i])
        return abs_board

    def _is_locked(self, player, pos):
        """
        Check if the piece at board index pos (for the given player) is locked.
        A piece is locked if, in the six absolute points immediately ahead (in anticlockwise order),
        every such point that exists on the board is occupied by the opponent.
        """
        abs_pos = self._absolute_position(player, pos)
        abs_board = self.get_absolute_board()
        opp = 'black' if player == 'white' else 'white'
        locked = True
        for offset in range(1, 7):
            check_pos = abs_pos - offset
            if check_pos < 1:
                break  # reached borne-off area
            occupant, count = abs_board[check_pos]
            if occupant != opp or count == 0:
                locked = False
                break
        return locked

    def _would_block_all_opponent(self, candidate_move):
        """
        Simulate applying candidate_move and check if it would block (lock) all 15 of the opponent's pieces.
        Returns True if after the move all opponent pieces are locked.
        """
        clone = copy.deepcopy(self)
        current = self.current_player
        opp = 'black' if current == 'white' else 'white'
        board = clone.white if current == 'white' else clone.black
        from_pos, move_distance, _ = candidate_move
        new_pos = from_pos - move_distance
        # Apply move on clone
        board[from_pos] -= 1
        if new_pos >= 0:
            board[new_pos] += 1
        else:
            if current == 'white':
                clone.white_off += 1
            else:
                clone.black_off += 1

        # Count opponent pieces that are locked.
        sim_locked = 0
        opp_board = clone.black if opp == 'black' else clone.white
        for pos in range(12):
            if opp_board[pos] > 0 and clone._is_locked(opp, pos):
                sim_locked += opp_board[pos]
        return sim_locked == 15

    def get_valid_moves(self):
        """
        Generate and return a list of valid moves for the current player given the current dice_remaining.
        
        Each move is a tuple: (from_pos, move_distance, dice_info)
         - from_pos: board index (0-11) from which a piece is moved.
         - move_distance: total pips used.
         - dice_info: either a tuple (d,) for a single-die move or 'combined' if using two dice.
         
        Bearing off moves (destination < 0) are included.
        Moves that violate the locking rules or would illegally form a block (prime) are filtered out.
        """
        valid_moves = []
        board = self.white if self.current_player == 'white' else self.black
        
        # Bearing off is allowed only when all 15 pieces are in home (indices 0-5).
        pieces_in_home = sum(board[0:6])
        can_bear_off = (pieces_in_home == 15)

        # Consider two kinds of dice usage.
        available_dice = self.dice_remaining.copy()
        single_moves = set(available_dice)
        combined_moves = set()
        if len(available_dice) >= 2:
            for i in range(len(available_dice)):
                for j in range(i + 1, len(available_dice)):
                    combined_moves.add(available_dice[i] + available_dice[j])
                    
        # For each board index with pieces.
        for pos in range(12):
            if board[pos] <= 0:
                continue
            # Do not allow moves from a locked piece.
            if self._is_locked(self.current_player, pos):
                continue
            # Enforce one move from the head (index 11) per turn.
            if pos == 11 and self.head_moved:
                continue

            # For each single-die move.
            for d in single_moves:
                new_pos = pos - d
                # Normal move on board.
                if new_pos >= 0:
                    # Check: if the move would form a block (i.e. destination becomes 6), then
                    # at least one opponent piece must be in his home.
                    if board[new_pos] + 1 == 6:
                        opp_home = (self.black[0:6] if self.current_player == 'white' else self.white[0:6])
                        if sum(opp_home) == 0:
                            continue
                        # Also, simulate the move to ensure not all opponent pieces become blocked.
                        candidate = (pos, d, (d,))
                        if self._would_block_all_opponent(candidate):
                            continue
                    valid_moves.append((pos, d, (d,)))
                else:
                    # Bearing off move.
                    if can_bear_off:
                        if pos == d or (d > pos and sum(board[pos+1:6]) == 0):
                            valid_moves.append((pos, d, (d,)))

            # For each combined dice move.
            for d in combined_moves:
                new_pos = pos - d
                if new_pos >= 0:
                    if board[new_pos] + 1 == 6:
                        opp_home = (self.black[0:6] if self.current_player == 'white' else self.white[0:6])
                        if sum(opp_home) == 0:
                            continue
                        candidate = (pos, d, 'combined')
                        if self._would_block_all_opponent(candidate):
                            continue
                    valid_moves.append((pos, d, 'combined'))
                else:
                    if can_bear_off:
                        if pos == d or (d > pos and sum(board[pos+1:6]) == 0):
                            valid_moves.append((pos, d, 'combined'))
        return valid_moves

    def step(self, action):
        """
        Execute the given action and update the game state.
        
        The action is a tuple (from_pos, move_distance, dice_info).
        This method updates the board, removes the dice used,
        and if no moves remain it switches the turn (with a new dice roll).
        
        Returns:
            state: current game state.
            reward: reward from this action.
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

        reward = 0
        done = self.is_finished()
        if done:
            if current == 'white' and self.white_off == 15:
                bonus = 2 if self.black_off == 0 else 1
                reward += bonus
            elif current == 'black' and self.black_off == 15:
                bonus = 2 if self.white_off == 0 else 1
                reward += bonus

        # If no dice remain or no further moves are valid, end the turn.
        if len(self.dice_remaining) == 0 or len(self.get_valid_moves()) == 0:
            self.current_player = 'black' if current == 'white' else 'white'
            self.roll_dice()

        state = self.get_state()
        return state, reward, done

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
            'white_board': self.white.copy(),
            'black_board': self.black.copy(),
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