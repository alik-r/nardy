import unittest
import numpy as np
from long_nardy import LongNardy

class TestDoubleHeadMove(unittest.TestCase):
    def test_double_head_move_4_4_white(self):
        test_board = LongNardy()
        ln = LongNardy()
        ln.state.dice_remaining = [4, 4, 4, 4]
        sts = ln.get_states_after_dice()
        
        self.assertGreaterEqual(len(sts), 1)
        
        test_board.state.board[15] = 2
        test_board.state.board[23] = 13
        ln.step(sts[0])
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board)
    
    def test_double_head_move_3_3_white(self):
        test_board = LongNardy()
        ln = LongNardy()
        ln.state.dice_remaining = [3, 3, 3, 3]
        sts = ln.get_states_after_dice()
        
        self.assertGreaterEqual(len(sts), 1)
        
        test_board.state.board[14] = 1
        test_board.state.board[20] = 1
        test_board.state.board[23] = 13
        ln.step(sts[0])
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board)
    
    def test_double_head_move_6_6_white(self):
        test_board = LongNardy()
        ln = LongNardy()
        ln.state.dice_remaining = [6, 6, 6, 6]
        sts = ln.get_states_after_dice()
        
        self.assertGreaterEqual(len(sts), 1)
        
        test_board.state.board[17] = 2
        test_board.state.board[23] = 13
        ln.step(sts[0])
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board)
    
    def test_double_head_move_3_3_black(self):
        test_board = LongNardy()
        ln = LongNardy()
        test_board.state.is_white = False
        ln.state.is_white = False
        
        ln.state.dice_remaining = [3, 3, 3, 3]
        sts = ln.get_states_after_dice()
        self.assertGreaterEqual(len(sts), 1)
        
        test_board.state.board[8] = -1
        test_board.state.board[2] = -1
        test_board.state.board[11] = -13
        ln.step(sts[0])
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board)
    
    def test_double_head_move_4_4_black(self):
        test_board = LongNardy()
        ln = LongNardy()
        test_board.state.is_white = False
        ln.state.is_white = False
        
        ln.state.dice_remaining = [4, 4, 4, 4]
        sts = ln.get_states_after_dice()
        self.assertGreaterEqual(len(sts), 1)
        
        test_board.state.board[3] = -2
        test_board.state.board[11] = -13
        ln.step(sts[0])
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board) 

    def test_double_head_move_6_6_black(self):
        test_board = LongNardy()
        ln = LongNardy()
        test_board.state.is_white = False
        ln.state.is_white = False
        
        ln.state.dice_remaining = [6, 6, 6, 6]
        sts = ln.get_states_after_dice()
        self.assertGreaterEqual(len(sts), 1)
        
        test_board.state.board[5] = -2
        test_board.state.board[11] = -13
        ln.step(sts[0])
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board)
        
    def test_edge_case1_for_black(self):
        ln = LongNardy()
        test_board = LongNardy()
        
        ln.state.dice_remaining = [5, 5, 5, 5]
        ln.step(ln.get_states_after_dice()[0])
        test_board.step(ln.get_states_after_dice()[0])

        ln.state.dice_remaining = [3, 5]
        states = ln.get_states_after_dice()
        ln.step(states[0])
        self.assertGreaterEqual(len(states), 1)
        
        # double head move for balck since first move, and slot board[3] is busy with white piece
        test_board.state.board[8] = -1
        test_board.state.board[6] = -1
        test_board.state.board[11] = -13
        
        np.testing.assert_array_equal(ln.state.board, test_board.state.board)
        

class TestCapturePrevention(unittest.TestCase):
    def test_no_capture_allowed(self):
        ln = LongNardy()
        
        ln.state.dice_remaining = [5, 5, 5, 5]
        ln.step(ln.get_states_after_dice()[0])

        ln.state.dice_remaining = [3, 5]
        states = ln.get_states_after_dice()
        ln.step(states[0])
        self.assertGreaterEqual(len(states), 1)
        
        for state in states:
            self.assertEqual(state.board[3], 1)  

            

class TestRandomCases(unittest.TestCase):
    def test_real_game_move(self):
        ln = LongNardy()
        test_board = LongNardy()

        ln.state.board[9] = 1
        ln.state.board[12] = 1
        ln.state.board[13] = 1
        ln.state.board[17] = 1
        ln.state.board[19] = 1
        ln.state.board[20] = 1
        ln.state.board[22] = 1
        ln.state.board[23] = 8

        ln.state.board[18] = -1
        ln.state.board[21] = -1
        ln.state.board[10] = -2
        ln.state.board[8] = -1
        ln.state.board[7] = -1
        ln.state.board[6] = -2
        ln.state.board[11] = -7

        for i in range(len(test_board.state.board)):
            test_board.state.board[i] = ln.state.board[i]

        ln.state.dice_remaining = [3, 3, 3, 3]
        states = ln.get_states_after_dice()

        test_board.state.board[12] = 0
        test_board.state.board[14] = 1
        test_board.state.board[9] = 2
        test_board.state.board[23] = 7


        # Check if real game played state is among possible states
        match_found = any(np.array_equal(state.board, test_board.state.board) for state in states)

        self.assertTrue(match_found, "The real game move is NOT among the possible moves.")

    # def test_real_game_move_2(self):
    #     ln = LongNardy()
    #     test_board = LongNardy()

    #     ln.state.board[3] = 2
    #     ln.state.board[12] = 1
    #     ln.state.board[16] = 1
    #     ln.state.board[21] = 5
    #     ln.state.board[23] = 6

    #     ln.state.board[9] = -1
    #     ln.state.board[13] = -1
    #     ln.state.board[19] = -2

    #     for i in range(len(test_board.state.board)):
    #         test_board.state.board[i] = ln.state.board[i]

    #     ln.state.dice_remaining = [2, 3]
    #     states = ln.get_states_after_dice()

    #     test_board.state.board[3] = 0
    #     test_board.state.board[5] = 2
    #     test_board.state.board[21] = 4

    #     match_found = any(np.array_equal(state.board, test_board.state.board) for state in states)
    #     self.assertTrue(match_found, "The real game move is NOT among the possible moves.")

    # def test_real_game_move_3(self):
    #     ln = LongNardy()
    #     test_board = LongNardy()

    #     # Initialize board state before the move
    #     ln.state.board[6] = 1
    #     ln.state.board[10] = 1
    #     ln.state.board[15] = 1
    #     ln.state.board[22] = 3

    #     ln.state.board[8] = -1
    #     ln.state.board[12] = -2
    #     ln.state.board[18] = -1
    #     ln.state.board[23] = -5

    #     # Copy initial state to test_board
    #     for i in range(len(test_board.state.board)):
    #         test_board.state.board[i] = ln.state.board[i]

    #     # Apply dice roll
    #     ln.state.dice_remaining = [3, 6]
    #     states = ln.get_states_after_dice()

    #     # Apply real-game move
    #     test_board.state.board[6] = 0
    #     test_board.state.board[9] = 1
    #     test_board.state.board[15] = 0
    #     test_board.state.board[21] = 1
    #     test_board.state.board[22] = 2

    #     # Check if the played state is among possible states
    #     match_found = any(np.array_equal(state.board, test_board.state.board) for state in states)
    #     self.assertTrue(match_found, "The real game move is NOT among the possible moves.")


class TestEndGame(unittest.TestCase):
    def test_all_pieces_at_last_quarter(self):
        ln = LongNardy()
        
        ln.state.board[0] = 2
        ln.state.board[1] = 3
        ln.state.board[2] = 2
        ln.state.board[3] = 2
        ln.state.board[4] = 3
        ln.state.board[8] = 1
        ln.state.board[10] = 1
        ln.state.board[14] = 1
        ln.state.board[23] = 0

        ln.state.board[12] = -3
        ln.state.board[13] = -4
        ln.state.board[15] = -4
        ln.state.board[16] = -3
        ln.state.board[17] = -1
        ln.state.board[11] = 0
        
            
        ln.state.dice_remaining = [3, 1]
        states = ln.get_states_after_dice()
        
        for state in states:
            for pos in range(0, 6):  # Slots 0-5
                # If a piece moves from here, the test should fail
                self.assertEqual(state.board[pos], ln.state.board[pos], f"Illegal move detected at slot {pos} when other pieces are outside the last quarter.")


    def test_opponent_piece_at_head(self):
        ln = LongNardy()
        
        ln.state.board[0] = 4
        ln.state.board[1] = 3
        ln.state.board[2] = 2
        ln.state.board[3] = 2
        ln.state.board[4] = 3
        ln.state.board[5] = 1
        

        ln.state.board[12] = -3
        ln.state.board[13] = -4
        ln.state.board[15] = -4
        ln.state.board[16] = -3
        ln.state.board[23] = -1 #black piece at white head
        ln.state.board[11] = 0
        
        
        ln.state.dice_remaining = [2, 1]
        states = ln.get_states_after_dice()
        
        for state in states:
            # black is piece is not capured at slot 23
            self.assertEqual(state.board[23], -1, "White wrongly placed a piece at slot 23.")

            initial_white_pieces = sum(ln.state.board[i] for i in range(6))  
            final_white_pieces = sum(state.board[i] for i in range(6)) 

            dif = initial_white_pieces - final_white_pieces
            self.assertGreaterEqual(dif, 0)
            
            diff = np.not_equal(ln.state.board[0:6], state.board[0:6])
            if np.any(diff):  # If any element is different, there is a difference
                self.assertTrue(True)  
            else:
                print("Arrays are the same.")
    
        
    def test_play_larger_dice_at_the_end(self):
        ln = LongNardy()
        
        ln.state.board[0] = 6
        ln.state.board[1] = 6
        ln.state.board[2] = 3
        ln.state.board[23] = 0
        
        ln.state.dice_remaining = [6, 6, 6, 6]
        states = ln.get_states_after_dice()
        self.assertEqual(len(states), 1) #only one possible case: you play 3 pieces from slot board[2], then remaining one from board[1] since it is the next lower
        
        for state in states:
            # self.assertEqual(state.board[23], 4) this assertion is not necessary since pieces may be gathered aside
            self.assertEqual(state.board[2], 0)
            self.assertEqual(state.board[1], 5)
        
        
               
if __name__ == "__main__":
    unittest.main()
