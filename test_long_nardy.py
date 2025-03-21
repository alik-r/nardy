import unittest
import numpy as np
from state import State
from long_nardy import LongNardy

class TestLongNardy(unittest.TestCase):
    def setUp(self):
        self.game = LongNardy()
    
    def test_initial_state(self):
        """Test if the game initializes correctly."""
        self.assertIsInstance(self.game.state, State)
        self.assertIsNotNone(self.game.state.board)
    
    def test_roll_dice(self):
        """Test if dice are rolled correctly at the start."""
        self.assertEqual(len(self.game.state.dice_remaining), 2)
    
    def test_get_states_after_dice(self):
        """Test if applying dice generates possible states."""
        states = self.game.get_states_after_dice()
        self.assertIsInstance(states, list)
        self.assertGreater(len(states), 0)  # Should generate at least one state
        self.assertIsInstance(states[0], State)
    
    def test_step(self):
        """Test if stepping into a new state works correctly."""
        initial_board = self.game.state.board.copy()
        next_states = self.game.get_states_after_dice()
        self.game.step(next_states[0])
        self.assertFalse(np.array_equal(initial_board, self.game.state.board))
    
    def test_is_finished(self):
        """Test if the game correctly detects a finished state."""
        self.game.state.white_off = 15  # Simulate white bearing off all pieces
        self.assertTrue(self.game.is_finished())
        
        self.game.state.white_off = 0
        self.game.state.black_off = 15  # Simulate black bearing off all pieces
        self.assertTrue(self.game.is_finished())
    
if __name__ == "__main__":
    unittest.main()
