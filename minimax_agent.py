from state import State
from long_nardy import LongNardy

def evaluate(game:LongNardy, agent):
    return agent.get_value(game.state).item()


def minimax(game:LongNardy, depth, maximizing_player, agent):
    if game.is_finished() or depth == 0:
        return evaluate(game, agent)
    
    states = game.get_states_after_dice()
    
    if not states:
        game.apply_dice(game.state)
        return minimax(game, depth-1, not maximizing_player, agent)
    
    values = []
    
    for s in states:
        new_game = game.copy()
        new_game.step(s)
        v = minimax(new_game, depth - 1, not maximizing_player, agent)
        values.append(v)
        
    if(maximizing_player):
        return max(values)
    
    return min(values)


def minimax_move(game:LongNardy, agent, depth=2):
    best_value = -float("inf")
    best_state = None 
    
    for s in game.get_states_after_dice():
        new_game = game.copy()
        new_game.step(s)
        value = minimax(new_game, depth-1, False, agent)
        if value > best_value:
            best_value = value
            best_state = s
    
    return best_state 