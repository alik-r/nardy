import pygame
import numpy as np
import sys
from long_nardy import LongNardy
from state import State
from human_vs_strong import strong_agent
import torch
import pathlib
from minimax_agent import minimax_move
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = pathlib.Path(__file__).parent / "v2" / "td_gammon_selfplay_535000.pth"
strong_agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
strong_agent.eval()

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Nardy Game (Pygame Interface)")
FONT = pygame.font.SysFont("Arial", 24)
CLOCK = pygame.time.Clock()
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BEIGE = (245, 222, 179)
BLUE = (50, 100, 200)

# Board layout constants
BOARD_TOP = 50
BOARD_BOTTOM = HEIGHT - 50
POINT_WIDTH = 30
POINT_HEIGHT = 200
MID_GAP = 60

selected_point = None
used_dice = []

def draw_board(state: State):
    screen.fill(BEIGE)
    LIGHT_BROWN = (222, 184, 135)
    DARK_BROWN = (139, 69, 19)
    radius = 13

    for i in range(24):
        row = 0 if i < 12 else 1
        index_in_row = i if i < 12 else 23 - i
        x = 50 + index_in_row * (POINT_WIDTH + 5)
        if index_in_row >= 6:
            x += MID_GAP
        y_top = BOARD_TOP
        y_bottom = BOARD_BOTTOM
        triangle_color = LIGHT_BROWN if i % 2 == 0 else DARK_BROWN

        if row == 0:
            pygame.draw.polygon(screen, triangle_color, [
                (x, y_top),
                (x + POINT_WIDTH, y_top),
                (x + POINT_WIDTH // 2, y_top + POINT_HEIGHT)
            ])
        else:
            pygame.draw.polygon(screen, triangle_color, [
                (x, y_bottom),
                (x + POINT_WIDTH, y_bottom),
                (x + POINT_WIDTH // 2, y_bottom - POINT_HEIGHT)
            ])

        if i == selected_point:
            pygame.draw.rect(screen, BLUE, (x, y_top if row == 0 else y_bottom - POINT_HEIGHT, POINT_WIDTH, POINT_HEIGHT), 3)

        checker_count = abs(state.board[i])
        if checker_count == 0:
            continue
        is_white = state.board[i] > 0
        color = WHITE if is_white else BLACK
        spacing = min(2 * radius + 2, (POINT_HEIGHT - 10) // checker_count)

        for j in range(checker_count):
            cx = x + POINT_WIDTH // 2
            cy = y_top + POINT_HEIGHT - (j + 1) * spacing if row == 0 else y_bottom - POINT_HEIGHT + (j + 1) * spacing
            pygame.draw.circle(screen, color, (cx, cy), radius)
            pygame.draw.circle(screen, BLACK, (cx, cy), radius, 1)

def draw_dice(dice_values):
    dice_x = WIDTH // 2 - 50
    dice_y = HEIGHT // 2 - 25
    for i in range(len(dice_values)):
        pygame.draw.rect(screen, WHITE, (dice_x + i * 60, dice_y, 50, 50))
        pygame.draw.rect(screen, BLACK, (dice_x + i * 60, dice_y, 50, 50), 2)
        value_surf = FONT.render(str(dice_values[i]), True, BLACK)
        screen.blit(value_surf, (dice_x + i * 60 + 18, dice_y + 12))

def roll_dice():
    d1, d2 = random.randint(1, 6), random.randint(1, 6)
    return [d1, d2, d1, d2] if d1 == d2 else [d1, d2]

def get_clicked_point(mx, my):
    for i in range(24):
        draw_index = 23 - i
        row = 1 if draw_index < 12 else 0
        index_in_row = draw_index if draw_index < 12 else 23 - draw_index
        x = 50 + index_in_row * (POINT_WIDTH + 5)
        if index_in_row >= 6:
            x += MID_GAP
        y_top = BOARD_TOP
        y_bottom = BOARD_BOTTOM

        if row == 0 and y_top <= my <= y_top + POINT_HEIGHT:
            if x <= mx <= x + POINT_WIDTH:
                return i
        elif row == 1 and y_bottom - POINT_HEIGHT <= my <= y_bottom:
            if x <= mx <= x + POINT_WIDTH:
                return i
    return None

def is_valid_move(state: State, start: int, end: int, move_distance: int):
    if state.board[start] <= 0:
        return False
    if state.board[end] < 0:  # blocked by opponent
        return False
    if move_distance not in state.dice_remaining:
        return False
    return True

def main():
    global selected_point
    game = LongNardy()
    game.state.board = np.zeros(24, dtype=np.int32)
    game.state.board[23] = 15
    game.state.board[11] = -15

    selected_point = None
    dice = roll_dice()
    game.state.dice_remaining = dice
    used_dice.clear()
    player_turn = True

    running = True
    while running:
        CLOCK.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and player_turn:
                mx, my = pygame.mouse.get_pos()
                clicked_point = get_clicked_point(mx, my)

                if clicked_point is not None:
                    if selected_point is None:
                        if game.state.board[clicked_point] > 0:
                            selected_point = clicked_point
                    else:
                        direction = -1
                        move_distance = direction * (clicked_point - selected_point) % 24

                        if selected_point != clicked_point and is_valid_move(game.state, selected_point, clicked_point, move_distance):
                            game.state.board[clicked_point] += 1
                            game.state.board[selected_point] -= 1
                            game.state.dice_remaining.remove(move_distance)
                            used_dice.append(move_distance)
                            selected_point = None

                            if not game.state.dice_remaining:
                                player_turn = False  # AI turn
                        else:
                            selected_point = None

        if not player_turn:
            with torch.no_grad():
                game.state.change_turn()
                dice = roll_dice()
                game.state.dice_remaining = dice.copy()
                print(f"AI rolled dice: {dice}")

                candidates = game.get_states_after_dice()
                if not candidates:
                    print("AI has no more valid moves.")
                    break

                chosen_state = minimax_move(game, strong_agent, depth=2)
                chosen_state.pretty_print()
                # game.step(chosen_state)
                if any(np.array_equal(chosen_state.board, cand.board) for cand in candidates):
                    print(f"AI made a move. Used: {set(dice) - set(chosen_state.dice_remaining)}, Remaining: {chosen_state.dice_remaining}")
                    game.state = chosen_state
                    game.state.is_white = False
                else:
                    print("⚠️ AI made an invalid move! Skipping remaining turn.")
                    break

                if not game.state.dice_remaining:
                    print("AI used all dice.")
                    break


                used_dice.clear()
                player_turn = True
                game.state.change_turn()


        draw_board(game.state)
        draw_dice(game.state.dice_remaining)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
