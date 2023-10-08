import copy
import itertools
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np

import pygame
import pygame.gfxdraw

HERE = Path(__file__).parent.resolve()


##################### POTENTIALLY USEFUL FUNCTIONS ############################



def is_terminal(board: np.ndarray) -> bool:
    """Can no more moves be played on the board?"""
    return np.sum(board != 0) == board.shape[0] ** 2 or (
        not has_legal_move(board, 1) and not has_legal_move(board, -1)
    )


def play_othello_game(
    your_choose_move: Callable[[np.ndarray], Optional[Tuple[int, int]]],
    opponent_choose_move: Callable[[np.ndarray], Optional[Tuple[int, int]]],
    game_speed_multiplier: float = 1.0,
    render: bool = False,
    verbose: bool = False,
) -> int:
    """Play a game where moves are chosen by `your_choose_move()` and `opponent_choose_move()`. Who
    goes first is chosen at random. You can render the game by setting `render=True`.

    Args:
        your_choose_move: function that chooses move (takes state as input)
        opponent_choose_move: function that picks your opponent's next move
        game_speed_multiplier: multiplies the speed of the game. High == fast
        render: whether to render the game using pygame or not
        verbose: whether to print board states to console. Useful for debugging

    Returns: total_return, which is the sum of return from the game
    """
    total_return = 0
    game = OthelloEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )
    state, reward, done, _ = game.reset()

    while not done:
        action = your_choose_move(state)
        state, reward, done, _ = game.step(action)
        total_return += reward

    return total_return


####### THESE FUNCTIONS ARE LESS USEFUL ############


def make_move(board: np.ndarray, move: Tuple[int, int], current_player: int) -> np.ndarray:
    """Returns board after move has been made."""
    if is_legal_move(board, move, current_player):
        board_after_move = copy.deepcopy(board)
        board_after_move[move] = current_player
        board_after_move = flip_tiles(board_after_move, move, current_player)
    else:
        raise ValueError(f"Move {move} is not a valid move!")
    return board_after_move


def is_legal_move(board: np.ndarray, move: Tuple[int, int], current_player: int) -> bool:
    board_dim = board.shape[0]
    if is_valid_coord(board_dim, move[0], move[1]) and board[move] == 0:
        for direction in MOVE_DIRS:
            if has_tile_to_flip(board, move, direction, current_player):
                return True
    return False


def is_valid_coord(board_dim: int, row: int, col: int) -> bool:
    """Is the coord (row, col) in the board."""
    return 0 <= row < board_dim and 0 <= col < board_dim


def has_tile_to_flip(
    board: np.ndarray,
    move: Tuple[int, int],
    direction: Tuple[int, int],
    current_player: int,
) -> bool:

    """Checks whether the current_player has any adversary's tile to flip with the move they make
    (in a specific direction)."""

    board_dim = board.shape[0]
    i = 1
    while True:
        row = move[0] + direction[0] * i
        col = move[1] + direction[1] * i
        if not is_valid_coord(board_dim, row, col) or board[row, col] == 0:
            return False
        elif board[row, col] == current_player:
            break
        else:
            i += 1

    return i > 1


def flip_tiles(board: np.ndarray, move: Tuple[int, int], current_player: int) -> np.ndarray:
    """Flips the adversary's tiles for current move and updates the running tile count for each
    player.

    Arg:
        move (Tuple[int, int]): The move just made to
                                trigger the flips
    """
    for direction in MOVE_DIRS:
        if has_tile_to_flip(board, move, direction, current_player):
            i = 1
            while True:
                row = move[0] + direction[0] * i
                col = move[1] + direction[1] * i
                if board[row][col] == current_player:
                    break
                board[row][col] = current_player
                i += 1
    return board


def has_legal_move(board: np.ndarray, current_player: int) -> bool:
    """True if current_player has legal move on the board."""

    board_dim = len(board)
    for row, col in itertools.product(range(board_dim), range(board_dim)):
        move = (row, col)
        if is_legal_move(board, move, current_player):
            return True
    return False


def get_legal_moves(board: np.ndarray, current_player: int) -> List[Tuple[int, int]]:
    """Return a list of legal moves that can be made by player 1 on the current board."""

    moves = []
    board_dim = len(board)
    for row, col in itertools.product(range(board_dim), range(board_dim)):
        move = (row, col)
        if is_legal_move(board, move, current_player):
            moves.append(move)
    return moves


def get_empty_board(board_dim: int = 6, player_start: int = 1) -> np.ndarray:
    board = np.zeros((board_dim, board_dim))
    if board_dim < 2:
        return board
    coord1 = int(board_dim / 2 - 1)
    coord2 = board_dim // 2
    initial_squares = [
        (coord1, coord2),
        (coord1, coord1),
        (coord2, coord1),
        (coord2, coord2),
    ]

    for i in range(len(initial_squares)):
        row = initial_squares[i][0]
        col = initial_squares[i][1]
        player = player_start if i % 2 == 0 else player_start * -1
        board[row, col] = player

    return board


# Directions relative to current counter (0, 0) that a tile to flip can be
MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0), (+1, +1)]

# Graphics constants
BACKGROUND_COLOR = (0, 158, 47)
BLACK_COLOR = (6, 9, 16)
WHITE_COLOR = (255, 255, 255)
GREY_COLOR = (201, 199, 191)
SQUARE_SIZE = 100
DISC_SIZE = int(SQUARE_SIZE * 0.4)


class OthelloEnv:
    def __init__(
        self,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
        board_dim: int = 8,
    ):
        self._board_visualizer = np.vectorize(lambda x: "X" if x == 1 else "O" if x == -1 else "*")
        self.verbose = verbose
        self.render = render
        self.game_speed_multiplier = game_speed_multiplier
        self.board_dim = board_dim
        if self.render:
            self.init_graphics()

    def init_graphics(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.board_dim * SQUARE_SIZE, self.board_dim * SQUARE_SIZE)
        )
        pygame.display.set_caption("Othello")
        self.screen.fill(WHITE_COLOR)

    def reset(self) -> Tuple[np.ndarray, int, bool, Dict]:
        """Resets game & takes 1st opponent move if they are chosen to go first."""
        self._player = random.choice([-1, 1])
        self._board = get_empty_board(self.board_dim, self._player)
        self.running_tile_count = 4 if self.board_dim > 2 else 0
        self.done = False
        self.winner = None
        if self.verbose:
            print(f"Starting game. Player {self._player} has first move\n", self)

        reward = 0

        if self.render:
            draw_game(self.screen, self._board, get_legal_moves(self._board, self._player))
            time.sleep(1 / self.game_speed_multiplier)

        return self._board, reward, self.done, self.info

    def __repr__(self) -> str:
        return str(self._board_visualizer(self._board)) + "\n"

    @property
    def info(self) -> Dict[str, Any]:
        return {"player_to_take_next_move": self._player, "winner": self.winner}

    @property
    def tile_count(self) -> Dict:
        return {1: np.sum(self._board == 1), -1: np.sum(self._board == -1)}

    def switch_player(self) -> None:
        """Change self.player only when game isn't over."""
        self._player *= 1 if self.done else -1

    def _step(self, move: Optional[Tuple[int, int]]) -> int:
        """Takes 1 turn, internal to this class.

        Do not call
        """
        assert not self.done, "Game is over, call .reset() to start a new game"

        if move is None:
            assert not has_legal_move(
                self._board, self._player
            ), "Your move is None, but you must make a move when a legal move is available!"

            if self.verbose:
                print(f"Player {self._player} has no legal move, switching player")
            self.switch_player()
            if self.render:
                draw_game(self.screen, self._board, get_legal_moves(self._board, self._player))
                time.sleep(1 / self.game_speed_multiplier)
            return 0

        assert is_legal_move(self._board, move, self._player), f"Move {move} is not valid!"

        self.running_tile_count += 1
        self._board = make_move(self._board, move, self._player)

        # Check for game completion
        tile_difference = self.tile_count[self._player] - self.tile_count[self._player * -1]
        self.done = self.game_over
        self.winner = (
            None
            if self.tile_count[1] == self.tile_count[-1] or not self.done
            # mypy sad, probably bug: github.com/python/mypy/issues/9765
            else max(self.tile_count, key=self.tile_count.get)  # type: ignore
        )
        won = self.done and tile_difference > 0

        if self.verbose:
            print(f"Player {self._player} places counter at row {move[0]}, column {move[1]}")
            print(self)
            if self.done:
                if won:
                    print(f"Player {self._player} has won!\n")
                elif self.running_tile_count == self.board_dim**2 and tile_difference == 0:
                    print("Board full. It's a tie!")
                else:
                    print(f"Player {self._player * -1} has won!\n")

        self.switch_player()

        if self.render:
            draw_game(self.screen, self._board, get_legal_moves(self._board, self._player))
            time.sleep(1 / self.game_speed_multiplier)

        return 1 if won else 0

    def step(self, move: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, int, bool, Dict[str, int]]:
        """Called by user - takes 2 turns, yours and your opponent's"""

        reward = self._step(move)

        if self.done:
            if np.sum(self._board == 1) > np.sum(self._board == -1):
                reward = 1
            elif np.sum(self._board == 1) < np.sum(self._board == -1):
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        return self._board, reward, self.done, self.info

    @property
    def game_over(self) -> bool:
        return (
            not has_legal_move(self._board, self._player)
            and not has_legal_move(self._board, self._player * -1)
            or self.running_tile_count == self.board_dim**2
        )


def draw_game(
    screen: pygame.surface.Surface, board: np.ndarray, legal_moves: List[Tuple[int, int]]
) -> None:

    origin = (0, 0)
    n_rows = board.shape[0]
    n_cols = board.shape[1]

    # Draw background of the board
    pygame.gfxdraw.box(
        screen,
        pygame.Rect(
            origin[0],
            origin[1],
            board.shape[0] * SQUARE_SIZE,
            board.shape[1] * SQUARE_SIZE,
        ),
        BACKGROUND_COLOR,
    )

    # Draw the lines on the board
    size = SQUARE_SIZE

    for x, y in itertools.product(range(n_rows), range(n_cols)):
        pygame.gfxdraw.rectangle(
            screen,
            (origin[0] + x * size, origin[1] + y * size, size, size),
            BLACK_COLOR,
        )

    # Draw the in play tiles
    for r, c in itertools.product(range(n_rows), range(n_cols)):
        space = board[r, c]

        color = BLACK_COLOR if space == 1 else WHITE_COLOR if space == -1 else BACKGROUND_COLOR
        outline_color = BACKGROUND_COLOR if (r, c) not in legal_moves else GREY_COLOR

        draw_counter(
            screen,
            origin[0] + c * SQUARE_SIZE + SQUARE_SIZE // 2,
            origin[1] + r * SQUARE_SIZE + SQUARE_SIZE // 2,
            DISC_SIZE,
            fill_color=color,
            outline_color=outline_color,
        )

    pygame.display.update()


def draw_counter(
    screen: pygame.surface.Surface,
    x_pos: int,
    y_pos: int,
    size: int,
    outline_color: Tuple[int, int, int],
    fill_color: Optional[Tuple[int, int, int]],
) -> None:

    if fill_color is not None:
        pygame.gfxdraw.filled_circle(
            screen,
            x_pos,
            y_pos,
            size,
            fill_color,
        )

    pygame.gfxdraw.aacircle(
        screen,
        x_pos,
        y_pos,
        size,
        outline_color,
    )


def pos_to_coord(pos: Tuple[int, int]) -> Tuple[int, int]:  # Assume square board
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    return row, col


LEFT = 1


# def human_player(state: np.ndarray, *args: Any, **kwargs: Any) -> Optional[Tuple[int, int]]:

#     print("Your move, click to place a tile!")

#     legal_moves = get_legal_moves(state)
#     if not legal_moves:
#         return None

#     while True:
#         ev = pygame.event.get()
#         for event in ev:
#             if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
#                 coord = pos_to_coord(pygame.mouse.get_pos())
#                 if coord in legal_moves:
#                     return coord
