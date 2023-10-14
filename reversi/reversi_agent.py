"""
This module contains agents that play reversi.

Version 3.1
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value
from typing import Tuple, List

import numpy as np

from game import is_terminal, make_move, get_legal_moves

# ============================================================
# Formulation (DO NOT CHANGE)
# ============================================================

def actions(board: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """Return valid actions."""
    return get_legal_moves(board, player)


def transition(board: np.ndarray, player: int, action: Tuple[int, int]) -> np.ndarray:
    """Return a new board if the action is valid, otherwise None."""

    new_board = make_move(board, action, player)
    return new_board


def terminal_test(board: np.ndarray) -> bool:
    return is_terminal(board)

# ============================================================
# Agent Template (DO NOT CHANGE)
# ============================================================

class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self) -> int:
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self) -> Tuple[int, int]:
        """Return move that skips the turn."""
        return (-1, 0)

    @property
    def best_move(self) -> Tuple[int, int]:
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions) -> Tuple[int, int]:
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)
            p = Process(
                target=self.search,
                args=(
                    board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
                self._move = (int(output_move_row.value), int(output_move_column.value))
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = (int(output_move_row.value), int(output_move_column.value))
        return self.best_move

    @abc.abstractmethod
    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for
            `output_move_row.value` and `output_move_column.value`
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


# ============================================================
# Random Agent (DO NOT CHANGE)
# ============================================================


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            # time.sleep(0.1)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)




class NorAgent(ReversiAgent):
    """A minimax agent."""
    DEPTH_LIMIT = 5

    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # valid_actions = actions(board, self.player)
        # print(valid_actions)
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.
        v = -999999
        # default to first valid action
        output_move_row.value = valid_actions[0][0]
        output_move_column.value = valid_actions[0][1]
        for action in valid_actions:
            new_v = self.min_value(transition(board, self.player, action), depth=1)
            if new_v > v:
                v = new_v
                output_move_row.value = action[0]
                output_move_column.value = action[1]
        return v


    def min_value(self, board: np.ndarray, depth: int) -> float:
        opponent = self.player * -1 # opponent's turn
        if is_terminal(board):
            return self.utility(board)
        if depth >= NorAgent.DEPTH_LIMIT:
            return self.evaluation(board)
        valid_actions = actions(board, opponent)
        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1)  # skip the turn.
        v = 999999
        for action in valid_actions:
            v = min(v, self.max_value(transition(board, opponent, action), depth+1))
        return v

    def max_value(self, board: np.ndarray, depth: int) -> float:
        if is_terminal(board):
            return self.utility(board)
        if depth >= NorAgent.DEPTH_LIMIT:
            return self.evaluation(board)
        valid_actions = actions(board, self.player)
        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1)  # skip the turn.
        v = -999999
        for action in valid_actions:
            v = min(v, self.min_value(transition(board, self.player, action), depth+1))
        return v

    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:
        # a dummy evaluation that return diff scores
        return (board == self.player).sum() - (board == (self.player * -1)).sum()


# TODO: Create your own agent

class NorAgent_Killer(ReversiAgent):
    """def search(self, board, valid_actions, output_move_row, output_move_column):
        return super().search(board, valid_actions, output_move_row, output_move_column)"""
    
    DEPTH_LIMIT = 6

    """def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        # valid_actions = actions(board, self.player)
        # print(valid_actions)
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.
        v = -999999
        # default to first valid action
        output_move_row.value = valid_actions[0][0]
        output_move_column.value = valid_actions[0][1]
        for action in valid_actions:
            new_v = self.min_value(transition(board, self.player, action), depth=1)
            if new_v > v:
                v = new_v
                output_move_row.value = action[0]
                output_move_column.value = action[1]
        return v"""
    
    def search(self, board, valid_actions, output_move_row, output_move_column, max_time):
        start_time = time.time()
        depth = 1
        best_move = valid_actions[0]
    
        while time.time() - start_time < max_time and depth <= self.DEPTH_LIMIT:
            v, move = self.max_value(board, depth)
            if v > best_move[0]:
                best_move = (v, move)
            depth += 1
    
        output_move_row.value = best_move[1][0]
        output_move_column.value = best_move[1][1]



    def min_value(self, board: np.ndarray, depth: int) -> float:
        opponent = self.player * -1  # opponent's turn
        if is_terminal(board) or depth >= NorAgent.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, opponent)
        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1)  # skip the turn.
    
        v = float('inf')  # Initialize v to positive infinity
        for action in valid_actions:
            v = min(v, self.max_value(transition(board, opponent, action), depth + 1))
    
        return v

    def max_value(self, board: np.ndarray, depth: int) -> float:
        if is_terminal(board) or depth >= NorAgent.DEPTH_LIMIT:
            return self.evaluation(board)
    
        valid_actions = actions(board, self.player)
        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1)  # skip the turn.
    
        v = float('-inf')  # Initialize v to negative infinity
        for action in valid_actions:
            v = max(v, self.min_value(transition(board, self.player, action), depth + 1))
    
        return v

    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:
        # Initialize weights for different features
        piece_count_weight = 1.0
        mobility_weight = 0.5
        positional_weight = 0.2

        player_pieces = (board == self.player).sum()
        opponent_pieces = (board == -self.player).sum()
        player_mobility = len(actions(board, self.player))
        opponent_mobility = len(actions(board, -self.player))

        # Calculate the evaluation score based on weighted features
        score = (
            piece_count_weight * (player_pieces - opponent_pieces) +
            mobility_weight * (player_mobility - opponent_mobility)
        )

        return score
    
class BossAgent(ReversiAgent):
    DEPTH_LIMIT = 6
    # From our experiment, this depth limit gives the best results within a reasonable amount of time
    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.

        alpha = float("-inf")
        beta = float("inf")
        best_move = valid_actions[0]

        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = self.min_value(new_board, depth=1, alpha=alpha, beta=beta)

            if score > alpha:
                alpha = score
                best_move = action

        # Check if best_move is not None before updating the shared values
        if best_move is not None:
            output_move_row.value = best_move[0]
            output_move_column.value = best_move[1]

    def min_value(self, board, depth, alpha, beta):
        opponent = self.player * -1  # Opponent's turn

        if is_terminal(board):
            return self.utility(board)

        if depth >= BossAgent.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, opponent)

        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1, alpha, beta)  # Skip the turn.

        for action in valid_actions:
            new_board = transition(board, opponent, action)
            score = self.max_value(new_board, depth + 1, alpha, beta)

            beta = min(beta, score)

            if beta <= alpha:
                break

        return beta

    def max_value(self, board, depth, alpha, beta):
        if is_terminal(board):
            return self.utility(board)

        if depth >= BossAgent.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, self.player)

        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1, alpha, beta)  # Skip the turn.

        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = self.min_value(new_board, depth + 1, alpha, beta)

            alpha = max(alpha, score)

            if beta <= alpha:
                break

        return alpha
    
    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:
        agent_pieces = np.count_nonzero(board == self.player)
        opponent_pieces = np.count_nonzero(board == -self.player)
        
        # Weights for piece count and corner control
        piece_count_weight = 1.0
        corner_weight = 10.0
        
        """
        Count the corners controlled by the agent and the opponent.
        Agent will try to avoid actions that lead opponent to get the corner
        Citation: https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
        """
        
        agent_corner_count = 0
        opponent_corner_count = 0
        
        if board[0, 0] == self.player:
            agent_corner_count += 1
        if board[0, 7] == self.player:
            agent_corner_count += 1
        if board[7, 0] == self.player:
            agent_corner_count += 1
        if board[7, 7] == self.player:
            agent_corner_count += 1
        
        if board[0, 0] == -self.player:
            opponent_corner_count += 1
        if board[0, 7] == -self.player:
            opponent_corner_count += 1
        if board[7, 0] == -self.player:
            opponent_corner_count += 1
        if board[7, 7] == -self.player:
            opponent_corner_count += 1
        
        # Evaluate the board using piece count and corner control
        evaluation_score = (agent_pieces - opponent_pieces) * piece_count_weight 
        + (agent_corner_count - opponent_corner_count) * corner_weight
        
        return evaluation_score
    
class Agent007(ReversiAgent):
    DEPTH_LIMIT = 6
    # From our experiment, this depth limit gives the best results within a reasonable amount of time
    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.

        alpha = float("-inf")
        beta = float("inf")
        best_move = valid_actions[0]

        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = self.min_value(new_board, depth=1, alpha=alpha, beta=beta)

            if score > alpha:
                alpha = score
                best_move = action

        # Check if best_move is not None before updating the shared values
        if best_move is not None:
            output_move_row.value = best_move[0]
            output_move_column.value = best_move[1]

    def min_value(self, board, depth, alpha, beta):
        opponent = self.player * -1  # Opponent's turn

        if is_terminal(board):
            return self.utility(board)

        if depth >= Agent007.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, opponent)

        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1, alpha, beta)  # Skip the turn.

        for action in valid_actions:
            new_board = transition(board, opponent, action)
            score = self.max_value(new_board, depth + 1, alpha, beta)

            beta = min(beta, score)

            if beta <= alpha:
                break

        return beta

    def max_value(self, board, depth, alpha, beta):
        if is_terminal(board):
            return self.utility(board)

        if depth >= Agent007.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, self.player)

        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1, alpha, beta)  # Skip the turn.

        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = self.min_value(new_board, depth + 1, alpha, beta)

            alpha = max(alpha, score)

            if beta <= alpha:
                break

        return alpha
    
    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:
        """
        Use static weights heuristic values assigned to each position on the board
        Citation: https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
        """
        STATIC_WEIGHTS = [
            [4, -3, 2, 2, 2, 2, -3, 4],
            [-3, -4, -1, -1, -1, -1, -4, -3],
            [2, -1, 1, 0, 0, 1, -1, 2],
            [2, -1, 0, 1, 1, 0, -1, 2],
            [2, -1, 0, 1, 1, 0, -1, 2],
            [2, -1, 1, 0, 0, 1, -1, 2],
            [-3, -4, -1, -1, -1, -1, -4, -3],
            [4, -3, 2, 2, 2, 2, -3, 4]
        ]
        agent_pieces = np.count_nonzero(board == self.player)
        opponent_pieces = np.count_nonzero(board == -self.player)
        
        # Weight for piece count
        piece_count_weight = 1.0
        
        # Evaluate the board using piece count and static weights
        evaluation_score = 0
        
        for i in range(8):
            for j in range(8):
                if board[i, j] == self.player:
                    evaluation_score += STATIC_WEIGHTS[i][j]
                elif board[i, j] == -self.player:
                    evaluation_score -= STATIC_WEIGHTS[i][j]
        
        evaluation_score += (agent_pieces - opponent_pieces) * piece_count_weight
        
        return evaluation_score










        


