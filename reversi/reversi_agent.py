"""
This module contains agents that play reversi.

Version 3.1
"""
"""
Members:
6488018 Ramita Deeprom
6488025 Thitiwut Harnphatcharapanukorn
6488079 Burit Sihabut
6488215 Pongsakorn Kongkaewrasamee
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

"""Uses Static Weight as evaluation function"""
class Agent007(ReversiAgent):
    DEPTH_LIMIT = 6

    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        # Check if there are no valid actions to take
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.

        alpha = float("-inf")
        beta = float("inf")
        best_move = valid_actions[0]
        
        # default to first valid action
        output_move_row.value = valid_actions[0][0]
        output_move_column.value = valid_actions[0][1]

        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = self.min_value(new_board, depth=1, alpha=alpha, beta=beta)

            if score > alpha:
                alpha = score
                best_move = action
                output_move_row.value = best_move[0]
                output_move_column.value = best_move[1]
        return score

    def min_value(self, board, depth, alpha, beta):
        opponent = self.player * -1  # Opponent's turn

        if is_terminal(board):
            return self.utility(board) # Return the utility if it's a terminal state

        if depth >= Agent007.DEPTH_LIMIT:
            return self.evaluation(board) # Return the evaluation if depth limit is reached

        valid_actions = actions(board, opponent) # Get valid actions for the opponent

        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1, alpha, beta)  # Skip the turn

        for action in valid_actions:
            new_board = transition(board, opponent, action) # Make a hypothetical opponent's move
            score = self.max_value(new_board, depth + 1, alpha, beta) # Calculate the score

            beta = min(beta, score) # Update beta value

            if beta <= alpha:
                return beta

        return beta

    def max_value(self, board, depth, alpha, beta):
        if is_terminal(board):
            return self.utility(board) # Return the utility if it's a terminal state

        if depth >= Agent007.DEPTH_LIMIT:
            return self.evaluation(board) # Return the evaluation if depth limit is reached

        valid_actions = actions(board, self.player) # Get valid actions for the player

        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1, alpha, beta)  # Skip the turn

        for action in valid_actions:
            new_board = transition(board, self.player, action) # Make a hypothetical player's move
            score = self.min_value(new_board, depth + 1, alpha, beta) # Calculate the score

            alpha = max(alpha, score) # Update alpha value

            if beta <= alpha:
                return beta

        return alpha
    
    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0 # Return utility values based on piece count

    def evaluation(self, board: np.ndarray) -> float:
        """
        Use static weights heuristic values assigned to each position on the board
        Citation: https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=10AB4B0966FEE51BE133255498065C42?doi=10.1.1.580.8400&rep=rep1&type=pdf
        "The weights for the overall champion"
        Page 8, Figure 10
        Evaluation Score = Max Player Value - Min Player Value
        """
        STATIC_WEIGHTS = [ # Static weight matrix for board evaluation
            [4.622507, -1.477853, 1.409644, -0.066975, -0.305214, 1.633019, -1.050899, 4.365550],
            [-1.329145, -2.245663, -1.060633, -0.541089, -0.332716, -0.475830, -2.274535, -0.032595],
            [2.681550, -0.906628, 0.229372, 0.059260, -0.150415, 0.321982, -1.145060, 2.986767],
            [-0.746066, -0.317389, 0.140040, -0.045266, 0.236595, 0.158543, -0.720833, -0.131124],
            [-0.305566, -0.328398, 0.073872, -0.131472, -0.172101, 0.016603, -0.511448, -0.264125],
            [2.777411, -0.769551, 0.676483, 0.282190, 0.007184, 0.269876, -1.408169, 2.396238],
            [-1.566175, -3.049899, -0.637408, -0.077690, -0.648382, -0.911066, -3.329772, -0.870962],
            [5.046583, -1.468806, 1.545046, -0.031175, 0.263998, 2.063148, -0.148002, 5.781035]
        ]
        
        # Evaluate the board using piece count and static weights
        # Initialize player scores
        max_player = 0
        min_player = 0
        
        for i in range(8):
            for j in range(8):
                if board[i, j] == self.player:
                    max_player += STATIC_WEIGHTS[i][j]
                elif board[i, j] == -self.player:
                    min_player += STATIC_WEIGHTS[i][j]
        
        evaluation_score = max_player - min_player # Calculate the evaluation score
        
        return evaluation_score # Return the evaluation score
    
"""Uses heuristic components and static weight table as evaluation function"""
class Agent47(ReversiAgent):
    DEPTH_LIMIT = 10

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
        # default to first valid action
        output_move_row.value = best_move[0]
        output_move_column.value = best_move[1]

        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = self.min_value(new_board, depth=1, alpha=alpha, beta=beta)

            if score > alpha:
                alpha = score
                best_move = action
                output_move_row.value = best_move[0]
                output_move_column.value = best_move[1]

        return score  

    def min_value(self, board, depth, alpha, beta):
        opponent = self.player * -1  # Opponent's turn

        if is_terminal(board):
            return self.utility(board)

        if depth >= Agent47.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, opponent)

        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1, alpha, beta)  # Skip the turn.

        score = float("inf")
        for action in valid_actions:
            new_board = transition(board, opponent, action)
            score = min(score, self.max_value(new_board, depth + 1, alpha, beta))

            if score <= alpha:
                return score

            beta = min(beta, score)

        return score

    def max_value(self, board, depth, alpha, beta):
        if is_terminal(board):
            return self.utility(board)

        if depth >= Agent47.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, self.player)

        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1, alpha, beta)  # Skip the turn.

        score = float("-inf")
        for action in valid_actions:
            new_board = transition(board, self.player, action)
            score = max(score, self.min_value(new_board, depth + 1, alpha, beta))

            if score >= beta:
                return score

            alpha = max(alpha, score)

        return score
    
    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:
        """
        Use Component wise heuristic function
        Source: https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
        Section 5.1.1-5.1.3 Page 5-6
        """
        max_player = self.player
        min_player = -self.player

        max_player_coins = np.count_nonzero(board == max_player)
        min_player_coins = np.count_nonzero(board == min_player)

        max_player_mobility = len(actions(board, max_player))
        min_player_mobility = len(actions(board, min_player))

        max_player_corner = 0
        min_player_corner = 0

        if board[0, 0] == max_player:
            max_player_corner += 1
        if board[0, 7] == max_player:
            max_player_corner += 1
        if board[7, 0] == max_player:
            max_player_corner += 1
        if board[7, 7] == max_player:
            max_player_corner += 1
        if board[0, 0] == min_player:
            min_player_corner += 1
        if board[0, 7] == min_player:
            min_player_corner += 1
        if board[7, 0] == min_player:
            min_player_corner += 1
        if board[7, 7] == min_player:
            min_player_corner += 1
        
        coin_parity = 100 * (max_player_coins - min_player_coins) / (max_player_coins + min_player_coins)

        if max_player_mobility + min_player_mobility != 0:
            mobility = 100 * (max_player_mobility - min_player_mobility) / (max_player_mobility + min_player_mobility)
        else:
            mobility = 0

        if max_player_corner + min_player_corner != 0:
            corner = 100 * (max_player_corner - min_player_corner) / (max_player_corner + min_player_corner)
        else:
            corner = 0

        """
        Use static weights heuristic values assigned to each position on the board
        Source: https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=10AB4B0966FEE51BE133255498065C42?doi=10.1.1.580.8400&rep=rep1&type=pdf
        "The weights for the overall champion"
        Page 8, Figure 10
        Evaluation Score = Max Player Value - Min Player Value
        """
        STATIC_WEIGHTS = [
            [4.622507, -1.477853, 1.409644, -0.066975, -0.305214, 1.633019, -1.050899, 4.365550],
            [-1.329145, -2.245663, -1.060633, -0.541089, -0.332716, -0.475830, -2.274535, -0.032595],
            [2.681550, -0.906628, 0.229372, 0.059260, -0.150415, 0.321982, -1.145060, 2.986767],
            [-0.746066, -0.317389, 0.140040, -0.045266, 0.236595, 0.158543, -0.720833, -0.131124],
            [-0.305566, -0.328398, 0.073872, -0.131472, -0.172101, 0.016603, -0.511448, -0.264125],
            [2.777411, -0.769551, 0.676483, 0.282190, 0.007184, 0.269876, -1.408169, 2.396238],
            [-1.566175, -3.049899, -0.637408, -0.077690, -0.648382, -0.911066, -3.329772, -0.870962],
            [5.046583, -1.468806, 1.545046, -0.031175, 0.263998, 2.063148, -0.148002, 5.781035]
        ]
        
        # Evaluate the board using static weights
        max_player_wscore = 0
        min_player_wscore = 0
        
        for i in range(8):
            for j in range(8):
                if board[i, j] == max_player:
                    max_player_wscore += STATIC_WEIGHTS[i][j]
                elif board[i, j] == min_player:
                    min_player_wscore += STATIC_WEIGHTS[i][j]
        
        weight = max_player_wscore - min_player_wscore
    
        # Combine the individual component scores
        evaluation_score = coin_parity + mobility + corner + weight

        return evaluation_score







        


