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
        # Count the number of pieces controlled by the agent and the opponent
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
        
        # Check each corner of the board for agent and opponent control
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
        
        return evaluation_score # Return the evaluation score

class Agent007(ReversiAgent):
    DEPTH_LIMIT = 6 # Maximum depth for the minimax search tree
    # From our experiment, this depth limit gives the best results within a reasonable amount of time
    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        # Check if there are no valid actions to take
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.

        alpha = float("-inf") # Initialize alpha value for alpha-beta pruning
        beta = float("inf") # Initialize beta value for alpha-beta pruning
        best_move = valid_actions[0] # Initialize best move to the first valid action

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
                break # Beta pruning if the beta value is less than or equal to alpha

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
                break # Alpha pruning if the alpha value is greater than or equal to beta

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
    
    
"""Annie's agent"""
class StudentAgent(ReversiAgent):
    DEPTH_LIMIT = 5

    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        alpha = -float('inf')
        beta = float('inf')

        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.

        best_move = valid_actions[0]
        v = -float('inf')

        for action in valid_actions:
            new_v = self.min_value(transition(board, self.player, action), depth=1, alpha=alpha, beta=beta)
            if new_v > v:
                v = new_v
                best_move = action
            alpha = max(alpha, v)
            if v >= beta:
                break  # Beta cutoff

        output_move_row.value = best_move[0]
        output_move_column.value = best_move[1]

    def min_value(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> float:
        opponent = self.player * -1  # opponent's turn
        if is_terminal(board):
            return self.utility(board)
        if depth >= StudentAgent.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, opponent)
        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1, alpha, beta)  # skip the turn.

        v = float('inf')
        for action in valid_actions:
            v = min(v, self.max_value(transition(board, opponent, action), depth + 1, alpha, beta))
            if v <= alpha:
                return v  # Alpha cutoff
            beta = min(beta, v)
        return v

    def max_value(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> float:
        if is_terminal(board):
            return self.utility(board)
        if depth >= StudentAgent.DEPTH_LIMIT:
            return self.evaluation(board)

        valid_actions = actions(board, self.player)
        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1, alpha, beta)  # skip the turn.

        v = -float('inf')
        for action in valid_actions:
            v = max(v, self.min_value(transition(board, self.player, action), depth + 1, alpha, beta))
            if v >= beta:
                return v  # Beta cutoff
            alpha = max(alpha, v)
        return v

    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:

        """
        source of weights table: 
        https://play-othello.appspot.com/files/Othello.pdf
        """
        
        weights = [
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10, 5, 5, 10, -20, 100]
        ]

        player_score = 0
        opponent_score = 0

        for i in range(8):
            for j in range(8):
                if board[i, j] == self.player:
                    player_score += weights[i][j]
                elif board[i, j] == -self.player:
                    opponent_score += weights[i][j]

        return player_score - opponent_score







        


