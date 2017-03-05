"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = 0.
    opponent = game.get_opponent(player)

    def improved_score(game, player):
        # improved_score for testing
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(opponent))
        return float(own_moves - opp_moves)
    #score = score + improved_score(game, player)

    # Heuristic I
    def distance_from_center(game, player):
        # Score depends on the distance of center point.
        row, col = game.get_player_location(player)
        row_middle = math.floor(game.height / 2)
        col_middle = math.floor(game.width / 2)
        row_delta = abs(row - row_middle)
        col_delta = abs(col - col_middle)
        return float(row_middle + col_middle / (row_delta + col_delta + 1))
    #score = score + distance_from_center(game, player)
    #score = score - distance_from_center(game, opponent)

    # Heuristic II
    def distance_from_others(game, player):
        # Distance from other elements
        score = 0.

        blank = game.get_blank_spaces()
        row, col = game.get_player_location(player)
        area_size = 2

        for i in range(-area_size, area_size + 1):
            for j in range(-area_size, area_size + 1):
                delta_row = i + row
                delta_col = j + col
                if 0 <= delta_row < game.height and \
                   0 <= delta_col < game.width and \
                   game.__board_state__[delta_row][delta_col] != game.BLANK:
                    score -= (1 / (i**2 + j**2 + 1))

        return score
    #score = score + distance_from_others(game, player)
    #score = score - distance_from_others(game, opponent)

    # Heuristic III
    def available_spaces_around(game, player):
        # Available spaces around the player
        score = 0.

        blank = game.get_blank_spaces()
        row, col = game.get_player_location(player)
        area_size = 2

        for i in range(-area_size, area_size + 1):
            for j in range(-area_size, area_size + 1):
                delta_row = i + row
                delta_col = j + col
                if 0 <= delta_row < game.height and \
                   0 <= delta_col < game.width and \
                   game.__board_state__[delta_row][delta_col] == game.BLANK:
                    score += 1.

        return score
    #score = score + available_spaces_around(game, player)
    #score = score - available_spaces_around(game, opponent)

    # Heuristic IV
    def away_from_opponent(game, player):
        row_player, col_player = game.get_player_location(player)
        row_opponent, col_opponent = game.get_player_location(opponent)
        return abs(row_player - row_opponent) + abs(col_player - col_opponent)
    #score = score + away_from_opponent(game, player)

    # Heuristic V
    # Checking requisively the available future steps (without making steps with opponent).
    def future_steps(new_board, player, depth):

        legal_moves = new_board.get_legal_moves(player)

        score = depth * len(legal_moves)

        if depth == 1:
            return score

        for m in legal_moves:
            row, col = m
            new_board.__board_state__[row][col] = new_board.__player_symbols__[player]
            score += future_steps(new_board, player, depth - 1)

        return score
    # Copy board, not to change it.
    new_board = game.copy()
    depth = 6
    score = score + future_steps(new_board, player, depth)

    return score

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        # If it is on the beginning of game, occupy the middle point.
        if game.move_count <= 1:

            row = math.floor(game.height / 2)
            col = math.floor(game.width / 2)
            move = (row, col)
            if game.move_is_legal(move):
                return move
            else:
                # Get neighbour place. There are 8 neighbour spaces.
                delta = randint(0, 7)
                if delta <= 2:
                    row -= 1
                elif delta >= 5:
                    row += 1
                if delta == 0 or delta == 3 or delta == 5:
                    col -= 1
                elif delta == 2 or delta == 4 or delta == 7:
                    col += 1
                move = (row, col)
                # Just to be sure.
                if game.move_is_legal(move):
                    return move

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            move = (-1, -1)
            max_depth = 10

            if not legal_moves:
                return move

            def get_simple_move(depth):
                if self.method == "minimax":
                    #_, move = max([(self.minimax(game.forecast_move(m), depth, False), m) for m in legal_moves])
                    _, move = self.minimax(game, depth, False)
                elif self.method == "alphabeta":
                    _, move = self.alphabeta(game, depth)

                return move

            if self.iterative:
                for d in range(1, max_depth):
                    move = get_simple_move(d)
            else:
                move = get_simple_move(self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player = game.active_player
        legal_moves = game.get_legal_moves(player)
        move = (-1, -1)

        if maximizing_player:
            value = float("-inf")
        else:
            value = float("inf")

        if not legal_moves:
            return [value, move]

        if depth <= 0:
            new_value = self.score(game, self)
            return [new_value, (-1, -1)]

        new_depth = depth-1
        new_maximizing_player = not maximizing_player

        for m in legal_moves:
            new_value, _ = self.minimax(game.forecast_move(m), new_depth, new_maximizing_player)

            if (maximizing_player and new_value > value) or ((not maximizing_player) and new_value < value):
                value = new_value
                move = m

        return (value, move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player = game.active_player
        legal_moves = game.get_legal_moves(player)
        move = (-1, -1)

        if maximizing_player:
            value = float("-inf")
        else:
            value = float("inf")

        if not legal_moves:
            return [value, move]

        if depth <= 0:
            new_value = self.score(game, self)
            return [new_value, (-1, -1)]

        new_depth = depth-1
        new_maximizing_player = not maximizing_player

        for m in legal_moves:
            new_value, _ = self.alphabeta(game.forecast_move(m), new_depth, alpha, beta, new_maximizing_player)

            if maximizing_player:
                if new_value > value:
                    value = new_value
                    move = m
                if value >= beta:
                    return (value, move)
                alpha = max(alpha, value)
            else:
                if new_value < value:
                    value = new_value
                    move = m
                if value <= alpha:
                    return (value, move)
                beta = min(beta, value)

        return (value, move)
