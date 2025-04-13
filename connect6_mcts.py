import math
import random
import sys
import numpy as np
import time


class MCTSNode:
    def __init__(self, state, player_turn, parent=None, move=None, game_logic=None):
        self.state = state  # The board state (numpy array)
        self.parent = parent
        self.move = move  # The move (r, c) that led TO this state
        self.player_turn = player_turn  # Whose turn it is IN this state

        self.children = []
        self.wins = 0  # Wins from the perspective of the player who JUST MOVED to reach this state
        self.visits = 0

        # Need access to game logic (passed in or accessed via parent)
        self._game_logic = game_logic if game_logic else parent._game_logic

        # Determine untried moves based on the state
        self.untried_moves = self._game_logic.get_legal_moves_on_board(self.state)
        random.shuffle(self.untried_moves)  # Explore in random order

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        # Check if the state represents a finished game
        winner = self._game_logic.check_win_on_board(self.state)
        return winner != 0 or self._game_logic.is_board_full(self.state)

    def select_best_child(
        self, C_param=1.414
    ):  # sqrt(2) is a common exploration constant
        """Selects the best child node using the UCB1 formula."""
        if not self.children:
            return None

        log_parent_visits = math.log(self.visits)

        def ucb1_score(child):
            if child.visits == 0:
                return float("inf")  # Prioritize unvisited children
            exploitation_term = child.wins / child.visits
            exploration_term = C_param * math.sqrt(log_parent_visits / child.visits)
            return exploitation_term + exploration_term

        scores = [ucb1_score(child) for child in self.children]

        max_score = -1.0
        best_children = []
        for i, score in enumerate(scores):
            # Handle potential floating point inaccuracies if needed
            if score > max_score:
                max_score = score
                best_children = [self.children[i]]
            elif score == max_score:
                best_children.append(self.children[i])

        return random.choice(best_children)  # Break ties randomly

    def expand(self):
        """Expands the node by adding one child for an untried move."""
        if not self.untried_moves:
            # Should not happen if called correctly after checking is_fully_expanded
            raise RuntimeError("Cannot expand a fully expanded node.")

        move = self.untried_moves.pop()  # Take one untried move
        r, c = move

        # Create the next state by applying the move
        next_state = self.state.copy()
        # The player making the move is the one whose turn it is *at this node*
        player_making_move = self.player_turn
        next_state[r, c] = player_making_move

        # --- Determine whose turn it is in the *new* child state ---
        # This requires careful handling of Connect 6 rules
        stones_played_in_turn = self._game_logic.count_stones_played_this_logical_turn(
            next_state, player_making_move
        )
        stones_required_this_turn = self._game_logic.get_stones_required_this_turn(
            next_state, player_making_move
        )

        next_player_turn = player_making_move  # Assume player continues...
        if stones_played_in_turn >= stones_required_this_turn:
            next_player_turn = 3 - player_making_move  # ...unless turn ends

        # Create and add the new child node
        child_node = MCTSNode(
            state=next_state,
            player_turn=next_player_turn,  # Whose turn in child state
            parent=self,
            move=move,
            game_logic=self._game_logic,
        )
        self.children.append(child_node)
        return child_node

    def update(self, simulation_result):
        """Updates the node's visits and wins based on the simulation outcome."""
        self.visits += 1
        # The player who moved *into* this state is (3 - self.player_turn)
        player_who_moved_here = 3 - self.player_turn
        if simulation_result == player_who_moved_here:
            self.wins += 1
        # Handle draws? Could add 0.5 wins for a draw? Optional.
        # elif simulation_result == 0: # Draw
        #     self.wins += 0.5


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.last_opponent_move = None

    def reset_board(self):
        """Clears the board and resets the game state."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets a new board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if (
                            0 <= prev_r < self.size
                            and 0 <= prev_c < self.size
                            and self.board[prev_r, prev_c] == current_color
                        ):
                            continue
                        count = 0
                        rr, cc = r, c
                        while (
                            0 <= rr < self.size
                            and 0 <= cc < self.size
                            and self.board[rr, cc] == current_color
                        ):
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts a column index to a letter (skipping 'I')."""
        return chr(ord("A") + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (handling the missing 'I')."""
        col_char = col_char.upper()
        if col_char >= "J":  # 'I' is skipped
            return ord(col_char) - ord("A") - 1
        else:
            return ord(col_char) - ord("A")

    def play_move(self, color, move):
        """Processes a move and updates the board."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(",")
        positions = []

        for stone in stones:
            stone = stone.strip()
            col_char = stone[0].upper()
            col = self.label_to_index(col_char)
            row = int(stone[1:]) - 1
            if (
                not (0 <= row < self.size and 0 <= col < self.size)
                or self.board[row, col] != 0
            ):
                print("? Invalid move")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == "B" else 2

        self.last_opponent_move = positions[-1]  # Track the opponent's last move
        self.turn = 3 - self.turn
        print("= ", end="", flush=True)

    def generate_move(self, color):
        """Generates a random move near the opponent's last move."""
        if self.game_over:
            print("? Game over")
            return

        if self.last_opponent_move:
            last_r, last_c = self.last_opponent_move
            potential_moves = [
                (r, c)
                for r in range(max(0, last_r - 2), min(self.size, last_r + 3))
                for c in range(max(0, last_c - 2), min(self.size, last_c + 3))
                if self.board[r, c] == 0
            ]
        else:
            potential_moves = [
                (r, c)
                for r in range(self.size)
                for c in range(self.size)
                if self.board[r, c] == 0
            ]

        if not potential_moves:
            print("? No valid moves")
            return

        selected = random.choice(potential_moves)
        move_str = f"{self.index_to_label(selected[1])}{selected[0]+1}"
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end="", flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board in text format."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join(
                (
                    "X"
                    if self.board[row, col] == 1
                    else "O" if self.board[row, col] == 2 else "."
                )
                for col in range(self.size)
            )
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print("", flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

    # MCTS logic
    def check_win_on_board(self, board):
        """Checks if a player has won on a given board state (numpy array)."""
        size = board.shape[0]
        if size < 6:
            return 0  # Cannot win on smaller boards
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, Diag\, Diag/
        for r in range(size):
            for c in range(size):
                player = board[r, c]
                if player != 0:
                    for dr, dc in directions:
                        count = 0
                        for i in range(6):  # Check up to 6 stones
                            rr, cc = r + i * dr, c + i * dc
                            if (
                                0 <= rr < size
                                and 0 <= cc < size
                                and board[rr, cc] == player
                            ):
                                count += 1
                            else:
                                break  # Sequence broken or off board
                        if count >= 6:  # Found 6 or more
                            return player
        return 0  # No winner

    def is_board_full(self, board):
        """Checks if the board is full (draw condition if no winner)."""
        return not np.any(board == 0)

    def get_legal_moves_on_board(self, board):
        """Returns list of (r, c) tuples for empty squares on a given board."""
        size = board.shape[0]
        return [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]

    # --- Connect 6 Turn Logic Helpers ---

    def get_stones_required_this_turn(self, board_state, player):
        """Determines if the player should place 1 or 2 stones this turn."""
        num_stones_on_board = np.count_nonzero(board_state)
        if player == 1 and num_stones_on_board == 0:
            return 1  # Black's very first move
        else:
            return 2  # All other turns

    def count_stones_played_this_logical_turn(self, board_state, player):
        """
        Counts how many stones the 'player' seems to have placed consecutively
        ending in the current state. This helps determine if their turn should end.
        NOTE: This is heuristic - assumes opponent didn't play between! Relies on
        being called correctly during simulation/expansion.
        """
        # This is tricky. A simple proxy: If board total is 1 and player is Black -> 1 stone played.
        # If board total is even and player is White -> White just finished turn (played 2)
        # If board total is odd > 1 and player is Black -> Black just finished turn (played 2)
        # If board total is odd > 1 and player is White -> White must have played 1 stone this turn.
        # If board total is even > 0 and player is Black -> Black must have played 1 stone this turn.
        num_stones_on_board = np.count_nonzero(board_state)

        if num_stones_on_board == 0:
            return 0
        if player == 1:  # Black
            if num_stones_on_board == 1:
                return 1  # Just played first stone ever
            if num_stones_on_board % 2 != 0:
                return 2  # Odd total > 1, Black finished turn
            else:
                return 1  # Even total > 0, Black played 1st stone of turn
        else:  # White
            if num_stones_on_board % 2 == 0:
                return 2  # Even total > 0, White finished turn
            else:
                return 1  # Odd total > 1, White played 1st stone of turn

    # --- MCTS Simulation ---
    def simulate_random_game(self, board_state, player_turn):
        """
        Simulates a random game from the given state respecting Connect 6 rules.
        Returns winner (1 or 2) or 0 for draw.
        """
        current_board = board_state.copy()
        current_player = player_turn
        size = current_board.shape[0]

        while True:
            # Check for win/draw BEFORE making a move
            winner = self.check_win_on_board(current_board)
            if winner != 0:
                return winner
            legal_moves = self.get_legal_moves_on_board(current_board)
            if not legal_moves:
                return 0  # Draw

            stones_to_play = self.get_stones_required_this_turn(
                current_board, current_player
            )

            for i in range(stones_to_play):
                # Check again for win/draw between placing 1st and 2nd stone
                winner = self.check_win_on_board(current_board)
                if winner != 0:
                    return winner
                legal_moves = self.get_legal_moves_on_board(current_board)
                if not legal_moves:
                    return 0  # Draw

                # Choose and play random move
                move = random.choice(legal_moves)
                r, c = move
                current_board[r, c] = current_player

                # If only one stone required, break inner loop after placing it
                if stones_to_play == 1:
                    break

            # Switch player after placing required stones
            current_player = 3 - current_player

    def find_best_move_mcts(self, iterations=None, time_limit=None):
        """
        Performs MCTS search from the current game state (self.board, self.turn).
        Returns the best move (r, c) found.
        """
        if iterations is None and time_limit is None:
            time_limit = 2.0  # Default to 2 seconds if nothing else specified

        start_time = time.time()
        elapsed_time = 0

        # Create root node representing the current actual game state
        root_node = MCTSNode(
            state=self.board.copy(),
            player_turn=self.turn,
            parent=None,
            move=None,
            game_logic=self,
        )  # Pass self for access to helper methods

        iteration_count = 0
        while True:
            # Check termination conditions
            if time_limit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_limit:
                    break
            elif iterations is not None:
                if iteration_count >= iterations:
                    break
            else:  # Safety break if no condition set (shouldn't happen with default)
                if iteration_count >= 1000:
                    break

            # --- MCTS Cycle ---
            node = root_node

            # 1. Selection: Descend tree using UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_best_child()
                if (
                    node is None
                ):  # Should only happen if root is terminal or has no children
                    print(
                        "Warning: Selection reached unexpected None node",
                        file=sys.stderr,
                    )
                    break  # Cannot proceed further down this path

            if node is None:
                continue  # Start next iteration if selection failed

            # 2. Expansion: If node is not terminal and not fully expanded, add a child
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()  # Expand creates and returns the new child node

            # 3. Simulation: Simulate from the selected or newly expanded node
            # The simulation starts from 'node.state' with 'node.player_turn'
            simulation_result = self.simulate_random_game(node.state, node.player_turn)

            # 4. Backpropagation: Update stats from leaf node ('node') back to root
            temp_node = node
            while temp_node is not None:
                temp_node.update(simulation_result)
                temp_node = temp_node.parent

            iteration_count += 1
            # --- End of MCTS Cycle ---

        # --- Choose Best Move ---
        if not root_node.children:
            print("Warning: MCTS root has no children after search!", file=sys.stderr)
            # Fallback: return a random legal move from the original state
            legal_moves = self.get_legal_moves()
            return random.choice(legal_moves) if legal_moves else None

        # Select child with the highest visit count (most robust)
        # Alternatively, could select child with highest win rate (child.wins / child.visits)
        most_visited_child = max(root_node.children, key=lambda c: c.visits)

        print(
            f"MCTS Info: Ran {iteration_count} iterations in {elapsed_time:.2f}s. Best move visits: {most_visited_child.visits}",
            file=sys.stderr,
        )
        return most_visited_child.move  # Return the (r, c) tuple of the best move


if __name__ == "__main__":
    game = Connect6Game()
    game.run()
