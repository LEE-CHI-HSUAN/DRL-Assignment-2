import math
import random
import sys
import numpy as np
import time
import copy  # For deep copies of board state if needed


class Connect6Logic:
    """Stateless container for Connect 6 game rules."""

    def __init__(self, size=19):
        self.size = size

    def get_player_for_n_stones(self, n_stones):
        """Returns the player (1 or 2) whose turn it is *before* placing the (n_stones+1)-th stone."""
        if n_stones == 0:
            return 1  # Black starts
        # After Black's first move (n_stones=1), it's White's turn.
        # White plays 2 stones (n_stones becomes 2, then 3). After n_stones=3, it's Black's turn.
        # Black plays 2 stones (n_stones becomes 4, then 5). After n_stones=5, it's White's turn.
        # Pattern:
        # n=0: P1
        # n=1, 2: P2
        # n=3, 4: P1
        # n=5, 6: P2
        # n=7, 8: P1
        if n_stones % 4 == 1 or n_stones % 4 == 2:
            return 2  # White's turn
        else:  # n_stones % 4 == 0 (but >0) or n_stones % 4 == 3
            return 1  # Black's turn

    def get_stones_required_for_n_stones(self, n_stones):
        """Determines how many stones the current player should place."""
        if n_stones == 0:
            return 1  # Black's very first move
        else:
            return 2  # All other turns require 2 stones

    def check_win_on_board(self, board):
        """Checks if a player has won on a given board state (numpy array).
        Returns: 1 (Black Win), 2 (White Win), 0 (No Winner Yet), 3 (Draw - board full)
        """
        size = board.shape[0]
        if size < 6:
            return 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, Diag\, Diag/

        board_is_full = True  # Assume full until proven otherwise

        for r in range(size):
            for c in range(size):
                player = board[r, c]
                if player == 0:
                    board_is_full = False  # Found an empty cell
                    continue  # No need to check win from empty cell

                # Check for win starting from (r, c)
                for dr, dc in directions:
                    count = 0
                    for i in range(6):  # Check up to 6 stones in this direction
                        rr, cc = r + i * dr, c + i * dc
                        if (
                            0 <= rr < size
                            and 0 <= cc < size
                            and board[rr, cc] == player
                        ):
                            count += 1
                        else:
                            break  # Sequence broken or off board
                    if count >= 6:
                        return player  # Found a win

        # If no winner found after checking all cells
        if board_is_full:
            return 3  # Draw
        else:
            return 0  # Game ongoing

    def is_board_full(self, board):
        """Checks if the board is full."""
        return not np.any(board == 0)

    def get_legal_moves_on_board(self, board):
        """Returns list of (r, c) tuples for empty squares on a given board."""
        size = board.shape[0]
        # Optimization: Use np.where for potentially faster finding of empty cells
        empty_cells = np.where(board == 0)
        return list(zip(empty_cells[0], empty_cells[1]))  # List of (r, c) tuples


class MCTSNode:
    def __init__(self, state, n_stones, parent=None, move=None, game_logic=None):
        self.state = state  # The board state (numpy array)
        self.n_stones = n_stones  # Total stones on the board IN this state
        self.parent = parent
        self.move = move  # The move (r, c) that led TO this state

        self.children = []
        self.wins = 0  # Wins from the perspective of the player who JUST MOVED to reach this state
        self.visits = 0

        # Get game logic from parent or passed in
        self._game_logic = game_logic if game_logic else parent._game_logic

        # --- Determine Player Turn & Cache Terminal Status ---
        # Player whose turn it is *in* this state (derived from n_stones)
        self.player_turn = self._game_logic.get_player_for_n_stones(self.n_stones)
        self._cached_winner = self._game_logic.check_win_on_board(
            self.state
        )  # Cache winner (0, 1, 2, or 3 for draw)

        # Determine untried moves only if the game is not over
        self.untried_moves = []
        if not self.is_terminal():
            self.untried_moves = self._game_logic.get_legal_moves_on_board(self.state)
            random.shuffle(self.untried_moves)

    def is_fully_expanded(self):
        # A terminal node is implicitly fully expanded (no more moves possible)
        return self.is_terminal() or len(self.untried_moves) == 0

    def is_terminal(self):
        # Use the cached result
        return self._cached_winner != 0  # 0 means ongoing

    def get_winner(self):
        """Returns the cached winner (0, 1, 2, or 3 for draw)."""
        return self._cached_winner

    def select_best_child(self, C_param=1.414):
        """Selects the best child node using the UCB1 formula."""
        if not self.children:
            return None

        log_parent_visits = math.log(self.visits)

        best_score = -1.0
        best_children = []

        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children - assign effectively infinite score
                # Return immediately if an unvisited child is found? Optional optimization.
                return child  # Select unvisited child first

            # UCB calculation
            exploitation_term = child.wins / child.visits
            exploration_term = C_param * math.sqrt(log_parent_visits / child.visits)
            score = exploitation_term + exploration_term

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return (
            random.choice(best_children) if best_children else None
        )  # Break ties randomly

    def expand(self):
        """Expands the node by adding one child for an untried move."""
        if self.is_fully_expanded():
            raise RuntimeError(
                "Cannot expand a node that is terminal or fully expanded."
            )
        if not self.untried_moves:
            raise RuntimeError(
                "Expand called with no untried moves but not terminal/fully expanded."
            )

        move = self.untried_moves.pop()
        r, c = move

        # Create the next state by applying the move
        next_state = self.state.copy()
        # The player making the move is the one whose turn it is *at this node*
        player_making_move = self.player_turn
        next_state[r, c] = player_making_move

        # Calculate n_stones for the child state
        next_n_stones = self.n_stones + 1

        # Create and add the new child node
        child_node = MCTSNode(
            state=next_state,
            n_stones=next_n_stones,
            parent=self,
            move=move,
            game_logic=self._game_logic,
        )
        self.children.append(child_node)
        return child_node

    def update(self, simulation_result):
        """Updates the node's visits and wins based on the simulation outcome."""
        self.visits += 1

        # Determine the player who made the move *into* this state.
        # This is the player whose turn it was in the parent node.
        # If this node is the root (no parent), this doesn't directly apply,
        # but the perspective holds for children.
        # The player who moved *to* this state is (3 - self.player_turn) if self.n_stones > 0.
        # Let's stick to the original convention: wins are from the perspective of the
        # player whose turn it was IN THE PARENT.
        # Which player was it in the parent? It was the player who is *not* self.player_turn.
        parent_player_turn = 3 - self.player_turn

        # simulation_result is the winner (1, 2, or 3 for draw)
        if simulation_result == parent_player_turn:
            self.wins += 1
        elif simulation_result == 3:  # Draw
            self.wins += 0.5  # Optional: Award half point for draw


class Connect6Game:
    def __init__(self, size=19):
        self.logic = Connect6Logic(size)
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.n_stones = 0  # Track total stones played
        self.game_over = False
        # self.last_opponent_move = None # Might not be needed anymore

    def reset_board(self):
        """Clears the board and resets the game state."""
        self.board.fill(0)
        self.n_stones = 0
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets a new board size and resets the game."""
        self.size = size
        self.logic = Connect6Logic(size)  # Update logic object too
        self.board = np.zeros((size, size), dtype=int)
        self.n_stones = 0
        self.game_over = False
        print("= ", flush=True)

    # Keep index_to_label and label_to_index as they are utility functions
    def index_to_label(self, col):
        """Converts a column index to a letter (skipping 'I')."""
        if col < 0 or col >= self.size:
            return "?"  # Handle invalid index
        return chr(ord("A") + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (handling the missing 'I')."""
        col_char = col_char.upper()
        if not "A" <= col_char <= "T" or col_char == "I":
            return -1  # Basic validation
        if col_char >= "J":  # 'I' is skipped
            return ord(col_char) - ord("A") - 1
        else:
            return ord(col_char) - ord("A")

    def play_move(self, color, move_str):
        """Processes a move string (e.g., "H10" or "H10,J9"), updates the board and n_stones."""
        if self.game_over:
            print("? Game over")
            return False  # Indicate failure

        player = 1 if color.upper() == "B" else 2
        stones = move_str.split(",")
        positions = []

        # Validate all parts of the move first
        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print(f"? Invalid move format part: {stone}")
                return False
            col_char = stone[0].upper()
            row_str = stone[1:]
            col = self.label_to_index(col_char)
            try:
                row = int(row_str) - 1
            except ValueError:
                print(f"? Invalid row in move part: {stone}")
                return False

            if not (0 <= row < self.size and 0 <= col < self.size):
                print(f"? Move out of bounds: {stone} ({row},{col})")
                return False
            if self.board[row, col] != 0:
                print(f"? Square occupied: {stone} ({row},{col})")
                return False
            # Check for duplicate positions within the same move command
            if (row, col) in positions:
                print(f"? Duplicate stone placement in move: {stone}")
                return False
            positions.append((row, col))

        # --- Check if the number of stones played matches the rules ---
        expected_stones = self.logic.get_stones_required_for_n_stones(self.n_stones)
        if len(positions) != expected_stones:
            print(
                f"? Incorrect number of stones played. Expected {expected_stones}, got {len(positions)}"
            )
            # Allow it maybe? Ludii might send moves that don't strictly follow n_stones
            # if it's just replaying history or setting up a position.
            # Let's proceed but log a warning.
            print(
                f"Warning: Move '{move_str}' has {len(positions)} stones, but expected {expected_stones} based on n_stones={self.n_stones}",
                file=sys.stderr,
            )
            # Consider if this should be a hard error depending on Ludii's behavior.

        # If validation passes, apply the moves
        for row, col in positions:
            self.board[row, col] = player
            self.n_stones += 1  # Increment stone count FOR EACH stone placed

        # Check for game over after the move(s)
        winner = self.logic.check_win_on_board(self.board)
        if winner != 0:
            self.game_over = True
            # Ludii handles declaring the winner, we just note it internally

        # self.turn is NOT managed here. GTP loop determines next command.
        print("= ", end="", flush=True)  # GTP Success response
        return True  # Indicate success

    def generate_move(self, color_str):
        """Generates an intelligent move (or sequence of moves) using MCTS."""
        if self.game_over:
            print("? Game is over")
            return

        player = 1 if color_str.upper() == "B" else 2
        current_player_expected = self.logic.get_player_for_n_stones(self.n_stones)

        if player != current_player_expected:
            print(
                f"? Received genmove for {color_str} (Player {player}), but expected Player {current_player_expected}'s turn (n_stones={self.n_stones})",
                file=sys.stderr,
            )
            # This might happen if GTP state desyncs. Should we proceed or error out?
            # Let's proceed assuming Ludii knows what it's doing, but use 'player' requested.
            # Force our internal state to match? Risky. Let MCTS plan for 'player'.

        print(
            f"Generating move for Player {player} ({color_str}). Current n_stones={self.n_stones}",
            file=sys.stderr,
        )

        stones_to_play = self.logic.get_stones_required_for_n_stones(self.n_stones)
        print(f"Need to generate {stones_to_play} stone(s).", file=sys.stderr)

        # --- Call MCTS ---
        # We need to find the best *sequence* of 1 or 2 moves.
        # Strategy: Run MCTS once. Find best first move. If needed, find best second move
        # from the children of the node corresponding to the first move.
        time_limit_sec = 4.8  # Example time limit
        start_time = time.time()

        # MCTS needs the *current* board state and stone count
        # It will plan for the 'player' whose turn it *should* be based on n_stones
        root_node, iterations = self.run_mcts(
            self.board.copy(), self.n_stones, time_limit=time_limit_sec
        )

        elapsed_time = time.time() - start_time
        print(
            f"MCTS Info: Ran {iterations} iterations in {elapsed_time:.2f}s.",
            file=sys.stderr,
        )

        moves_coords = []
        if not root_node or not root_node.children:
            print("? MCTS failed or no moves found. Playing randomly.", file=sys.stderr)
            # Fallback: generate random legal moves
            legal_moves = self.logic.get_legal_moves_on_board(self.board)
            random.shuffle(legal_moves)
            moves_coords = legal_moves[:stones_to_play]
            if len(moves_coords) < stones_to_play:
                print("= resign\n\n", end="", flush=True)  # Resign if no moves
                print("? No legal moves found, resigning.", file=sys.stderr)
                self.game_over = True
                return
        else:
            # 1. Find the best first move (most visited child of root)
            best_child_1 = max(root_node.children, key=lambda c: c.visits)
            move1 = best_child_1.move
            moves_coords.append(move1)
            print(
                f"MCTS Best Move 1: {self.format_move_coord(move1)} (Visits: {best_child_1.visits}, WinRate: {best_child_1.wins/best_child_1.visits:.3f})",
                file=sys.stderr,
            )

            if stones_to_play == 2:
                # 2. Find the best second move (most visited child of best_child_1)
                # Ensure the best_child_1 node was expanded enough
                if not best_child_1.children:
                    # This can happen if time runs out before exploring the second move deeply.
                    # Fallback: Play a random valid move after the first one.
                    print(
                        "Warning: MCTS didn't explore second move well. Choosing random second move.",
                        file=sys.stderr,
                    )
                    temp_board = self.board.copy()
                    temp_board[move1[0], move1[1]] = player
                    legal_moves_after_1 = self.logic.get_legal_moves_on_board(
                        temp_board
                    )
                    if legal_moves_after_1:
                        move2 = random.choice(legal_moves_after_1)
                        moves_coords.append(move2)
                    else:
                        # This shouldn't happen unless board is full after 1 move
                        print(
                            "Error: No legal moves for second stone?", file=sys.stderr
                        )
                        # Just send the first move found
                        pass
                else:
                    best_child_2 = max(best_child_1.children, key=lambda c: c.visits)
                    move2 = best_child_2.move
                    moves_coords.append(move2)
                    print(
                        f"MCTS Best Move 2: {self.format_move_coord(move2)} (Visits: {best_child_2.visits}, WinRate: {best_child_2.wins/best_child_2.visits:.3f})",
                        file=sys.stderr,
                    )

        # --- Format and Play Move ---
        move_str = ",".join(self.format_move_coord(mc) for mc in moves_coords)

        # Play the move(s) on our internal board *before* sending response
        success = self.play_move(color_str, move_str)

        if success:
            # Output for GTP (play_move already printed '= ')
            print(f"{move_str}\n\n", end="", flush=True)
            print(f"Played move: {move_str}", file=sys.stderr)
        else:
            # Should not happen if MCTS/random generated valid moves
            print("? Error generating or playing move internally.", file=sys.stderr)
            # Maybe resign?
            print("= resign\n\n", end="", flush=True)
            self.game_over = True

    def format_move_coord(self, move_coord):
        """Converts (r, c) tuple to GTP string like H10."""
        r, c = move_coord
        return f"{self.index_to_label(c)}{r+1}"

    def show_board(self):
        """Displays the board in text format (GTP compliant)."""
        print("= ")  # Start GTP response block
        header = "  " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(header)
        for r in range(self.size - 1, -1, -1):
            line = (
                f"{r+1:<2}"
                + "".join(  # Use join without spaces for denser view if preferred
                    (
                        " B"
                        if self.board[r, col] == 1  # Black = B
                        else (
                            " W" if self.board[r, col] == 2 else " ."  # White = W
                        )  # Empty = .
                    )
                    for col in range(self.size)
                )
            )
            print(line)
        print(flush=True)

    def list_commands(self):
        """Lists known GTP commands."""
        # Send an empty success response first if required by GTP spec for list_commands
        print(
            "= boardsize\nclear_board\nplay\ngenmove\nshowboard\nlist_commands\nquit\nname\nversion\nknown_command\n",
            flush=True,
        )
        # Note: Added some common GTP commands like name, version, known_command

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        parts = command.split()
        if not parts:
            return
        cmd_id = ""
        if parts[0].isdigit():
            cmd_id = parts[0]
            parts = parts[1:]
            if not parts:
                print(f"?{cmd_id} Missing command")
                return

        cmd = parts[0].lower()
        args = parts[1:]

        # Prepend ID to success/failure markers
        success_prefix = f"={cmd_id} " if cmd_id else "= "
        failure_prefix = f"?{cmd_id} " if cmd_id else "? "

        try:
            if cmd == "name":
                print(f"{success_prefix}Connect6 MCTS Agent")
            elif cmd == "version":
                print(f"{success_prefix}1.1")
            elif cmd == "known_command":
                known = cmd in [
                    "name",
                    "version",
                    "known_command",
                    "list_commands",
                    "quit",
                    "boardsize",
                    "clear_board",
                    "play",
                    "genmove",
                    "showboard",
                ]
                print(f"{success_prefix}{'true' if known else 'false'}")
            elif cmd == "list_commands":
                print(
                    f"{success_prefix}boardsize\nclear_board\nplay\ngenmove\nshowboard\nlist_commands\nquit\nname\nversion\nknown_command"
                )
            elif cmd == "quit":
                print(f"{success_prefix}")  # Empty success response
                sys.exit(0)
            elif cmd == "boardsize":
                if len(args) != 1:
                    raise ValueError("Expected 1 argument for boardsize")
                size = int(args[0])
                if size < 6 or size > 19:
                    raise ValueError(
                        "Board size must be between 6 and 19"
                    )  # Typical range
                self.set_board_size(size)  # This already prints success internally
                # Adjust the success prefix for set_board_size's internal print
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)
            elif cmd == "clear_board":
                self.reset_board()  # Prints success internally
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)
            elif cmd == "play":
                if len(args) != 2:
                    raise ValueError("Expected 2 arguments for play: color move")
                color, move_str = args[0], args[1]
                if not self.play_move(
                    color, move_str
                ):  # Prints success internally if ok
                    raise ValueError("Invalid move provided")
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)
            elif cmd == "genmove":
                if len(args) != 1:
                    raise ValueError("Expected 1 argument for genmove: color")
                color = args[0]
                # genmove handles its own GTP output including success/failure prefix + newline
                self.generate_move(color)
                # We need to adjust the print logic in generate_move slightly
                # Let's assume generate_move prints the '=' or '?' itself without ID
                # So we just print the ID here if needed.
                if cmd_id:
                    print(
                        f"{cmd_id}", end="", flush=True
                    )  # Hmm, this is awkward with genmove's output.

                # --- Revised approach for genmove + ID ---
                # generate_move will *not* print the '= ID' or '? ID' part.
                # It will print '= MOVES\n\n' or '? ERROR_MSG\n\n' or '= resign\n\n'
                # We capture its success/failure status implicitly or explicitly
                # This requires generate_move to return status or for process_command
                # to check self.game_over or similar after the call.
                # For now, let's stick to the simpler way: generate_move prints everything.
                # If Ludii requires IDs back, this needs more careful handling.

            elif cmd == "showboard":
                self.show_board()  # Prints success block internally
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)
            else:
                raise ValueError(f"Unknown command: {cmd}")

        except ValueError as e:
            print(f"{failure_prefix}{str(e)}")
        except Exception as e:
            # Catch unexpected errors
            print(f"{failure_prefix}Internal error: {str(e)}")
            import traceback

            traceback.print_exc(file=sys.stderr)  # Log stack trace for debugging

        sys.stdout.flush()  # Ensure output is sent

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    print("Input stream closed.", file=sys.stderr)
                    break
                print(f"Received command: {line.strip()}", file=sys.stderr)  # Log input
                self.process_command(line)
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting.", file=sys.stderr)
                break
            except EOFError:
                print("EOF received, exiting.", file=sys.stderr)
                break
            # Let process_command handle other exceptions and print GTP errors

    # --- MCTS Simulation ---
    def simulate_random_game(self, board_state, n_stones_start):
        """
        Simulates a random game from the given state respecting Connect 6 rules.
        Uses n_stones to manage turns.
        Returns winner (1 or 2) or 3 for draw.
        """
        current_board = board_state.copy()
        n_stones = n_stones_start
        size = current_board.shape[0]

        while True:
            # Check for win/draw BEFORE making a move
            winner = self.logic.check_win_on_board(current_board)
            if winner != 0:  # Win (1 or 2) or Draw (3)
                return winner

            # Determine current player and stones needed for *this* turn
            current_player = self.logic.get_player_for_n_stones(n_stones)
            stones_to_play_this_turn = self.logic.get_stones_required_for_n_stones(
                n_stones
            )

            legal_moves = self.logic.get_legal_moves_on_board(current_board)
            if not legal_moves:
                # Should have been caught by check_win_on_board as a draw, but double check
                return 3  # Draw (no moves left)

            # Play the required number of stones randomly
            for i in range(stones_to_play_this_turn):
                # Check for win/draw *between* placing 1st and 2nd stone
                # Although check_win_on_board is relatively quick now,
                # we can optimize by only checking *after* the full turn usually.
                # However, a win can happen on the first stone of a 2-stone turn.
                winner_interim = self.logic.check_win_on_board(current_board)
                if winner_interim != 0:
                    return winner_interim

                legal_moves = self.logic.get_legal_moves_on_board(current_board)
                if not legal_moves:
                    # If no moves left after placing first stone, it's a draw
                    return 3  # Draw

                # Choose and play random move
                move = random.choice(legal_moves)
                r, c = move
                current_board[r, c] = current_player
                n_stones += 1  # Increment stone count

                # If only one stone required this turn, break inner loop
                if stones_to_play_this_turn == 1:
                    break
            # Loop continues to next player's turn

    def run_mcts(
        self, initial_board_state, initial_n_stones, iterations=None, time_limit=None
    ):
        """
        Performs MCTS search from the given state.
        Returns the root node of the search tree and the number of iterations performed.
        """
        if iterations is None and time_limit is None:
            time_limit = 2.0  # Default time limit

        start_time = time.time()
        elapsed_time = 0

        # Create root node
        root_node = MCTSNode(
            state=initial_board_state.copy(),  # Ensure root has its own copy
            n_stones=initial_n_stones,
            parent=None,
            move=None,
            game_logic=self.logic,
        )

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
            else:  # Safety break
                if iteration_count >= 100000:  # Adjust as needed
                    break

            node = root_node

            # --- MCTS Cycle ---
            # 1. Selection: Descend tree using UCB1 until a non-terminal, not fully expanded node is found
            while not node.is_terminal() and node.is_fully_expanded():
                best_child = node.select_best_child()
                if best_child is None:
                    # This might happen if selection hits a terminal node mistakenly marked as non-terminal,
                    # or if there's an issue with expansion/selection logic.
                    print(
                        f"Warning: select_best_child returned None for non-terminal, fully_expanded node. Visits={node.visits}, Children={len(node.children)}",
                        file=sys.stderr,
                    )
                    # If stuck, break this simulation path and start a new iteration
                    node = None  # Signal failure to proceed down this path
                    break
                node = best_child

            if node is None:  # Selection failed path
                iteration_count += 1  # Count the attempt
                continue

            # 2. Expansion: If node is not terminal and not fully expanded, add a child
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()  # Expand creates and returns the new child node

            # 3. Simulation: Simulate from the selected or newly expanded node
            # Check if the node we landed on (or expanded to) is already terminal
            winner = node.get_winner()
            if winner != 0:  # Game already ended at this node
                simulation_result = winner
            else:
                # Simulate from 'node.state' and 'node.n_stones'
                simulation_result = self.simulate_random_game(node.state, node.n_stones)

            # 4. Backpropagation: Update stats from leaf node ('node') back to root
            temp_node = node
            while temp_node is not None:
                temp_node.update(simulation_result)
                temp_node = temp_node.parent

            iteration_count += 1
            # --- End of MCTS Cycle ---

        return root_node, iteration_count


if __name__ == "__main__":
    game = Connect6Game()
    game.run()
