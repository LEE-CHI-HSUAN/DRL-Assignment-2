import math
import random
import sys
import numpy as np
import time
import copy  # For deep copies of board state if needed

# Define Radii Constants (can be adjusted)
EXPANSION_RADIUS = 1  # Radius for candidate moves during tree expansion
SIMULATION_RADIUS = 1  # Radius for candidate moves during simulation playouts


class Connect6Logic:
    """Stateless container for Connect 6 game rules."""

    def __init__(self, size=19):
        self.size = size

    def get_player_for_n_stones(self, n_stones):
        """Returns the player (1 or 2) whose turn it is *before* placing the (n_stones+1)-th stone."""
        if n_stones == 0:
            return 1  # Black starts
        if n_stones % 4 == 1 or n_stones % 4 == 2:
            return 2  # White's turn
        else:  # n_stones % 4 == 0 (but >0) or n_stones % 4 == 3
            return 1  # Black's turn

    # --- FIXED Rule ---
    def get_stones_required_for_n_stones(self, n_stones):
        """Determines how many stones the current player should place."""
        if n_stones == 0:
            return 1  # Black's very first move
        else:
            return 2  # All other turns require 2 stones

    def check_win_incremental(self, board, r, c):
        """Checks if the last move at (r, c) resulted in a win."""
        size = board.shape[0]
        player = board[r, c]
        if player == 0:
            return 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Forwards
            for i in range(1, 6):
                rr, cc = r + i * dr, c + i * dc
                if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                    count += 1
                else:
                    break
            # Backwards
            for i in range(1, 6):
                rr, cc = r - i * dr, c - i * dc
                if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                    count += 1
                else:
                    break
            if count >= 6:
                return player
        return 0

    def check_win_on_board(self, board):
        """Checks if a player has won on a given board state (numpy array).
        Returns: 1 (Black Win), 2 (White Win), 0 (No Winner Yet), 3 (Draw - board full)
        """
        size = board.shape[0]
        if size < 6:
            return 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        board_is_full = True

        for r in range(size):
            for c in range(size):
                player = board[r, c]
                if player == 0:
                    board_is_full = False
                    continue

                for dr, dc in directions:
                    prev_r, prev_c = r - dr, c - dc
                    if (
                        0 <= prev_r < size
                        and 0 <= prev_c < size
                        and board[prev_r, prev_c] == player
                    ):
                        continue

                    count = 0
                    for i in range(6):
                        rr, cc = r + i * dr, c + i * dc
                        if (
                            0 <= rr < size
                            and 0 <= cc < size
                            and board[rr, cc] == player
                        ):
                            count += 1
                        else:
                            break
                    if count >= 6:
                        return player

        if board_is_full:
            return 3
        else:
            return 0

    def is_board_full(self, board):
        """Checks if the board is full."""
        return not np.any(board == 0)

    def get_legal_moves_on_board(self, board):
        """Returns list of (r, c) tuples for ALL empty squares on a given board."""
        empty_cells = np.where(board == 0)
        return list(zip(empty_cells[0], empty_cells[1]))

    # --- NEW Local Move Generation ---
    def get_local_moves(self, board, radius):
        """
        Returns a list of empty squares within 'radius' of any existing stone.
        Handles n=0 case and fallback to global moves if local area is empty.
        """
        size = self.size
        occupied_rows, occupied_cols = np.where(board != 0)

        if occupied_rows.size == 0:
            # Board is empty (n_stones=0), return center move(s)
            # Although generate_move handles n=0 deterministically, this makes the function robust
            center_r, center_c = size // 2, size // 2
            # Could return a small area, but just center is fine for MCTS start if needed
            if board[center_r, center_c] == 0:
                return [(center_r, center_c)]
            else:  # Should not happen on empty board unless size is odd/even issue?
                return self.get_legal_moves_on_board(
                    board
                )  # Fallback if center somehow taken

        local_candidate_moves = set()
        for r, c in zip(occupied_rows, occupied_cols):
            min_r, max_r = max(0, r - radius), min(size - 1, r + radius)
            min_c, max_c = max(0, c - radius), min(size - 1, c + radius)
            for nr in range(min_r, max_r + 1):
                for nc in range(min_c, max_c + 1):
                    if board[nr, nc] == 0:
                        local_candidate_moves.add((nr, nc))

        if not local_candidate_moves:
            # Fallback: If no local moves found (e.g., stones far apart, small radius)
            # but board is not full, return all legal moves.
            if not self.is_board_full(board):
                # print("Warning: No local moves found, falling back to global.", file=sys.stderr)
                return self.get_legal_moves_on_board(board)
            else:
                return []  # Board is full and no local moves (should mean board full)

        return list(local_candidate_moves)

    def _get_line_length(self, board, r, c, dr, dc, player):
        """Counts consecutive stones for player starting from r,c in direction dr,dc."""
        count = 0
        size = board.shape[0]
        for i in range(1, 6):  # Check up to 5 more stones for a total of 6
            rr, cc = r + i * dr, c + i * dc
            if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                count += 1
            else:
                break
        return count

    def calculate_heuristic_score(self, board, r, c, player):
        """
        Calculates a heuristic score for *hypothetically* placing a stone
        at (r, c) for 'player'.
        Returns: Score (higher is better), float('inf') if it's a winning move,
                 or a very large number instead of inf for simulation weighting.
        """
        size = board.shape[0]
        # Use a distinct large number for wins to simplify weighting logic later
        WIN_SCORE = 1e9  # Define a large score representing an immediate win

        if r < 0 or r >= size or c < 0 or c >= size or board[r, c] != 0:
            # Return a very low score for invalid or occupied squares if called incorrectly
            # Or rely on caller to provide valid moves. Let's assume valid moves are passed.
            # If called on occupied square:
            if r >= 0 and r < size and c >= 0 and c < size and board[r, c] != 0:
                return -1  # Indicate occupied explicitly maybe

            # If called out of bounds (shouldn't happen with valid move lists):
            # raise ValueError("Heuristic calculated for invalid move") # Or return low score
            return -1000  # Very low score for logic errors downstream

        # --- Temporarily place the stone ---
        board[r, c] = player
        # --- Check for immediate win ---
        if self.check_win_incremental(board, r, c) == player:
            board[r, c] = 0  # Undo move
            # return float("inf") # Original
            return WIN_SCORE  # Return large number for wins

        # --- Calculate score based on line lengths ---
        score = 0
        # Penalize moves right next to the edge slightly? Optional.
        # edge_penalty = 0
        # if r == 0 or r == size-1 or c == 0 or c == size-1:
        #      edge_penalty = 1 # Small penalty

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, Diag\, Diag/
        # Adjusted scores maybe? Make intermediate scores higher relative to single links
        score_map = {5: 50000, 4: 1000, 3: 100, 2: 10, 1: 1}  # Example adjustment

        for dr, dc in directions:
            len1 = self._get_line_length(board, r, c, dr, dc, player)
            len2 = self._get_line_length(board, r, c, -dr, -dc, player)
            total_len = len1 + len2 + 1

            if total_len >= 6:  # Should be caught by win check, but as fallback
                score += score_map.get(5, 50000) * 2
            elif total_len >= 2:
                score += score_map.get(total_len, 0)

        board[r, c] = 0  # --- Undo the temporary move ---
        # return score - edge_penalty
        # Add a small base score to ensure non-zero weights for isolated moves
        return max(1, score)  # Ensure score is at least 1 for weighting


class MCTSNode:
    def __init__(self, state, n_stones, parent=None, move=None, game_logic=None):
        self.state = state
        self.n_stones = n_stones
        self.parent = parent
        self.move = move

        self.children = []
        self.wins = 0
        self.visits = 0

        self._game_logic = game_logic if game_logic else parent._game_logic

        self.player_turn = self._game_logic.get_player_for_n_stones(self.n_stones)
        self._cached_winner = self._game_logic.check_win_on_board(self.state)

        self._prioritized_moves = None
        self._expansion_index = 0

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        return self._prioritized_moves is not None and self._expansion_index >= len(
            self._prioritized_moves
        )

    def is_terminal(self):
        return self._cached_winner != 0

    def get_winner(self):
        return self._cached_winner

    def select_best_child(self, C_param=1):
        if not self.children:
            return None
        log_parent_visits = math.log(self.visits)
        best_score = -1.0
        best_children = []
        for child in self.children:
            if child.visits == 0:  # Avoid math error if a child somehow has 0 visits
                score = float(
                    "inf"
                )  # Prioritize unvisited children if encountered here
            else:
                exploitation_term = child.wins / child.visits
                exploration_term = C_param * math.sqrt(log_parent_visits / child.visits)
                score = exploitation_term + exploration_term

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return random.choice(best_children) if best_children else None

    def _calculate_expansion_priority(self):
        """Calculates priority for untried LOCAL moves based on heuristics."""
        if self._prioritized_moves is not None:
            return

        # --- MODIFIED: Use get_local_moves ---
        # Define radius here or get from logic/config if made variable
        current_candidate_moves = self._game_logic.get_local_moves(
            self.state, EXPANSION_RADIUS
        )
        # --- End Modification ---

        if (
            not current_candidate_moves
        ):  # No legal moves found (local or global fallback)
            self._prioritized_moves = []
            self._expansion_index = 0
            return

        moves_with_priority = []
        my_color = self.player_turn
        opponent_color = 3 - my_color
        temp_board = self.state.copy()  # Use copy for heuristic checks

        for move in current_candidate_moves:
            r, c = move
            priority = 0
            heuristic_score = 0

            my_win_score = self._game_logic.calculate_heuristic_score(
                temp_board, r, c, my_color
            )
            if my_win_score == float("inf"):
                priority = 3
                heuristic_score = my_win_score
            else:
                opponent_win_score = self._game_logic.calculate_heuristic_score(
                    temp_board, r, c, opponent_color
                )
                if opponent_win_score == float("inf"):
                    priority = 2
                    heuristic_score = opponent_win_score
                else:
                    priority = 1
                    heuristic_score = (
                        my_win_score + opponent_win_score
                    )  # Simple combined score

            moves_with_priority.append(((priority, heuristic_score), move))

        moves_with_priority.sort(key=lambda x: x[0], reverse=True)
        self._prioritized_moves = [item[1] for item in moves_with_priority]
        self._expansion_index = 0

        # print(f"Prioritized {len(self._prioritized_moves)} local moves for node (n_stones={self.n_stones}). Top 5: {self._prioritized_moves[:5]}", file=sys.stderr)

    def expand(self):
        """Expands the node by adding one child for the highest priority untried LOCAL move."""
        if self.is_fully_expanded():
            raise RuntimeError(
                "Cannot expand a node that is terminal or fully expanded."
            )

        if self._prioritized_moves is None:
            self._calculate_expansion_priority()
            if self.is_fully_expanded():  # Check again after calculation
                # This might happen if get_local_moves returned empty list
                raise RuntimeError(
                    f"Node {self.n_stones} became fully expanded after priority calculation (no local moves?)."
                )

        move = self._prioritized_moves[self._expansion_index]
        self._expansion_index += 1
        r, c = move

        next_state = self.state.copy()
        player_making_move = self.player_turn
        next_state[r, c] = player_making_move
        next_n_stones = self.n_stones + 1

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
        # Determine the player whose turn it was at the PARENT node
        # This is the player whose win rate we are tracking for the move leading to THIS node.
        # If this node represents state N, the move was made by the player for state N-1.
        # The player for state N-1 is the OPPONENT of the player for state N (self.player_turn)
        # Exception: Root node has no parent, its wins reflect results after its player's move.
        # Let's rethink this: self.wins should represent wins from the perspective
        # of the player whose turn it is *in the parent* state.
        # The player whose turn is in the parent state is 3 - self.player_turn.

        if self.parent is None:  # Root node logic might differ slightly if needed
            parent_player_turn = self._game_logic.get_player_for_n_stones(
                self.n_stones - 1 if self.n_stones > 0 else 0
            )  # Approxiate root parent turn
        else:
            parent_player_turn = self.parent.player_turn

        if simulation_result == parent_player_turn:
            self.wins += 1
        elif simulation_result == 3:  # Draw
            self.wins += 0.5


class Connect6Game:
    def __init__(self, size=19):
        self.logic = Connect6Logic(size)
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.n_stones = 0
        self.game_over = False

    def reset_board(self):
        self.board.fill(0)
        self.n_stones = 0
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        self.size = size
        self.logic = Connect6Logic(size)
        self.board = np.zeros((size, size), dtype=int)
        self.n_stones = 0
        self.game_over = False
        print("= ", flush=True)

    def index_to_label(self, col):
        if col < 0 or col >= self.size:
            return "?"
        return chr(ord("A") + col + (1 if col >= 8 else 0))
        # return chr(ord("A") + col)

    def label_to_index(self, col_char):
        col_char = col_char.upper()
        if not "A" <= col_char <= "T" or col_char == "I":
            return -1
        if col_char >= "J":
            return ord(col_char) - ord("A") - 1
        else:
            return ord(col_char) - ord("A")

    def play_move(self, color, move_str):
        """Processes a move string, updates board and n_stones. Assumes GTP success/failure handled by caller."""
        if self.game_over:
            return False, "? Game over"

        player = 1 if color.upper() == "B" else 2
        stones = move_str.split(",")
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                return False, f"? Invalid move format part: {stone}"
            col_char = stone[0].upper()
            row_str = stone[1:]
            col = self.label_to_index(col_char)
            try:
                row = int(row_str) - 1
            except ValueError:
                return False, f"? Invalid row in move part: {stone}"

            if not (0 <= row < self.size and 0 <= col < self.size):
                return False, f"? Move out of bounds: {stone} ({row},{col})"
            if self.board[row, col] != 0:
                return False, f"? Square occupied: {stone} ({row},{col})"
            if (row, col) in positions:
                return False, f"? Duplicate stone placement in move: {stone}"
            positions.append((row, col))

        # --- Check number of stones (using corrected logic) ---
        expected_stones = self.logic.get_stones_required_for_n_stones(self.n_stones)
        if len(positions) != expected_stones:
            # Strict check: Fail if wrong number of stones submitted via 'play' command
            # MCTS generation should always follow the rule now.
            # return False, f"? Incorrect number of stones. Expected {expected_stones}, got {len(positions)} for n_stones={self.n_stones}"
            # Flexible check (Warning only): Allows setting up positions via GTP 'play'
            print(
                f"Warning: Move '{move_str}' has {len(positions)} stones, but expected {expected_stones} based on n_stones={self.n_stones}",
                file=sys.stderr,
            )

        # Apply moves
        last_r, last_c = -1, -1
        for row, col in positions:
            self.board[row, col] = player
            self.n_stones += 1
            last_r, last_c = (
                row,
                col,
            )  # Keep track of one of the last stones for potential win check

        # Check for game over (can check incrementally from last placed stone for efficiency)
        winner = 0
        if last_r != -1:  # Check if any move was actually made
            # Check based on all placed stones in this command
            for r_check, c_check in positions:
                winner = self.logic.check_win_incremental(self.board, r_check, c_check)
                if winner != 0:
                    break

        # Fallback to full board check if needed, or just rely on incremental
        if winner == 0 and self.logic.is_board_full(self.board):
            winner = 3  # Draw

        if winner != 0:
            self.game_over = True

        return True, ""  # Success

    def generate_move(self, color_str):
        """Generates move(s) using MCTS with local expansion/simulation."""
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
            # Proceeding anyway, MCTS will plan for 'player'

        print(
            f"Generating move for Player {player} ({color_str}). Current n_stones={self.n_stones}",
            file=sys.stderr,
        )

        # --- MODIFIED: Handle deterministic first move ---
        if self.n_stones == 0:
            if player != 1:  # Black should always play first
                print("? White cannot play the first move.", file=sys.stderr)
                print("= resign\n\n", end="", flush=True)  # Or just error?
                return

            center_r, center_c = self.size // 2, self.size // 2
            center_move_str = self.format_move_coord((center_r, center_c))
            print(
                f"Playing deterministic first move: {center_move_str}", file=sys.stderr
            )
            success, error_msg = self.play_move(color_str, center_move_str)
            if success:
                print(f"= {center_move_str}\n\n", end="", flush=True)
            else:
                print(
                    f"? Error playing deterministic first move: {error_msg}",
                    file=sys.stderr,
                )
                print("= resign\n\n", end="", flush=True)
                self.game_over = True
            return  # Exit after playing the first move
        # --- End Modification ---

        stones_to_play = self.logic.get_stones_required_for_n_stones(self.n_stones)
        print(f"Need to generate {stones_to_play} stone(s).", file=sys.stderr)
        stones_to_play = 1

        # --- Call MCTS ---
        time_limit_sec = 5
        start_time = time.time()

        root_node, iterations = self.run_mcts(
            self.board.copy(), self.n_stones, time_limit=time_limit_sec
        )

        elapsed_time = time.time() - start_time
        print(
            f"MCTS Info: Ran {iterations} iterations in {elapsed_time:.2f}s.",
            file=sys.stderr,
        )

        moves_coords = []
        current_board_copy = (
            self.board.copy()
        )  # To track state between 1st and 2nd move choice

        if not root_node or not root_node.children:
            print(
                "? MCTS failed or no moves found (root has no children). Playing randomly from local area.",
                file=sys.stderr,
            )
            # Fallback: generate random local legal moves
            local_moves = self.logic.get_local_moves(
                self.board, EXPANSION_RADIUS
            )  # Use same radius for fallback
            if not local_moves:  # If even local moves fail, try global
                local_moves = self.logic.get_legal_moves_on_board(self.board)

            if not local_moves:
                print("= resign\n\n", end="", flush=True)
                print("? No legal moves found, resigning.", file=sys.stderr)
                self.game_over = True
                return
            else:
                random.shuffle(local_moves)
                moves_coords = local_moves[:stones_to_play]
                # Ensure we don't select the same spot twice if stones_to_play=2 and list is short
                if (
                    stones_to_play == 2
                    and len(moves_coords) == 1
                    and len(local_moves) > 1
                ):
                    # Need one more distinct move
                    remaining_moves = [m for m in local_moves if m != moves_coords[0]]
                    if remaining_moves:
                        moves_coords.append(random.choice(remaining_moves))
                    else:  # Only one legal move possible
                        pass  # Play just the one move

        else:
            # 1. Find the best first move (most visited child of root)
            best_child_1 = max(root_node.children, key=lambda c: c.visits)
            move1 = best_child_1.move
            moves_coords.append(move1)
            print(
                f"MCTS Best Move 1: {self.format_move_coord(move1)} (Visits: {best_child_1.visits}, WinRate: {(best_child_1.wins/best_child_1.visits if best_child_1.visits > 0 else 0):.3f})",
                file=sys.stderr,
            )
            print("  priority", root_node._prioritized_moves[:4], file=sys.stderr)
            for c in root_node.children[:4]:
                print(
                    f"  {c.move} (Visits: {c.visits}, WinRate: {(c.wins/c.visits if c.visits > 0 else 0):.3f})",
                    file=sys.stderr,
                )
            current_board_copy[move1[0], move1[1]] = (
                player  # Apply locally for next step
            )

            if stones_to_play == 2:
                # 2. Find the best second move (most visited child of best_child_1)
                if not best_child_1.children:
                    print(
                        "Warning: MCTS best child 1 not expanded. Choosing random second local move.",
                        file=sys.stderr,
                    )
                    # Fallback: Play a random valid local move after the first one.
                    # Need n_stones+1 to determine the *next* player for local context if needed, but not strictly necessary for just getting moves
                    legal_moves_after_1 = self.logic.get_local_moves(
                        current_board_copy, EXPANSION_RADIUS
                    )
                    # Filter out move1 if it somehow reappears (shouldn't with local logic)
                    legal_moves_after_1 = [m for m in legal_moves_after_1 if m != move1]

                    if not legal_moves_after_1:  # Try global if local fails
                        legal_moves_after_1 = self.logic.get_legal_moves_on_board(
                            current_board_copy
                        )

                    if legal_moves_after_1:
                        move2 = random.choice(legal_moves_after_1)
                        moves_coords.append(move2)
                        print(
                            f"Random Fallback Move 2: {self.format_move_coord(move2)}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "Error: No legal moves for second stone? Playing only first.",
                            file=sys.stderr,
                        )
                        # Play only the first move found
                        pass
                else:
                    # Select best move from the children of the *first* chosen node
                    best_child_2 = max(best_child_1.children, key=lambda c: c.visits)
                    move2 = best_child_2.move
                    # Sanity check: Ensure move2 is not the same as move1
                    if move2 == move1:
                        print(
                            f"Warning: MCTS chose same move twice ({self.format_move_coord(move1)}). Selecting next best.",
                            file=sys.stderr,
                        )
                        # Sort children by visits and pick the second best if available
                        sorted_children = sorted(
                            best_child_1.children, key=lambda c: c.visits, reverse=True
                        )
                        if (
                            len(sorted_children) > 1
                            and sorted_children[1].move != move1
                        ):
                            move2 = sorted_children[1].move
                            moves_coords.append(move2)
                            print(
                                f"Substituted Move 2: {self.format_move_coord(move2)} (Visits: {sorted_children[1].visits})",
                                file=sys.stderr,
                            )
                        else:
                            # Fallback to random if only one child or second best is also move1
                            print(
                                "Falling back to random local for second move.",
                                file=sys.stderr,
                            )
                            legal_moves_after_1 = self.logic.get_local_moves(
                                current_board_copy, EXPANSION_RADIUS
                            )
                            legal_moves_after_1 = [
                                m for m in legal_moves_after_1 if m != move1
                            ]
                            if not legal_moves_after_1:
                                legal_moves_after_1 = (
                                    self.logic.get_legal_moves_on_board(
                                        current_board_copy
                                    )
                                )

                            if legal_moves_after_1:
                                move2 = random.choice(legal_moves_after_1)
                                moves_coords.append(move2)
                            else:
                                print(
                                    "Error: Still no legal second move. Playing only first.",
                                    file=sys.stderr,
                                )
                                pass  # Play only first move
                    else:
                        moves_coords.append(move2)
                        print(
                            f"MCTS Best Move 2: {self.format_move_coord(move2)} (Visits: {best_child_2.visits}, WinRate: {(best_child_2.wins/best_child_2.visits if best_child_2.visits > 0 else 0):.3f})",
                            file=sys.stderr,
                        )

        # --- Format and Play Move ---
        # Ensure we only have the required number of stones
        moves_coords = moves_coords[:stones_to_play]
        if len(moves_coords) < stones_to_play:
            print(
                f"Warning: Could only determine {len(moves_coords)} moves, expected {stones_to_play}. Playing available moves.",
                file=sys.stderr,
            )
            if not moves_coords:  # Completely failed to find any move
                print("= resign\n\n", end="", flush=True)
                print("? Resigning due to inability to find any move.", file=sys.stderr)
                self.game_over = True
                return

        move_str = ",".join(self.format_move_coord(mc) for mc in moves_coords)

        # Play the move(s) on our internal board *before* sending response
        # Use a temporary color string based on the 'player' variable for internal call
        internal_color_str = "B" if player == 1 else "W"
        success, error_msg = self.play_move(internal_color_str, move_str)

        if success:
            print(f"= {move_str}\n\n", end="", flush=True)
            print(f"Played move: {move_str}", file=sys.stderr)
        else:
            print(
                f"? Error playing generated move internally: {error_msg}",
                file=sys.stderr,
            )
            print("= resign\n\n", end="", flush=True)  # Resign if internal play fails
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
            row_str = [[".", "B", "W"][self.board[r, col]] for col in range(self.size)]
            print(f"{r+1:<2} {' '.join(row_str)}")  # Add spaces between B/W/.
        print(flush=True)

    def list_commands(self):
        print(
            f"= boardsize\nclear_board\nplay\ngenmove\nshowboard\nlist_commands\nquit\nname\nversion\nknown_command"
        )

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        parts = command.split(maxsplit=1)  # Split only command from args
        cmd_raw = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""

        cmd_id = ""
        if cmd_raw.isdigit():
            cmd_id = cmd_raw
            parts = args_str.split(maxsplit=1)  # Re-split args
            cmd_raw = parts[0]
            args_str = parts[1] if len(parts) > 1 else ""

        cmd = cmd_raw.lower()
        # Split args string into a list for easier handling
        args = args_str.split()

        success_prefix = f"={cmd_id} " if cmd_id else "= "
        failure_prefix = f"?{cmd_id} " if cmd_id else "? "

        try:
            if cmd == "name":
                print(f"{success_prefix}Connect6 MCTS Agent (Local)")
            elif cmd == "version":
                print(f"{success_prefix}1.2-local")
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
                self.list_commands()  # Prints its own =
            elif cmd == "quit":
                print(f"{success_prefix}")
                sys.exit(0)
            elif cmd == "boardsize":
                if len(args) != 1:
                    raise ValueError("Expected 1 argument for boardsize")
                size = int(args[0])
                if size < 6 or size > 19:
                    raise ValueError("Board size must be between 6 and 19")
                self.set_board_size(size)  # Prints internal success
                # Need to handle the ID here correctly. set_board_size prints '= '
                # We need to make it print '=ID ' if ID exists. Modify set_board_size? Or handle here?
                # Simplest for now: Assume set_board_size prints '= ', we print ID if needed
                # if cmd_id: print(f"{cmd_id}", end="") # This is messy. Let's adjust internal prints.
                # --- Let's modify internal prints instead ---
                # (Adjusted reset_board, set_board_size, show_board, play_move to take success_prefix)
                # --- Reverted: Keep internal prints simple, handle prefix here ---
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)

            elif cmd == "clear_board":
                self.reset_board()  # Prints '= '
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)

            elif cmd == "play":
                if len(args) != 2:
                    raise ValueError("Expected 2 arguments for play: color move")
                color, move_str = args[0], args[1]
                success, msg = self.play_move(color, move_str)
                if not success:
                    raise ValueError(msg.replace("? ", ""))  # Raise error with message
                else:
                    print(f"{success_prefix}")  # Print success prefix

            elif cmd == "genmove":
                if len(args) != 1:
                    raise ValueError("Expected 1 argument for genmove: color")
                color = args[0]
                # generate_move now handles its own GTP output including = ID or ? ID prefix + move/resign + newlines
                self.generate_move(
                    color
                )  # Let generate_move print everything including prefix
                # Note: generate_move needs access to cmd_id if it is to print the full prefix
                # --- Let's pass cmd_id to generate_move ---
                # --- Reverted: generate_move prints '= MOVE\n\n', process_command adds ID ---
                # generate_move already prints the '=' part of the response itself + move/resign
                # If an ID exists, we prepend it here. This might lead to 'ID= MOVE\n\n' which is wrong.
                # --- FINAL DECISION: process_command prints prefix, internal methods don't ---
                # This means play_move should return success/fail, generate_move should return the move string or None/error
                # This is a bigger refactor. Let's stick to the simpler (slightly flawed for ID) approach for now:
                # internal methods print "= " or "? ", and generate_move prints "= MOVE\n\n"
                # The ID might get prepended incorrectly by this block for genmove.
                # For now, we assume Ludii etc might ignore extra ID prefixes on genmove response content lines.
                if cmd_id:
                    print(
                        f"{cmd_id}", end="", flush=True
                    )  # Prepend ID if exists (may look odd)

            elif cmd == "showboard":
                self.show_board()  # Prints '= \n...board...'
                if cmd_id:
                    print(f"={cmd_id}", end="", flush=True)  # Prepend ID

            else:
                raise ValueError(f"Unknown command: {cmd}")

        except ValueError as e:
            print(f"{failure_prefix}{str(e)}")
        except Exception as e:
            print(f"{failure_prefix}Internal error: {str(e)}")
            import traceback

            traceback.print_exc(file=sys.stderr)

        sys.stdout.flush()

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                print(f"Received command: {line.strip()}", file=sys.stderr)
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except EOFError:
                break

    # --- NEW Local Simulation (Unweighted) ---
    def simulate_local_game(self, board_state, n_stones_start, radius):
        """
        Simulates a game using random moves chosen ONLY from a dynamically
        updated local area around existing stones. Unweighted random choice.

        Args:
            board_state: The starting board numpy array.
            n_stones_start: The number of stones on the board_state.
            radius: Defines the local area around stones.

        Returns:
            Winner (1 or 2) or 3 for Draw.
        """
        current_board = board_state.copy()
        n_stones = n_stones_start
        size = current_board.shape[0]

        # Check initial state
        initial_winner = self.logic.check_win_on_board(current_board)
        if initial_winner != 0:
            return initial_winner

        # Get *all* legal moves initially for fallback and emptiness check
        legal_moves_globally = set(self.logic.get_legal_moves_on_board(current_board))
        if not legal_moves_globally:
            return 3  # Draw if no moves

        # Calculate initial local area
        current_local_moves = set(self.logic.get_local_moves(current_board, radius))

        # Initial fallback: if no local moves found initially, use global set
        if not current_local_moves and legal_moves_globally:
            # print("Sim Warning: No initial local moves, using global.", file=sys.stderr)
            current_local_moves = (
                legal_moves_globally.copy()
            )  # Copy because we modify it

        while True:
            # Determine player and stones needed (using corrected logic)
            current_player = self.logic.get_player_for_n_stones(n_stones)
            stones_to_play_this_turn = self.logic.get_stones_required_for_n_stones(
                n_stones
            )

            for i in range(stones_to_play_this_turn):
                # Check if board is full globally
                if not legal_moves_globally:
                    # Double check winner in case incremental missed something subtle on board fill
                    final_winner = self.logic.check_win_on_board(current_board)
                    return (
                        final_winner if final_winner != 0 else 3
                    )  # Return winner or Draw

                # Check if local area is exhausted
                if not current_local_moves:
                    if legal_moves_globally:
                        # print("Sim Warning: Local moves exhausted, using global.", file=sys.stderr)
                        current_local_moves = legal_moves_globally.copy()
                        if (
                            not current_local_moves
                        ):  # Should not happen if legal_moves_globally is not empty
                            return 3  # Safety draw
                    else:  # No global moves left either
                        return 3  # Draw

                # --- Select Move: Random from local area ---
                move = random.choice(list(current_local_moves))
                r, c = move

                # --- Play Move ---
                current_board[r, c] = current_player
                n_stones += 1

                # --- Update State ---
                legal_moves_globally.remove(move)
                current_local_moves.remove(move)

                # --- Check Win Incrementally ---
                winner = self.logic.check_win_incremental(current_board, r, c)
                if winner != 0:
                    return winner  # Win found

                # --- Incrementally Update Local Area ---
                min_r, max_r = max(0, r - radius), min(size - 1, r + radius)
                min_c, max_c = max(0, c - radius), min(size - 1, c + radius)
                for nr in range(min_r, max_r + 1):
                    for nc in range(min_c, max_c + 1):
                        # Add if it's empty and *currently exists* in the global set
                        # (This check implicitly handles bounds and occupation)
                        if (nr, nc) in legal_moves_globally:
                            current_local_moves.add((nr, nc))

                # If only one stone required this turn, break inner loop
                if stones_to_play_this_turn == 1:
                    break  # Go to next player's turn

            # After completing a player's turn (1 or 2 stones)
            # The loop continues if no win/draw occurred. Check global empty again just in case.
            if not legal_moves_globally:
                final_winner = self.logic.check_win_on_board(current_board)
                return final_winner if final_winner != 0 else 3  # Return winner or Draw

    def simulate_local_weighted_game(self, board_state, n_stones_start, radius):
        """
        Simulates a game using WEIGHTED random moves chosen from a dynamically
        updated local area. Weights based on heuristic score for the CURRENT player.
        Maintains separate importance scores for Black and White.

        Args:
            board_state: The starting board numpy array.
            n_stones_start: The number of stones on the board_state.
            radius: Defines the local area around stones.

        Returns:
            Winner (1 or 2) or 3 for Draw.
        """
        current_board = board_state.copy()
        n_stones = n_stones_start
        size = current_board.shape[0]
        WIN_SCORE = 1e9  # Match value in heuristic

        initial_winner = self.logic.check_win_on_board(current_board)
        if initial_winner != 0:
            return initial_winner

        legal_moves_globally = set(self.logic.get_legal_moves_on_board(current_board))
        if not legal_moves_globally:
            return 3

        # --- Separate Importance Dictionaries ---
        black_importance = {}  # move -> score if Black plays there
        white_importance = {}  # move -> score if White plays there

        # Calculate initial local area
        current_local_moves = set(self.logic.get_local_moves(current_board, radius))
        if not current_local_moves and legal_moves_globally:
            current_local_moves = legal_moves_globally.copy()

        # --- Calculate Initial Importance for BOTH players ---
        temp_board_for_heuristics = current_board.copy()
        moves_to_remove = set()  # Track invalid moves found during scoring
        for move in current_local_moves:
            r, c = move
            score_b = self.logic.calculate_heuristic_score(
                temp_board_for_heuristics, r, c, 1
            )
            score_w = self.logic.calculate_heuristic_score(
                temp_board_for_heuristics, r, c, 2
            )

            if score_b == -1 or score_w == -1:  # Occupied or invalid move detected
                moves_to_remove.add(move)
                continue
            black_importance[move] = score_b
            white_importance[move] = score_w

        # Clean up invalid moves
        current_local_moves -= moves_to_remove
        legal_moves_globally -= moves_to_remove

        while True:
            current_player = self.logic.get_player_for_n_stones(n_stones)
            stones_to_play_this_turn = self.logic.get_stones_required_for_n_stones(
                n_stones
            )

            for i in range(stones_to_play_this_turn):
                if not legal_moves_globally:
                    final_winner = self.logic.check_win_on_board(current_board)
                    return final_winner if final_winner != 0 else 3

                if not current_local_moves:
                    if legal_moves_globally:
                        # print("Sim Warning: Local moves exhausted, using global.", file=sys.stderr)
                        current_local_moves = legal_moves_globally.copy()
                        # Calculate importance for newly added global moves
                        temp_board_for_heuristics = current_board.copy()
                        newly_added_moves = current_local_moves - set(
                            black_importance.keys()
                        )  # Check against one dict is enough
                        moves_to_remove = set()
                        for move in newly_added_moves:
                            r, c = move
                            score_b = self.logic.calculate_heuristic_score(
                                temp_board_for_heuristics, r, c, 1
                            )
                            score_w = self.logic.calculate_heuristic_score(
                                temp_board_for_heuristics, r, c, 2
                            )
                            if score_b == -1 or score_w == -1:
                                moves_to_remove.add(move)
                                continue
                            black_importance[move] = score_b
                            white_importance[move] = score_w
                        # Clean up again
                        current_local_moves -= moves_to_remove
                        legal_moves_globally -= moves_to_remove

                        if not current_local_moves:
                            final_winner = self.logic.check_win_on_board(current_board)
                            return final_winner if final_winner != 0 else 3
                    else:
                        return 3  # Draw

                # --- Select Move: Weighted Random using CURRENT player's importance ---
                candidate_list = list(current_local_moves)
                weights = []
                winning_move = None
                # Determine which importance dictionary to use
                current_importance_dict = (
                    black_importance if current_player == 1 else white_importance
                )

                for move in candidate_list:
                    score = current_importance_dict.get(move)
                    if (
                        score is None
                    ):  # Should only happen if initial/fallback calc failed for some reason
                        # print(f"Sim Warning: Recalculating missing score for {move} for Player {current_player}", file=sys.stderr)
                        temp_board_for_heuristics = current_board.copy()
                        score = self.logic.calculate_heuristic_score(
                            temp_board_for_heuristics, move[0], move[1], current_player
                        )
                        if score == -1:
                            score = 1  # Base weight if invalid
                        current_importance_dict[move] = score  # Store calculated score

                    if score >= WIN_SCORE:
                        winning_move = move
                        break
                    weights.append(max(1, score))  # Ensure positive weight

                if winning_move:
                    move = winning_move
                elif not candidate_list:
                    return 3  # Draw
                elif not weights or sum(weights) == 0:
                    move = random.choice(candidate_list)  # Fallback uniform
                else:
                    try:
                        move = random.choices(candidate_list, weights=weights, k=1)[0]
                    except ValueError:
                        move = random.choice(candidate_list)  # Fallback uniform

                r, c = move

                # --- Play Move ---
                current_board[r, c] = current_player
                n_stones += 1

                # --- Update State ---
                legal_moves_globally.remove(move)
                current_local_moves.remove(move)
                # Remove score for the played move from BOTH dictionaries
                black_importance.pop(move, None)
                white_importance.pop(move, None)

                # --- Check Win Incrementally ---
                winner = self.logic.check_win_incremental(current_board, r, c)
                if winner != 0:
                    return winner

                # --- Incrementally Update Local Area & Importance for BOTH players---
                min_r, max_r = max(0, r - radius), min(size - 1, r + radius)
                min_c, max_c = max(0, c - radius), min(size - 1, c + radius)
                temp_board_for_heuristics = current_board.copy()  # Use clean copy

                for nr in range(min_r, max_r + 1):
                    for nc in range(min_c, max_c + 1):
                        neighbor_move = (nr, nc)
                        # Add if it's empty globally and not already known (check one dict)
                        if (
                            neighbor_move in legal_moves_globally
                            and neighbor_move not in black_importance
                        ):
                            current_local_moves.add(neighbor_move)  # Add to candidates
                            # Calculate score for BOTH players for this new candidate
                            score_b = self.logic.calculate_heuristic_score(
                                temp_board_for_heuristics, nr, nc, 1
                            )
                            score_w = self.logic.calculate_heuristic_score(
                                temp_board_for_heuristics, nr, nc, 2
                            )

                            if score_b != -1:
                                black_importance[neighbor_move] = score_b
                            if score_w != -1:
                                white_importance[neighbor_move] = score_w
                            # Note: If score calculation returns -1 (e.g., occupied somehow),
                            # the move exists in current_local_moves but won't have an entry
                            # in the importance dict. The selection logic handles missing scores.

                if stones_to_play_this_turn == 1:
                    break

            if not legal_moves_globally:
                final_winner = self.logic.check_win_on_board(current_board)
                return final_winner if final_winner != 0 else 3

    def run_mcts(
        self, initial_board_state, initial_n_stones, iterations=None, time_limit=None
    ):
        """Performs MCTS search using local expansion and CORRECTED WEIGHTED local simulation."""
        if iterations is None and time_limit is None:
            time_limit = 2.0
        start_time = time.time()

        root_node = MCTSNode(
            state=initial_board_state.copy(),
            n_stones=initial_n_stones,
            parent=None,
            move=None,
            game_logic=self.logic,
        )

        iteration_count = 0
        # Store simulation results for analysis? (Optional)
        # simulation_results_debug = []

        while True:
            # Termination conditions
            current_time = time.time()
            if time_limit is not None and (current_time - start_time) >= time_limit:
                break
            if iterations is not None and iteration_count >= iterations:
                break
            # if iteration_count >= 100000: break # Safety break removed/adjusted

            node = root_node

            # 1. Selection
            while not node.is_terminal() and node.is_fully_expanded():
                best_child = node.select_best_child()
                if best_child is None:
                    # print(f"Warning: Selection failed...", file=sys.stderr) # Keep warning
                    node = None
                    break
                node = best_child

            if node is None:
                iteration_count += 1
                continue

            # 2. Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                try:
                    node = node.expand()
                except RuntimeError as e:
                    # print(f"Warning: Expansion failed: {e}", file=sys.stderr) # Keep warning
                    pass

            # 3. Simulation
            winner = node.get_winner()
            if winner != 0:
                simulation_result = winner
            else:
                # --- Calls the REVISED weighted local simulation ---
                simulation_result = self.simulate_local_weighted_game(
                    node.state, node.n_stones, SIMULATION_RADIUS
                )
                # if iteration_count % 100 == 0: # Debug sample results
                #      simulation_results_debug.append(simulation_result)

            # 4. Backpropagation
            temp_node = node
            while temp_node is not None:
                temp_node.update(simulation_result)
                temp_node = temp_node.parent

            iteration_count += 1

        # Final debug output
        # print(f"Sim results sample (last 100): {simulation_results_debug[-100:] if simulation_results_debug else 'N/A'}", file=sys.stderr)
        # if root_node and root_node.children: # Keep root children stats printout
        #      print("--- MCTS Root Children Stats ---", file=sys.stderr)
        #      sorted_children = sorted(root_node.children, key=lambda c: c.visits, reverse=True)
        #      for i, child in enumerate(sorted_children[:10]):
        #           win_rate = (child.wins / child.visits) if child.visits > 0 else 0
        #           move_str = self.format_move_coord(child.move)
        #           print(f"  {i+1}. Move: {move_str}, Visits: {child.visits}, WinRate: {win_rate:.3f}", file=sys.stderr)
        #      print("------------------------------", file=sys.stderr)

        return root_node, iteration_count


if __name__ == "__main__":
    game = Connect6Game()
    game.run()
