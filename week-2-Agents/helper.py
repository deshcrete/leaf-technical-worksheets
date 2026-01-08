"""
Helper functions and AI classes for Tic Tac Toe agents.
Contains all AI implementations and simulation/training utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from ticTacToe import Model, View, Controller, Player


# =============================================================================
# AI AGENT CLASSES
# =============================================================================

class RandomAI:
    """Simple AI that plays random moves."""

    def __init__(self, model, playerType):
        self.model = model
        self.playerType = playerType

    def getAvailableMoves(self, board):
        """Find all empty cells on the board"""
        moves = []
        for rowIdx, row in enumerate(board):
            for colIdx, cell in enumerate(row):
                if cell == "+":
                    moves.append((rowIdx, colIdx))
        return moves

    def getMove(self, model):
        self.model = model
        board = model.board
        availableMoves = self.getAvailableMoves(board)
        return availableMoves[random.randint(0, len(availableMoves) - 1)]


class RuleBasedAI:
    """AI that uses hard-coded rules: win if possible, block opponent, else random."""

    def __init__(self, model, playerType):
        self.model = model
        self.playerType = playerType
        self.opponentType = "O" if playerType == "X" else "X"

    def getAvailableMoves(self, board):
        """Find all empty cells on the board"""
        moves = []
        for rowIdx, row in enumerate(board):
            for colIdx, cell in enumerate(row):
                if cell == "+":
                    moves.append((rowIdx, colIdx))
        return moves

    def checkWinningMove(self, board, player):
        """
        Check if 'player' can win with one move.
        Returns the winning position (x, y) if found, otherwise None.
        """
        size = len(board)

        # Check each row for a winning opportunity
        for row in range(size):
            cells = [board[row][col] for col in range(size)]
            if cells.count(player) == size - 1 and cells.count("+") == 1:
                col = cells.index("+")
                return (row, col)

        # Check each column for a winning opportunity
        for col in range(size):
            cells = [board[row][col] for row in range(size)]
            if cells.count(player) == size - 1 and cells.count("+") == 1:
                row = cells.index("+")
                return (row, col)

        # Check main diagonal (top-left to bottom-right)
        cells = [board[i][i] for i in range(size)]
        if cells.count(player) == size - 1 and cells.count("+") == 1:
            idx = cells.index("+")
            return (idx, idx)

        # Check anti-diagonal (top-right to bottom-left)
        cells = [board[i][size - 1 - i] for i in range(size)]
        if cells.count(player) == size - 1 and cells.count("+") == 1:
            idx = cells.index("+")
            return (idx, size - 1 - idx)

        return None

    def getMove(self, model):
        self.model = model
        board = model.board

        # Rule 1: If we can win, take the winning move
        winMove = self.checkWinningMove(board, self.playerType)
        if winMove:
            return winMove

        # Rule 2: If opponent can win, block them
        blockMove = self.checkWinningMove(board, self.opponentType)
        if blockMove:
            return blockMove

        # Rule 3: Otherwise, pick a random available move
        availableMoves = self.getAvailableMoves(board)
        return availableMoves[random.randint(0, len(availableMoves) - 1)]


class QLearningAI:
    """AI that learns to play using Q-learning (reinforcement learning)."""

    def __init__(self, model, playerType, epsilon=0.3, learning_rate=0.5):
        self.model = model
        self.playerType = playerType

        # THE Q-TABLE: stores (state, action) → value
        self.q_table = {}

        # EXPLORATION RATE: probability of trying a random move
        self.epsilon = epsilon

        # LEARNING RATE: how much we update Q-values after each game
        self.learning_rate = learning_rate

        # Track moves made during current game
        self.move_history = []

    def boardToString(self, board):
        """Convert board to a string for use as dictionary key."""
        return "|".join(["".join(row) for row in board])

    def getAvailableMoves(self, board):
        """Find all empty cells"""
        moves = []
        for rowIdx, row in enumerate(board):
            for colIdx, cell in enumerate(row):
                if cell == "+":
                    moves.append((rowIdx, colIdx))
        return moves

    def getQValue(self, state, action):
        """Get the Q-value for a (state, action) pair."""
        return self.q_table.get((state, action), 0.0)

    def getMove(self, model):
        """Choose a move using epsilon-greedy strategy."""
        self.model = model
        board = model.board
        state = self.boardToString(board)
        availableMoves = self.getAvailableMoves(board)

        # EXPLORATION: pick a random move
        if random.random() < self.epsilon:
            action = random.choice(availableMoves)
        # EXPLOITATION: pick the move with highest Q-value
        else:
            best_value = float('-inf')
            best_moves = []

            for move in availableMoves:
                q_value = self.getQValue(state, move)
                if q_value > best_value:
                    best_value = q_value
                    best_moves = [move]
                elif q_value == best_value:
                    best_moves.append(move)

            action = random.choice(best_moves)

        self.move_history.append((state, action))
        return action

    def learn(self, reward):
        """Update Q-values after a game ends."""
        for (state, action) in self.move_history:
            old_value = self.getQValue(state, action)
            new_value = old_value + self.learning_rate * (reward - old_value)
            self.q_table[(state, action)] = new_value
        self.move_history = []

    def decayEpsilon(self, decay_rate=0.99):
        """Reduce exploration over time."""
        self.epsilon = self.epsilon * decay_rate


class RLHF_AI:
    """AI that learns from human feedback comparisons."""

    def __init__(self, model, playerType, learning_rate=0.3):
        self.model = model
        self.playerType = playerType
        self.preference_table = {}
        self.learning_rate = learning_rate
        self.comparisons_made = 0

    def boardToString(self, board):
        """Convert board to string for use as dictionary key"""
        return "|".join(["".join(row) for row in board])

    def getAvailableMoves(self, board):
        """Find all empty cells"""
        moves = []
        for rowIdx, row in enumerate(board):
            for colIdx, cell in enumerate(row):
                if cell == "+":
                    moves.append((rowIdx, colIdx))
        return moves

    def getPreference(self, state, action):
        """Get preference score for a (state, action) pair"""
        return self.preference_table.get((state, action), 0.0)

    def displayBoard(self, board):
        """Display board state for human review"""
        print("    0   1   2")
        print("  +---+---+---+")
        for idx, row in enumerate(board):
            print(f"{idx} | {' | '.join(row)} |")
            print("  +---+---+---+")

    def displayComparison(self, board, move_a, move_b):
        """Show the human two possible moves to compare"""
        print("\n" + "="*50)
        print("CURRENT BOARD:")
        self.displayBoard(board)

        print(f"\nYou are playing as: {self.playerType}")
        print("\nWhich move is BETTER?")
        print(f"  [A] Play at position {move_a} (row={move_a[0]}, col={move_a[1]})")
        print(f"  [B] Play at position {move_b} (row={move_b[0]}, col={move_b[1]})")
        print("="*50)

    def collectHumanPreference(self, board, move_a, move_b):
        """Ask human to compare two moves."""
        self.displayComparison(board, move_a, move_b)

        while True:
            choice = input("Enter A or B (or 'skip' to skip): ").strip().upper()
            if choice == 'A':
                return (move_a, move_b)
            elif choice == 'B':
                return (move_b, move_a)
            elif choice == 'SKIP':
                return (None, None)
            else:
                print("Invalid input. Please enter A, B, or 'skip'")

    def updateFromComparison(self, state, preferred, rejected):
        """Update preference scores based on human choice."""
        if preferred is None:
            return

        old_pref = self.getPreference(state, preferred)
        self.preference_table[(state, preferred)] = old_pref + self.learning_rate

        old_rej = self.getPreference(state, rejected)
        self.preference_table[(state, rejected)] = old_rej - self.learning_rate

        self.comparisons_made += 1

    def train(self, num_comparisons=10):
        """Interactive training session with human feedback."""
        print("\n" + "="*50)
        print("RLHF TRAINING SESSION")
        print("="*50)
        print("You will be shown board positions with two possible moves.")
        print("Pick which move you think is BETTER.")
        print("Your preferences will train the AI!\n")

        for i in range(num_comparisons):
            print(f"\n--- Comparison {i+1}/{num_comparisons} ---")

            board = [["+" for _ in range(3)] for _ in range(3)]
            num_moves = random.randint(0, 5)

            pieces = ["X", "O"]
            for j in range(num_moves):
                available = self.getAvailableMoves(board)
                if available:
                    r, c = random.choice(available)
                    board[r][c] = pieces[j % 2]

            available = self.getAvailableMoves(board)
            if len(available) < 2:
                print("(Skipping - not enough moves available)")
                continue

            move_a, move_b = random.sample(available, 2)

            state = self.boardToString(board)
            preferred, rejected = self.collectHumanPreference(board, move_a, move_b)

            self.updateFromComparison(state, preferred, rejected)

        print("\n" + "="*50)
        print(f"Training complete! Made {self.comparisons_made} comparisons.")
        print(f"Preference table size: {len(self.preference_table)} entries")
        print("="*50)

    def getMove(self, model):
        """Choose best move according to learned human preferences."""
        self.model = model
        board = model.board
        state = self.boardToString(board)
        available = self.getAvailableMoves(board)

        best_score = float('-inf')
        best_moves = []

        for move in available:
            score = self.getPreference(state, move)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)


# =============================================================================
# SIMULATION & TRAINING UTILITIES
# =============================================================================

class BiasedHumanSimulator:
    """
    Simulates a human with a BIAS toward center/corner moves.
    Demonstrates reward misspecification.
    """

    def __init__(self, bias_type="center_corner"):
        self.bias_type = bias_type
        self.preferred = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]

    def choose(self, move_a, move_b):
        """Pick between two moves based on bias, not strategy."""
        a_preferred = move_a in self.preferred
        b_preferred = move_b in self.preferred

        if a_preferred and not b_preferred:
            return (move_a, move_b)
        elif b_preferred and not a_preferred:
            return (move_b, move_a)
        else:
            if random.random() < 0.5:
                return (move_a, move_b)
            else:
                return (move_b, move_a)


def trainQLearning(num_games=5000, epsilon=0.5):
    """Train Q-Learning AI by playing against RandomAI"""

    q_ai = QLearningAI(model=None, playerType="X", epsilon=epsilon, learning_rate=0.5)

    results = {"wins": 0, "losses": 0, "ties": 0}
    win_rates = []

    for game_num in range(num_games):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)

        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = q_ai.getMove(model)
            else:
                x, y = opponent.getMove(model)

            model.makeMove(x, y)
            model.checkGameOver()

            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            q_ai.learn(reward=1.0)
            results["wins"] += 1
        elif model.winner == "O":
            q_ai.learn(reward=-1.0)
            results["losses"] += 1
        else:
            q_ai.learn(reward=0.5)
            results["ties"] += 1

        q_ai.decayEpsilon(decay_rate=0.999)

        if (game_num + 1) % 100 == 0:
            recent_wr = results["wins"] / (game_num + 1) * 100
            win_rates.append(recent_wr)

    total = num_games
    print(f"Training complete! Results after {num_games} games:")
    print(f"  Wins:   {results['wins']} ({results['wins']/total*100:.1f}%)")
    print(f"  Losses: {results['losses']} ({results['losses']/total*100:.1f}%)")
    print(f"  Ties:   {results['ties']} ({results['ties']/total*100:.1f}%)")
    print(f"  Q-table size: {len(q_ai.q_table)} state-action pairs learned")
    print(f"  Final epsilon: {q_ai.epsilon:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(range(100, num_games + 1, 100), win_rates)
    plt.xlabel("Games Played")
    plt.ylabel("Win Rate (%)")
    plt.title("Q-Learning AI: Win Rate Over Training")
    plt.grid(True)
    plt.show()

    return q_ai


def demonstrateRewardMisspecification(num_comparisons=200):
    """
    Train RLHF AI with a biased human simulator.
    Demonstrates how reward misspecification affects learning.
    """
    print("="*60)
    print("REWARD MISSPECIFICATION DEMO")
    print("="*60)
    print("\nTraining an AI with a BIASED human who prefers center/corners...")
    print("The human doesn't consider whether the move actually helps WIN.\n")

    rlhf_ai = RLHF_AI(model=None, playerType="X")
    biased_human = BiasedHumanSimulator()

    for i in range(num_comparisons):
        board = [["+" for _ in range(3)] for _ in range(3)]
        num_moves = random.randint(0, 5)

        pieces = ["X", "O"]
        for j in range(num_moves):
            available = rlhf_ai.getAvailableMoves(board)
            if available:
                r, c = random.choice(available)
                board[r][c] = pieces[j % 2]

        available = rlhf_ai.getAvailableMoves(board)
        if len(available) < 2:
            continue

        move_a, move_b = random.sample(available, 2)
        state = rlhf_ai.boardToString(board)

        preferred, rejected = biased_human.choose(move_a, move_b)
        rlhf_ai.updateFromComparison(state, preferred, rejected)

    print(f"Training complete! {rlhf_ai.comparisons_made} comparisons made.\n")

    print("What the AI learned (preference scores):")
    print("  Center (1,1): HIGHLY preferred")
    print("  Corners: preferred")
    print("  Edges: NOT preferred")
    print("\nThis is REWARD MISSPECIFICATION - the AI learned our style,")
    print("not how to win!\n")

    print("Testing biased RLHF AI vs RandomAI (500 games)...")
    wins = 0
    losses = 0
    ties = 0

    for _ in range(500):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)
        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = rlhf_ai.getMove(model)
            else:
                x, y = opponent.getMove(model)

            model.makeMove(x, y)
            model.checkGameOver()
            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            wins += 1
        elif model.winner == "O":
            losses += 1
        else:
            ties += 1

    print(f"\nResults:")
    print(f"  Wins:   {wins} ({wins/5:.1f}%)")
    print(f"  Losses: {losses} ({losses/5:.1f}%)")
    print(f"  Ties:   {ties} ({ties/5:.1f}%)")

    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("The AI plays 'stylishly' (center/corners) but not OPTIMALLY.")
    print("It learned what we REWARDED, not what we actually WANTED (winning).")
    print("This is why specifying rewards correctly is so important in AI!")
    print("="*60)

    return rlhf_ai


def compareMoveDitributions(num_training_games=3000, num_test_games=500):
    """
    Train both Q-Learning and RLHF (with biased human), then compare
    which positions they prefer to play.
    """

    print("="*70)
    print("MOVE DISTRIBUTION COMPARISON: Q-Learning vs Biased RLHF")
    print("="*70)

    # Train Q-Learning AI
    print("\n[1/4] Training Q-Learning AI (learns from game outcomes)...")

    q_ai = QLearningAI(model=None, playerType="X", epsilon=0.5, learning_rate=0.5)

    for _ in range(num_training_games):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)
        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = q_ai.getMove(model)
            else:
                x, y = opponent.getMove(model)
            model.makeMove(x, y)
            model.checkGameOver()
            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            q_ai.learn(reward=1.0)
        elif model.winner == "O":
            q_ai.learn(reward=-1.0)
        else:
            q_ai.learn(reward=0.5)
        q_ai.decayEpsilon(decay_rate=0.999)

    q_ai.epsilon = 0
    print(f"   Q-Learning trained! Q-table size: {len(q_ai.q_table)}")

    # Train RLHF AI
    print("\n[2/4] Training RLHF AI (learns from biased human who loves center/corners)...")

    rlhf_ai = RLHF_AI(model=None, playerType="X", learning_rate=0.3)
    biased_human = BiasedHumanSimulator()

    for _ in range(500):
        board = [["+" for _ in range(3)] for _ in range(3)]
        num_moves = random.randint(0, 5)
        pieces = ["X", "O"]
        for j in range(num_moves):
            available = rlhf_ai.getAvailableMoves(board)
            if available:
                r, c = random.choice(available)
                board[r][c] = pieces[j % 2]

        available = rlhf_ai.getAvailableMoves(board)
        if len(available) < 2:
            continue

        move_a, move_b = random.sample(available, 2)
        state = rlhf_ai.boardToString(board)
        preferred, rejected = biased_human.choose(move_a, move_b)
        rlhf_ai.updateFromComparison(state, preferred, rejected)

    print(f"   RLHF trained! {rlhf_ai.comparisons_made} comparisons made.")

    # Collect move distributions
    print(f"\n[3/4] Playing {num_test_games} test games with each AI...")

    q_moves = np.zeros((3, 3))
    rlhf_moves = np.zeros((3, 3))

    # Q-Learning test games
    q_wins, q_losses, q_ties = 0, 0, 0
    for _ in range(num_test_games):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)
        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = q_ai.getMove(model)
                q_moves[x][y] += 1
            else:
                x, y = opponent.getMove(model)
            model.makeMove(x, y)
            model.checkGameOver()
            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            q_wins += 1
        elif model.winner == "O":
            q_losses += 1
        else:
            q_ties += 1

    # RLHF test games
    rlhf_wins, rlhf_losses, rlhf_ties = 0, 0, 0
    for _ in range(num_test_games):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)
        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = rlhf_ai.getMove(model)
                rlhf_moves[x][y] += 1
            else:
                x, y = opponent.getMove(model)
            model.makeMove(x, y)
            model.checkGameOver()
            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            rlhf_wins += 1
        elif model.winner == "O":
            rlhf_losses += 1
        else:
            rlhf_ties += 1

    # Visualize results
    print("\n[4/4] Creating visualizations...")

    q_moves_pct = q_moves / q_moves.sum() * 100
    rlhf_moves_pct = rlhf_moves / rlhf_moves.sum() * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Q-Learning heatmap
    im1 = axes[0].imshow(q_moves_pct, cmap='Blues', vmin=0, vmax=max(q_moves_pct.max(), rlhf_moves_pct.max()))
    axes[0].set_title(f'Q-Learning Move Distribution\nWins: {q_wins/num_test_games*100:.1f}%', fontsize=12)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f'{q_moves_pct[i,j]:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_xticklabels(['Col 0', 'Col 1', 'Col 2'])
    axes[0].set_yticklabels(['Row 0', 'Row 1', 'Row 2'])

    # Plot 2: RLHF heatmap
    im2 = axes[1].imshow(rlhf_moves_pct, cmap='Reds', vmin=0, vmax=max(q_moves_pct.max(), rlhf_moves_pct.max()))
    axes[1].set_title(f'RLHF (Biased) Move Distribution\nWins: {rlhf_wins/num_test_games*100:.1f}%', fontsize=12)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{rlhf_moves_pct[i,j]:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(['Col 0', 'Col 1', 'Col 2'])
    axes[1].set_yticklabels(['Row 0', 'Row 1', 'Row 2'])

    # Plot 3: Bar chart comparison
    positions = ['Corners\n(0,0)(0,2)\n(2,0)(2,2)', 'Edges\n(0,1)(1,0)\n(1,2)(2,1)', 'Center\n(1,1)']

    q_corners = q_moves_pct[0,0] + q_moves_pct[0,2] + q_moves_pct[2,0] + q_moves_pct[2,2]
    q_edges = q_moves_pct[0,1] + q_moves_pct[1,0] + q_moves_pct[1,2] + q_moves_pct[2,1]
    q_center = q_moves_pct[1,1]

    rlhf_corners = rlhf_moves_pct[0,0] + rlhf_moves_pct[0,2] + rlhf_moves_pct[2,0] + rlhf_moves_pct[2,2]
    rlhf_edges = rlhf_moves_pct[0,1] + rlhf_moves_pct[1,0] + rlhf_moves_pct[1,2] + rlhf_moves_pct[2,1]
    rlhf_center = rlhf_moves_pct[1,1]

    x = np.arange(3)
    width = 0.35

    bars1 = axes[2].bar(x - width/2, [q_corners, q_edges, q_center], width, label='Q-Learning', color='steelblue')
    bars2 = axes[2].bar(x + width/2, [rlhf_corners, rlhf_edges, rlhf_center], width, label='RLHF (Biased)', color='indianred')

    axes[2].set_ylabel('% of Total Moves')
    axes[2].set_title('Move Type Comparison', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(positions)
    axes[2].legend()
    axes[2].set_ylim(0, 60)

    for bar in bars1:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<25} {'Q-Learning':>15} {'RLHF (Biased)':>15}")
    print("-"*55)
    print(f"{'Win Rate':<25} {q_wins/num_test_games*100:>14.1f}% {rlhf_wins/num_test_games*100:>14.1f}%")
    print(f"{'Center moves':<25} {q_center:>14.1f}% {rlhf_center:>14.1f}%")
    print(f"{'Corner moves':<25} {q_corners:>14.1f}% {rlhf_corners:>14.1f}%")
    print(f"{'Edge moves':<25} {q_edges:>14.1f}% {rlhf_edges:>14.1f}%")

    return q_ai, rlhf_ai


# =============================================================================
# HUMAN VS AI GAME CONTROL FLOW
# =============================================================================

def displayBoard(board):
    """Display the current board state."""
    print("\n    0   1   2")
    print("  +---+---+---+")
    for idx, row in enumerate(board):
        print(f"{idx} | {' | '.join(row)} |")
        print("  +---+---+---+")


def getHumanMove(board):
    """Get a valid move from the human player."""
    available = []
    for rowIdx, row in enumerate(board):
        for colIdx, cell in enumerate(row):
            if cell == "+":
                available.append((rowIdx, colIdx))

    while True:
        try:
            move_input = input("Enter your move (row,col) e.g. '1,2': ").strip()
            row, col = map(int, move_input.split(','))

            if (row, col) in available:
                return (row, col)
            else:
                print(f"Invalid move! Position ({row},{col}) is not available.")
                print(f"Available moves: {available}")
        except (ValueError, IndexError):
            print("Invalid input! Please enter row,col (e.g., '1,2')")


def playAgainstAI(ai_type="random", grid_size=3):
    """
    Play a game against an AI opponent.

    Parameters:
    -----------
    ai_type : str
        Type of AI opponent: "random", "rule_based", "q_learning", or "rlhf"
    grid_size : int
        Size of the game board (default 3 for standard tic-tac-toe)

    Returns:
    --------
    str : The winner ("X", "O", or "Tie")
    """

    # Initialize the game
    board = [["+" for _ in range(grid_size)] for _ in range(grid_size)]
    playerX = Player("X")
    playerO = Player("O")
    model = Model(board, "X", playerX, playerO)

    # Create AI opponent based on type
    if ai_type == "random":
        ai = RandomAI(model=None, playerType="O")
        ai_name = "Random AI"
    elif ai_type == "rule_based":
        ai = RuleBasedAI(model=None, playerType="O")
        ai_name = "Rule-Based AI"
    elif ai_type == "q_learning":
        ai = QLearningAI(model=None, playerType="O", epsilon=0)  # No exploration
        ai_name = "Q-Learning AI (untrained)"
    elif ai_type == "rlhf":
        ai = RLHF_AI(model=None, playerType="O")
        ai_name = "RLHF AI (untrained)"
    else:
        print(f"Unknown AI type: {ai_type}. Using Random AI.")
        ai = RandomAI(model=None, playerType="O")
        ai_name = "Random AI"

    print("\n" + "="*50)
    print(f"TIC TAC TOE: You (X) vs {ai_name} (O)")
    print("="*50)
    print("You are X and will go first.")
    print("Enter moves as 'row,col' (e.g., '0,0' for top-left)")

    # Game loop
    while not model.gameOver:
        displayBoard(model.board)

        if model.turn == "X":
            # Human's turn
            print(f"\nYour turn (X):")
            x, y = getHumanMove(model.board)
            print(f"You played at ({x}, {y})")
        else:
            # AI's turn
            print(f"\n{ai_name}'s turn (O)...")
            x, y = ai.getMove(model)
            print(f"{ai_name} played at ({x}, {y})")

        model.makeMove(x, y)
        model.checkGameOver()

        if not model.gameOver:
            model.turn = "O" if model.turn == "X" else "X"

    # Game over
    displayBoard(model.board)
    print("\n" + "="*50)

    if model.winner == "X":
        print("Congratulations! YOU WIN!")
        result = "X"
    elif model.winner == "O":
        print(f"{ai_name} WINS! Better luck next time.")
        result = "O"
    else:
        print("It's a TIE!")
        result = "Tie"

    print("="*50)
    return result


def measurePerformance(ai_type="random", num_games=100, player_ai_type="random"):
    """
    Measure AI performance by playing multiple games.

    Parameters:
    -----------
    ai_type : str
        Type of AI to measure: "random", "rule_based", "q_learning"
    num_games : int
        Number of games to play
    player_ai_type : str
        Type of AI opponent to play against (default: "random")

    Returns:
    --------
    dict : Results with wins, losses, ties, and percentages
    """

    print(f"\nMeasuring {ai_type} AI performance over {num_games} games...")
    print(f"Playing against: {player_ai_type} AI\n")

    # Create the AI to test (plays as X)
    if ai_type == "random":
        test_ai = RandomAI(model=None, playerType="X")
    elif ai_type == "rule_based":
        test_ai = RuleBasedAI(model=None, playerType="X")
    elif ai_type == "q_learning":
        test_ai = QLearningAI(model=None, playerType="X", epsilon=0)
    else:
        test_ai = RandomAI(model=None, playerType="X")

    results = {"wins": 0, "losses": 0, "ties": 0}

    for game_num in range(num_games):
        # Initialize game
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)

        # Create opponent
        if player_ai_type == "random":
            opponent = RandomAI(model=None, playerType="O")
        elif player_ai_type == "rule_based":
            opponent = RuleBasedAI(model=None, playerType="O")
        else:
            opponent = RandomAI(model=None, playerType="O")

        # Play the game
        while not model.gameOver:
            if model.turn == "X":
                x, y = test_ai.getMove(model)
            else:
                x, y = opponent.getMove(model)

            model.makeMove(x, y)
            model.checkGameOver()

            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        # Record result
        if model.winner == "X":
            results["wins"] += 1
        elif model.winner == "O":
            results["losses"] += 1
        else:
            results["ties"] += 1

        # Progress indicator
        if (game_num + 1) % (num_games // 10) == 0:
            print(f"  Progress: {game_num + 1}/{num_games} games completed")

    # Calculate percentages
    total = num_games
    results["win_pct"] = results["wins"] / total * 100
    results["loss_pct"] = results["losses"] / total * 100
    results["tie_pct"] = results["ties"] / total * 100

    # Print results
    print("\n" + "="*50)
    print(f"PERFORMANCE RESULTS: {ai_type.upper()} AI")
    print("="*50)
    print(f"  Games played: {num_games}")
    print(f"  Opponent: {player_ai_type} AI")
    print("-"*50)
    print(f"  Wins:   {results['wins']:>5} ({results['win_pct']:.1f}%)")
    print(f"  Losses: {results['losses']:>5} ({results['loss_pct']:.1f}%)")
    print(f"  Ties:   {results['ties']:>5} ({results['tie_pct']:.1f}%)")
    print("="*50)

    return results


def runPerformanceBenchmark(num_games=500, training_games=3000):
    """
    Run a comprehensive benchmark comparing all AI types.

    Parameters:
    -----------
    num_games : int
        Number of games each AI plays for testing
    training_games : int
        Number of games to train Q-learning AI (default 3000)
    """

    print("\n" + "="*70)
    print("COMPREHENSIVE AI PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Each AI will play {num_games} games as X against Random AI (O)\n")

    all_results = {}

    # Test Random AI
    results = measurePerformance("random", num_games, "random")
    all_results["random"] = results
    print()

    # Test Rule-Based AI
    results = measurePerformance("rule_based", num_games, "random")
    all_results["rule_based"] = results
    print()

    # Train and test Q-Learning AI
    print("="*50)
    print("TRAINING Q-LEARNING AI")
    print("="*50)
    print(f"Training Q-Learning AI for {training_games} games...")

    q_ai = QLearningAI(model=None, playerType="X", epsilon=0.5, learning_rate=0.5)

    for game_num in range(training_games):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)
        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = q_ai.getMove(model)
            else:
                x, y = opponent.getMove(model)
            model.makeMove(x, y)
            model.checkGameOver()
            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            q_ai.learn(reward=1.0)
        elif model.winner == "O":
            q_ai.learn(reward=-1.0)
        else:
            q_ai.learn(reward=0.5)
        q_ai.decayEpsilon(decay_rate=0.999)

        if (game_num + 1) % (training_games // 5) == 0:
            print(f"  Training progress: {game_num + 1}/{training_games} games")

    q_ai.epsilon = 0  # No exploration during testing
    print(f"Training complete! Q-table size: {len(q_ai.q_table)}\n")

    # Test trained Q-Learning AI
    print(f"\nMeasuring Q-Learning (trained) AI performance over {num_games} games...")
    print(f"Playing against: random AI\n")

    q_results = {"wins": 0, "losses": 0, "ties": 0}

    for game_num in range(num_games):
        board = [["+" for _ in range(3)] for _ in range(3)]
        playerX = Player("X")
        playerO = Player("O")
        model = Model(board, "X", playerX, playerO)
        opponent = RandomAI(model=None, playerType="O")

        while not model.gameOver:
            if model.turn == "X":
                x, y = q_ai.getMove(model)
            else:
                x, y = opponent.getMove(model)
            model.makeMove(x, y)
            model.checkGameOver()
            if not model.gameOver:
                model.turn = "O" if model.turn == "X" else "X"

        if model.winner == "X":
            q_results["wins"] += 1
        elif model.winner == "O":
            q_results["losses"] += 1
        else:
            q_results["ties"] += 1

        if (game_num + 1) % (num_games // 10) == 0:
            print(f"  Progress: {game_num + 1}/{num_games} games completed")

    q_results["win_pct"] = q_results["wins"] / num_games * 100
    q_results["loss_pct"] = q_results["losses"] / num_games * 100
    q_results["tie_pct"] = q_results["ties"] / num_games * 100

    print("\n" + "="*50)
    print("PERFORMANCE RESULTS: Q-LEARNING (TRAINED) AI")
    print("="*50)
    print(f"  Games played: {num_games}")
    print(f"  Opponent: random AI")
    print("-"*50)
    print(f"  Wins:   {q_results['wins']:>5} ({q_results['win_pct']:.1f}%)")
    print(f"  Losses: {q_results['losses']:>5} ({q_results['loss_pct']:.1f}%)")
    print(f"  Ties:   {q_results['ties']:>5} ({q_results['tie_pct']:.1f}%)")
    print("="*50)

    all_results["q_learning (trained)"] = q_results

    # Summary comparison
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'AI Type':<25} {'Win %':>10} {'Loss %':>10} {'Tie %':>10}")
    print("-"*55)
    for ai_type, results in all_results.items():
        print(f"{ai_type:<25} {results['win_pct']:>9.1f}% {results['loss_pct']:>9.1f}% {results['tie_pct']:>9.1f}%")
    print("="*70)

    return all_results, q_ai
