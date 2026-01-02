import random
import numpy as np
import matplotlib.pyplot as plt
from ticTacToe import Player, Model

class RandomAI:
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
    def __init__(self, model, playerType):
        self.model = model
        self.playerType = playerType
        # Set opponent type (if we're X, opponent is O and vice versa)
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
        
        # Rule 1: If we can win, take the winning move!
        winMove = self.checkWinningMove(board, self.playerType)
        if winMove:
            return winMove
        
        # Rule 2: If opponent can win, block them!
        blockMove = self.checkWinningMove(board, self.opponentType)
        if blockMove:
            return blockMove
        
        # Rule 3: Otherwise, pick a random available move
        availableMoves = self.getAvailableMoves(board)
        return availableMoves[random.randint(0, len(availableMoves) - 1)]


class MinimaxAI:
    def __init__(self, model, playerType):
        self.model = model
        self.playerType = playerType  # The AI's symbol ("X" or "O")
        # The opponent's symbol
        self.opponentType = "O" if playerType == "X" else "X"
    
    def getAvailableMoves(self, board):
        """Find all empty cells where a move can be made"""
        moves = []
        for rowIdx, row in enumerate(board):
            for colIdx, cell in enumerate(row):
                if cell == "+":
                    moves.append((rowIdx, colIdx))
        return moves
    
    def checkWinner(self, board):
        """
        Check if someone has won the game.
        Returns: "X", "O", "Tie", or None (game still ongoing)
        """
        size = len(board)
        
        # Check rows
        for row in board:
            if row[0] != "+" and all(cell == row[0] for cell in row):
                return row[0]
        
        # Check columns
        for col in range(size):
            if board[0][col] != "+" and all(board[row][col] == board[0][col] for row in range(size)):
                return board[0][col]
        
        # Check main diagonal
        if board[0][0] != "+" and all(board[i][i] == board[0][0] for i in range(size)):
            return board[0][0]
        
        # Check anti-diagonal
        if board[0][size-1] != "+" and all(board[i][size-1-i] == board[0][size-1] for i in range(size)):
            return board[0][size-1]
        
        # Check for tie (board full, no winner)
        if all(cell != "+" for row in board for cell in row):
            return "Tie"
        
        # Game still ongoing
        return None
    
    def minimax(self, board, depth, isMaximizing):
        winner = self.checkWinner(board)
        
        if winner == self.playerType:
            return 10 - depth
        
        elif winner == self.opponentType:
            return depth - 10
        
        elif winner == "Tie":
            return 0
        
        if isMaximizing:
            bestScore = float('-inf')
            for (row, col) in self.getAvailableMoves(board):
                board[row][col] = self.playerType
                score = self.minimax(board, depth + 1, False)
                board[row][col] = "+"
                bestScore = max(score, bestScore)
            
            return bestScore
        
        else:
            bestScore = float('inf')
            for (row, col) in self.getAvailableMoves(board):
                board[row][col] = self.opponentType
                score = self.minimax(board, depth + 1, True)
                board[row][col] = "+"
                bestScore = min(score, bestScore)
            
            return bestScore
    
    def getMove(self, model):
        self.model = model
        board = model.board
        
        bestScore = float('-inf')
        bestMove = None
        
        # Try each available move and see which gives the best outcome
        for (row, col) in self.getAvailableMoves(board):
            # Make the move
            board[row][col] = self.playerType
            
            # Get the minimax score for this move
            # We pass False because after our move, it's opponent's turn (minimizing)
            score = self.minimax(board, 0, False)
            
            # Undo the move
            board[row][col] = "+"
            
            # Track the best move found so far
            if score > bestScore:
                bestScore = score
                bestMove = (row, col)
        
        return bestMove

# =============================================================================
# MINIMAX WITH CUSTOM SCORING FUNCTIONS - Demonstrating Reward Misspecification
# =============================================================================
#
# THE KEY INSIGHT:
# The Minimax algorithm is PERFECT at optimizing whatever score you give it.
# But if your scoring function is WRONG, you get unintended behavior!
#
# These scoring functions are INTENTIONALLY EXTREME to show clear differences:
# - AGGRESSIVE: Cares MORE about position than winning
# - DEFENSIVE: Actually PREFERS ties over wins
# - PACIFIST: Winning is PENALIZED!
#
# =============================================================================

# -----------------------------------------------------------------------------
# SCORING FUNCTIONS - Each represents a different "value system"
# -----------------------------------------------------------------------------

def optimalScoring(board, playerType, opponentType, depth):
    """OPTIMAL: Only cares about winning. This is the correct objective."""
    winner = checkWinnerForScoring(board)
    if winner == playerType:
        return 1000  # Winning is everything
    elif winner == opponentType:
        return -1000  # Losing is terrible
    elif winner == "Tie":
        return 0
    
    # Heuristic: favor positions that lead to wins
    score = 0
    for line in getLines(board):
        our_count = line.count(playerType)
        opp_count = line.count(opponentType)
        if opp_count == 0:
            score += our_count * 10
        if our_count == 0:
            score -= opp_count * 10
    return score


def aggressiveScoring(board, playerType, opponentType, depth):
    """
    AGGRESSIVE: Values controlling center/corners MORE than winning!
    
    MISSPECIFICATION: Position bonuses are SO high that the AI will
    grab a corner instead of taking a winning move or blocking!
    """
    winner = checkWinnerForScoring(board)
    if winner == playerType:
        return 100  # Winning is only worth 100
    elif winner == opponentType:
        return -1000
    elif winner == "Tie":
        return 0
    
    # Position bonuses are HUGE compared to win value!
    score = 0
    if board[1][1] == playerType:  # Center
        score += 200  # Worth MORE than winning!
    if board[1][1] == opponentType:
        score -= 50
    
    corners = [(0,0), (0,2), (2,0), (2,2)]
    for r, c in corners:
        if board[r][c] == playerType:
            score += 150  # Corners worth more than winning!
        elif board[r][c] == opponentType:
            score -= 30
    
    return score


def defensiveScoring(board, playerType, opponentType, depth):
    """
    DEFENSIVE: Prefers ties and blocking over winning!
    
    MISSPECIFICATION: Ties are scored HIGHER than wins!
    The AI will actively choose tie over victory.
    """
    winner = checkWinnerForScoring(board)
    if winner == playerType:
        return 50  # Winning is "meh"
    elif winner == opponentType:
        return -1000
    elif winner == "Tie":
        return 200  # Ties are GREAT! Better than winning!
    
    # Blocking is highly rewarded
    score = 0
    for line in getLines(board):
        opp_count = line.count(opponentType)
        our_count = line.count(playerType)
        if opp_count == 2 and our_count == 1:
            score += 300  # Blocking is worth MORE than winning!
        elif opp_count == 2 and our_count == 0:
            score -= 100
    return score


def pacifistScoring(board, playerType, opponentType, depth):
    """
    PACIFIST: Winning is BAD! The AI actively avoids winning.
    
    EXTREME MISSPECIFICATION: This AI will deliberately NOT take
    winning moves because winning has a NEGATIVE score!
    """
    winner = checkWinnerForScoring(board)
    if winner == playerType:
        return -100  # WINNING IS BAD! (negative score!)
    elif winner == opponentType:
        return -500  # Losing is worse though
    elif winner == "Tie":
        return 300  # Ties are the BEST outcome!
    
    # Avoid creating winning opportunities
    score = 0
    for line in getLines(board):
        our_count = line.count(playerType)
        opp_count = line.count(opponentType)
        # Penalize getting close to winning!
        if our_count == 2 and opp_count == 0:
            score -= 50  # Don't want to be close to winning!
    return score


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def checkWinnerForScoring(board):
    """Check if someone won"""
    size = len(board)
    for row in board:
        if row[0] != "+" and all(cell == row[0] for cell in row):
            return row[0]
    for col in range(size):
        if board[0][col] != "+" and all(board[row][col] == board[0][col] for row in range(size)):
            return board[0][col]
    if board[0][0] != "+" and all(board[i][i] == board[0][0] for i in range(size)):
        return board[0][0]
    if board[0][size-1] != "+" and all(board[i][size-1-i] == board[0][size-1] for i in range(size)):
        return board[0][size-1]
    if all(cell != "+" for row in board for cell in row):
        return "Tie"
    return None


def getLines(board):
    """Get all rows, columns, and diagonals"""
    size = len(board)
    lines = [list(row) for row in board]
    lines += [[board[row][col] for row in range(size)] for col in range(size)]
    lines.append([board[i][i] for i in range(size)])
    lines.append([board[i][size-1-i] for i in range(size)])
    return lines


# -----------------------------------------------------------------------------
# FAST CONFIGURABLE MINIMAX AI (with depth limit)
# -----------------------------------------------------------------------------

class ConfigurableMinimaxAI:
    """
    Minimax AI with configurable scoring function AND depth limit for speed.
    """
    
    def __init__(self, model, playerType, scoring_function=optimalScoring, name="Optimal", max_depth=2):
        self.model = model
        self.playerType = playerType
        self.opponentType = "O" if playerType == "X" else "X"
        self.scoring_function = scoring_function
        self.name = name
        self.max_depth = max_depth
    
    def getAvailableMoves(self, board):
        moves = []
        for rowIdx, row in enumerate(board):
            for colIdx, cell in enumerate(row):
                if cell == "+":
                    moves.append((rowIdx, colIdx))
        return moves
    
    def minimax(self, board, depth, isMaximizing):
        winner = checkWinnerForScoring(board)
        if winner is not None:
            return self.scoring_function(board, self.playerType, self.opponentType, depth)
        
        if depth >= self.max_depth:
            return self.scoring_function(board, self.playerType, self.opponentType, depth)
        
        available = self.getAvailableMoves(board)
        if not available:
            return self.scoring_function(board, self.playerType, self.opponentType, depth)
        
        if isMaximizing:
            bestScore = float('-inf')
            for (row, col) in available:
                board[row][col] = self.playerType
                result = self.minimax(board, depth + 1, False)
                board[row][col] = "+"
                bestScore = max(result, bestScore)
            return bestScore
        else:
            bestScore = float('inf')
            for (row, col) in available:
                board[row][col] = self.opponentType
                result = self.minimax(board, depth + 1, True)
                board[row][col] = "+"
                bestScore = min(result, bestScore)
            return bestScore
    
    def getMove(self, model):
        self.model = model
        board = model.board
        
        bestScore = float('-inf')
        bestMove = None
        
        for (row, col) in self.getAvailableMoves(board):
            board[row][col] = self.playerType
            score = self.minimax(board, 0, False)
            board[row][col] = "+"
            
            if score > bestScore:
                bestScore = score
                bestMove = (row, col)
        
        return bestMove

# =============================================================================
# DEMO: Compare Different Scoring Functions (FAST version)
# =============================================================================
# Same algorithm, different values → different behavior!
# Now with depth-limited search for speed!
# =============================================================================

def compareScoringFunctions(num_games=100):
    """
    Pit each scoring function against RandomAI and compare results.
    Uses depth-limited Minimax for speed (~5 seconds total instead of minutes).
    """
    
    scoring_functions = [
        (optimalScoring, "OPTIMAL", "Only cares about winning"),
        (aggressiveScoring, "AGGRESSIVE", "Loves center/corners"),
        (defensiveScoring, "DEFENSIVE", "Loves blocking, prefers ties"),
        (pacifistScoring, "PACIFIST", "Prefers ties over wins!"),
    ]
    
    print("="*70)
    print("REWARD MISSPECIFICATION IN MINIMAX")
    print("="*70)
    print("\nSame algorithm, different scoring functions → different behavior!")
    print(f"Each AI plays {num_games} games against RandomAI.\n")
    
    results = {}
    
    for scoring_func, name, description in scoring_functions:
        wins, losses, ties = 0, 0, 0
        
        for _ in range(num_games):
            board = [["+" for _ in range(3)] for _ in range(3)]
            playerX = Player("X")
            playerO = Player("O")
            model = Model(board, "X", playerX, playerO)
            
            minimax_ai = ConfigurableMinimaxAI(
                model=None, 
                playerType="X", 
                scoring_function=scoring_func,
                name=name,
                max_depth=3  # Fast!
            )
            opponent = RandomAI(model=None, playerType="O")
            
            while not model.gameOver:
                if model.turn == "X":
                    x, y = minimax_ai.getMove(model)
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
        
        results[name] = {"wins": wins, "losses": losses, "ties": ties}
        
        print(f"┌{'─'*68}┐")
        print(f"│ {name:12} - {description:50} │")
        print(f"├{'─'*68}┤")
        print(f"│   Wins: {wins:3} ({wins/num_games*100:5.1f}%)  │  Losses: {losses:3} ({losses/num_games*100:5.1f}%)  │  Ties: {ties:3} ({ties/num_games*100:5.1f}%)   │")
        print(f"└{'─'*68}┘\n")
    
    print("="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
    The algorithm is PERFECT at optimizing its objective.
    The problem is WE gave it the wrong objective!
    
    - OPTIMAL:    Maximizes wins → best performance
    - AGGRESSIVE: Maximizes position control → may miss strategic plays
    - DEFENSIVE:  Maximizes blocking/ties → avoids taking risks to win
    - PACIFIST:   Maximizes ties → actively avoids winning!
    """)
    print("="*70)
    
    # Visualize
    names = list(results.keys())
    wins = [results[n]["wins"] for n in names]
    ties = [results[n]["ties"] for n in names]
    losses = [results[n]["losses"] for n in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, wins, width, label='Wins', color='green')
    ax.bar(x, ties, width, label='Ties', color='gray')
    ax.bar(x + width, losses, width, label='Losses', color='red')
    
    ax.set_ylabel('Number of Games')
    ax.set_title('Minimax with Different Scoring Functions\n(Same Algorithm, Different Values = Different Behavior)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, num_games)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run the comparison - now takes ~5 seconds instead of minutes!
scoring_results = compareScoringFunctions(num_games=200)