import numpy as np
import matplotlib.pyplot as plt
import random

class Model:
    def __init__(self, board, turn, playerX, playerO):
        self.board = board
        self.turn = turn
        self.playerX = playerX
        self.playerO = playerO
        self.gameOver = False
        self.winner = None

    def makeMove(self, x, y):
        if self.board[x][y] != "+":
            return False
        
        if self.turn == "X":
            self.playerX.addMove(x, y)
            self.board[x][y] = "X"
        else:
            self.playerO.addMove(x, y) 
            self.board[x][y] = "O"
        return True
    
    def transposeBoard(self):
        rowSize = len(self.board)
        colSize = len(self.board[0])

        cols = []
        for i in range(colSize):
            col = []
            for j in range(rowSize):
                col.append(self.board[j][i])
            cols.append(col)
        return cols
    
    def getDiagonals(self):
        rowSize = len(self.board)
        colSize = len(self.board[0])
        mainDiag = []
        offDiag = []
        
        # Only check diagonals for square boards
        if rowSize == colSize:
            for i in range(rowSize):
                mainDiag.append(self.board[i][i])
                offDiag.append(self.board[i][(colSize - 1) - i])
            return [mainDiag, offDiag]
        else:
            # For non-square boards, return empty lists (no diagonals to check)
            return []
    
    def checkAllSame(self, lists):
        for l in lists:
            lSet = set(l)
            if len(lSet) == 1 and "+" not in lSet:
                self.gameOver = True
                self.winner = list(lSet)[0]
                break 
    
    def isBoardFull(self):
        for row in self.board:
            for cell in row:
                if cell == "+":
                    return False
        return True
            
    def checkGameOver(self):
        #check rows
        self.checkAllSame(self.board)
        #check columns
        self.checkAllSame(self.transposeBoard())
        #check diagonals
        self.checkAllSame(self.getDiagonals())
        
        #check for tie
        if not self.gameOver and self.isBoardFull():
            self.gameOver = True
            self.winner = "Tie"


class Player:
    def __init__(self, type):
        self.type = type
        self.moveHistory = []
    
    def addMove(self, x, y):
        self.moveHistory.append((x,y))
        self.currentPosition = (x,y)

class View:
    def __init__(self, model):
        self.model = model
    def printBoard(self):
        print("    0   1   2")
        print("  +---+---+---+")
        for idx, row in enumerate(self.model.board):
            print(f"{idx} | {' | '.join(row)} |")
            print("  +---+---+---+")

class Controller:
    def __init__(self, boardSize, ai=None):
        self.board = [["+" for i in range(boardSize)] for j in range(boardSize)]
        self.boardSize = boardSize
        self.ai = ai
        if random.randint(0,1) == 0:
            self.currentPlayer = "X"
        else:
            self.currentPlayer = "O"

    def getMove(self, model):
            if self.ai and model.turn == self.ai.playerType:
                x, y = self.ai.getMove(model)
            else:
                xCoord = input("What is the x-position you want to play? ")
                yCoord = input("What is the y-position you want to play? ")
                x = int(xCoord)
                y = int(yCoord)

            return (x, y)

    def playGame(self):
        playerX = Player("X")
        playerO = Player("O")
        model = Model(self.board, self.currentPlayer, playerX, playerO)

        view = View(model)
        view.printBoard()


        while (not model.gameOver):
            print(f"> Current Player: {model.turn}\n")
            
            try:
                (x, y) = self.getMove(model)
                
                if x < 0 or x >= self.boardSize or y < 0 or y >= self.boardSize:
                    print(f"Invalid position! Please enter values between 0 and {self.boardSize - 1}")
                    continue
                
                if not model.makeMove(x, y):
                    print("That position is already occupied! Try again.")
                    continue
                    
            except ValueError:
                print("Invalid input! Please enter numbers only.")
                continue
            
            view = View(model)
            view.printBoard()
            
            model.checkGameOver()
            
            if not model.gameOver:
                if model.turn == "X":
                    model.turn = "O"
                else:
                    model.turn = "X"
        
        print("\n" + "="*30)
        if model.winner == "Tie":
            print("Game Over! It's a TIE!")
        else:
            print(f"Game Over! Player {model.winner} WINS!")
        print("="*30)