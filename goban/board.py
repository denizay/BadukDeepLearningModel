import os
import time
import random
import numpy as np

## 1: Black, -1: White
class Stone:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y
   
class Board:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size))
        self.groups = []

    def __len__(self):
        return len(self.board)

    def __repr__(self):
        return f"board size: {self.board_size}"

    def print_board(self):
        print("  A B C D E F G H I")
        for i, row in enumerate(self.board):
            row_str = f"{i+1} "
            for j, point in enumerate(row):
                point_str = ""
                if point == 0:
                    if i in [2,4,6] and j in [2,4,6]:
                        point_str = "• "
                    else:
                        point_str = ". "
                elif point == 1:
                    point_str = "X "
                elif point == -1:
                    point_str = "O "
                row_str += point_str
            print(row_str)
        print()

    def put_stone(self, stone):
        self.board[stone.x, stone.y] = stone.color
        row = stone.x
        col = stone.y
        opponent_color = - stone.color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr, nc] == opponent_color:
                    if self.count_liberties(nr, nc) == 0:
                        self.remove_group(nr, nc)
        
        # Check if the placed stone has no liberties (self-capture)
        if self.count_liberties(row, col) == 0:
                self.remove_group(row, col)

    def count_liberties(self, row, col):
        """Count liberties of a stone at (row, col)."""
        color = self.board[row, col]
        visited = set()
        stack = [(row, col)]
        liberties = 0
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            # Check all four directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board.shape[0] and 0 <= nc < self.board.shape[1]:
                    if self.board[nr, nc] == 0:
                        liberties += 1
                    elif self.board[nr, nc] == color and (nr, nc) not in visited:
                        stack.append((nr, nc))
                        
        return liberties

    def remove_group(self, row, col):
        """Remove the group of stones connected to (row, col)."""
        color = self.board[row, col]
        stack = [(row, col)]
        group = []
        
        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.append((r, c))
            
            # Check all four directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board.shape[0] and 0 <= nc < self.board.shape[1]:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        stack.append((nr, nc))
        
        for r, c in group:
            self.board[r, c] = 0

