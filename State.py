import numpy as np

class State:
    def __init__(self, board = None, player = 1):
        if board is not None:
            self.board = board
        else:
            self.board = self.init_board()
        self.player = player
        self.end_of_game = 0

    def init_board(self):
        board = np.zeros((3,3))
        # board[1,1] = 1
        # board[0,0] = -1
        return board
    
    def reset (self):
        self.board = self.init_board()
        self.player = 1
        self.end_of_game = 0

    def switch_players (self):
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def __eq__(self, other) ->bool:
        # b1 = np.equal(self.board, other.board).all()
        return np.equal(self.board, other.board).all()

    def __hash__(self) -> int:
        return hash(tuple(self.board.astype(np.int8).ravel()))
    
    def copy (self):
        newBoard = np.copy(self.board)
        return State(board=newBoard, player=self.player)
    
    def __str__(self) -> str:
        return str(self.board)