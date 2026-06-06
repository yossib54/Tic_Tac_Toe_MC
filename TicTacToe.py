from State import State
import numpy as np

class TicTacToe:
    def __init__(self, state):
        self.state : State = state

    def move (self, action):
        self.state.board[action] = self.state.player
        self.switch_players(self.state)
        self.end_of_game(self.state)
   
    def legal (self, state:State, action):
        if state.board[action]==0:
            return True
        return False
    
    def next_state (self, state: State, action):
        next_state = state.copy()
        next_state.board[action] = state.player
        next_state.switch_players()
        self.end_of_game(next_state)
        if next_state.end_of_game == 2:
            reward = 0
        else:
            reward = next_state.end_of_game
        return next_state, reward

    def end_of_game (self, state: State):
        board = state.board
        row_sum = np.sum(board, axis=1)
        col_sum = np.sum(board, axis=0)
        diagonals = [np.trace(board), np.trace(np.fliplr(board))]
        piece_num =  np.count_nonzero(board)
        
        # print (f'row_sum: {row_sum} col_sum: {col_sum} diagonals: {diagonals} piece_num: {piece_num}')
        if 3 in row_sum or 3 in col_sum or 3 in diagonals:
            state.end_of_game = 1
            return True
        if -3 in row_sum or -3 in col_sum or -3 in diagonals:
            state.end_of_game = -1
            return True
        if piece_num == 9:
            state.end_of_game = 2
            return True
        return False
    
    def switch_players (self, state:  State):
        if state.player == 1:
            state.player = -1
        else:
            state.player = 1