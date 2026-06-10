from TicTacToe import TicTacToe
from State import State
from Random_Agent import Random_Agent
from AI_Agent import AI_Agent

# Save path for the trained Q-table
PATH = 'Data/Q_QLearning_yossi.pth'

env = TicTacToe(State())

# The AI agent plays as X (player value = 1) and moves FIRST.
# O (player value = -1) always moves second — played by the Random agent here.
player1 = AI_Agent(1, env, graphics=None, Q_table_PATH=None)   # X: AI agent to train, goes first
player2 = Random_Agent(-1, env, graphics=None)                  # O: random opponent, goes second

# Convenience aliases so the update formula reads closer to textbook notation
Q = player1.Q
get_Q = player1.get_Q

gamma = 0.9   # discount factor for future rewards
alpha = 0.1   # Q-Learning step size


def main():
    num_epochs = 100000  # total training episodes

    for epoch in range(num_epochs):
        state = State()   # fresh board; X always opens

        while not env.end_of_game(state):
            # --- AI (X) selects action using epsilon-greedy policy ---
            action = player1.get_action(state, epoch)

            # Apply X's action and observe the immediate outcome
            next_state, reward = env.next_state(state, action)

            # Terminal: game ended on X's move (X wins or draw)
            # Future Q is 0 because there is no next state to act from.
            if env.end_of_game(next_state):
                Q[(state, action)] = get_Q(state, action) + alpha * (reward - get_Q(state, action))
                break

            # --- Opponent (O) makes a random move ---
            opp_action = player2.get_action(state=next_state)
            after_opp_state, opp_reward = env.next_state(next_state, opp_action)

            # Terminal: game ended on O's move (O wins or draw after opponent acts).
            # opp_reward = -1 when O wins, 0 for draw — both correct from X's viewpoint.
            if env.end_of_game(after_opp_state):
                Q[(state, action)] = get_Q(state, action) + alpha * (opp_reward - get_Q(state, action))
                break

            # --- Q-Learning (off-policy) TD update ---
            # Unlike SARSA, which uses Q(s', a') for the *actual* next action chosen
            # by the behaviour policy, Q-Learning bootstraps from the *best possible*
            # action at s'.  This makes the algorithm off-policy: the target is always
            # the greedy value regardless of how the agent will actually explore.
            #
            # Update rule:  Q(s,a) ← Q(s,a) + α * [r + γ·max_a' Q(s',a') − Q(s,a)]
            #
            # s'  = after_opp_state  (state after both X's move AND O's response)
            # r   = 0                (no intermediate reward; game is still ongoing)
            max_next_Q = max(get_Q(after_opp_state, a) for a in player1.legal_actions(after_opp_state))
            Q[(state, action)] = get_Q(state, action) + alpha * (reward + gamma * max_next_Q - get_Q(state, action))

            state = after_opp_state   # advance to the state where X acts next

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{num_epochs}", end="\r")

    player1.save_Q(PATH)
    print("\nTraining complete. Test results (x_win, o_win, tie):", test(100))


def test(num):
    """Evaluates the trained X agent against the random O player over `num` games."""
    x_win = 0
    o_win = 0
    tie = 0

    player1.train = False   # switch AI to pure greedy (no exploration)
    player1.load_Q(PATH)

    for n in range(num):
        current_player = player1   # X (AI) always opens
        state = State()
        while not env.end_of_game(state):
            if current_player == player1:
                action = player1.get_action(state=state)
            else:
                action = player2.get_action(state=state)
            state, _ = env.next_state(state, action)
            current_player = switch_players(current_player)

        if state.end_of_game == 1:
            x_win += 1
        elif state.end_of_game == -1:
            o_win += 1
        else:
            tie += 1

        state.reset()
        print(n, end="\r")

    return x_win, o_win, tie


def switch_players(player):
    if player == player1:
        return player2
    else:
        return player1


if __name__ == '__main__':
    main()
