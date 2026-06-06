from TicTacToe import TicTacToe
from State import State
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from AI_Agent import AI_Agent

# Save path for the trained Q-table
PATH = 'Data/Q_MC_yossi.pth'

env = TicTacToe(State())

# The agent now plays as O (player value = -1) and moves SECOND.
# X (player value = 1) always moves first — played by the Random agent here.
player1 = Random_Agent(1, env, graphics=None)   # X: random opponent, goes first
player2 = AI_Agent(-1, env, graphics=None, Q_table_PATH=None)  # O: AI agent to train, goes second

gamma = 0.9   # discount factor for future rewards
alpha = 0.1   # MC learning rate


def main():
    num_epochs = 50000  # total training episodes

    for epoch in range(num_epochs):
        # Generate one full game episode; only O's (AI's) transitions are recorded
        episode = Generate_episode(player2, epoch)

        # Monte Carlo update: traverse episode backwards to compute discounted returns
        G = 0  # cumulative discounted return initialised at episode end
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward  # G_t = r_{t+1} + γ·G_{t+1}

            # Every-visit MC: update Q toward observed return
            current_Q = player2.Q.get((state, action), 0)
            player2.Q[(state, action)] = current_Q + alpha * (G - current_Q)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{num_epochs}", end="\r")

    player2.save_Q(PATH)
    print("\nTraining complete. Test results (x_win, o_win, tie):", test(100))


def Generate_episode(player, epoch):
    """
    Plays one full game and records the AI agent's (O's) transitions.

    The AI agent (player2 / O) moves second — state.player == -1 is its turn.
    The random agent (player1 / X) always moves first.

    Returns:
        episods: list of (state, action, reward) from O's perspective.
                 Intermediate rewards are 0; the terminal reward is
                 +1 if O wins, -1 if X wins, 0 for a tie.
    """
    episods = []
    state = State()               # fresh board; state.player starts as 1 (X's turn)
    current_player = player1      # X always opens

    while not env.end_of_game(state):
        # Each agent picks an action on its own turn
        if current_player == player2:
            # AI uses epsilon-greedy to balance exploration and exploitation
            action = player2.get_action(state=state, epoch=epoch)
        else:
            # Random agent picks uniformly among legal moves
            action = player1.get_action(state=state)

        next_state, env_reward = env.next_state(state, action)

        if current_player == player2:
            # Convert environment reward to O's perspective:
            #   env_reward = -1 (O wins) → agent_reward = +1
            #   env_reward =  0 (ongoing / tie) → agent_reward = 0
            # Wins by X are handled below after the loop ends.
            agent_reward = env_reward * (-1)
            episods.append((state, action, agent_reward))

        state = next_state
        current_player = switch_players(current_player)

    # If X won after O's last move, retroactively assign a loss reward to O's
    # last recorded transition (its reward was 0 at the time it was appended).
    if state.end_of_game == 1 and len(episods) > 0:
        last_s, last_a, _ = episods[-1]
        episods[-1] = (last_s, last_a, -1)  # O loses when X wins

    return episods


def test(num):
    """Evaluates the trained O agent against the random X player over `num` games."""
    x_win = 0
    o_win = 0
    tie = 0

    player2.train = False   # switch AI to pure greedy (no exploration)
    player2.load_Q(PATH)

    for n in range(num):
        current_player = player1   # X (Random) always opens
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


def print_episodes(episode):
    for i, e in enumerate(episode):
        print(f'\n i= {i} player = {e[0].player} ')
        for item in e:
            print(item, end=" ")


def switch_players(player):
    if player == player1:
        return player2
    else:
        return player1


if __name__ == '__main__':
    main()
