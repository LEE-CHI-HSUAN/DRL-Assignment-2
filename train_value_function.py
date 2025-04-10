import pickle
import random
import numpy as np
from collections import deque, namedtuple


Transition = namedtuple("Transition", ("pre_after_state", "after_state", "reward"))


def td_learning(
    env,
    approximator,
    num_episodes: int = 100000,
    alpha: float = 0.01,
    gamma: float = 0.0,
    epsilon: float = 0.0,
    memory_size: int = 10000,
    batch_size: int = 32,
    log_steps: int = 100,
    save_steps: int = 1000,
):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    replay_memory = deque(maxlen=memory_size)

    for episode in range(num_episodes):
        pre_after_state = env.reset()
        pre_after_state = pre_after_state.copy()
        max_tile = np.max(env.board)

        while True:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # action selection
            action = approximator.get_action(env, legal_moves, epsilon)

            # move
            reward = env.st(action)
            after_state = env.board.copy()
            _, _, done, _ = env.ep()
            max_tile = max(max_tile, np.max(env.board))

            # if put here, memory won't see terminal states
            if done:
                break

            # Store transition to replay memory
            replay_memory.append(
                Transition(
                    pre_after_state=pre_after_state,
                    after_state=after_state,
                    reward=reward,
                )
            )

            # # update
            # delta = (
            #     reward
            #     + gamma * approximator.value(after_state)
            #     - approximator.value(pre_after_state)
            # )
            # approximator.update(pre_after_state, alpha * delta)

            # batch update
            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)
                for tr in batch:
                    delta = (
                        tr.reward
                        + gamma * approximator.value(tr.after_state)
                        - approximator.value(tr.pre_after_state)
                    )
                    approximator.update(tr.pre_after_state, alpha * delta)

            pre_after_state = after_state

        # log
        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % log_steps == 0:
            avg_score = np.mean(final_scores[-log_steps:])
            success_rate = np.mean(success_flags[-log_steps:])
            print(
                f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}"
            )

        # save checkpoint
        if (episode + 1) % save_steps == 0:
            avg_score = np.mean(final_scores[-log_steps:])
            ckpt_name = f"ckpt/value_ckpt_{episode + 1}-{avg_score}.pkl"
            with open(ckpt_name, "wb") as file:
                pickle.dump(approximator.LUTs, file)

    return final_scores


def main():
    from game2048 import Game2048Env, NTupleApproximator

    patterns = [
        ((0, 0), (0, 1), (1, 0), (1, 1)),  # corner square
        ((0, 1), (0, 2), (1, 1), (1, 2)),  # edge square
        ((0, 0), (0, 1), (0, 2), (0, 3)),  # edge line
        ((1, 0), (1, 1), (1, 2), (1, 3)),  # middle line
    ]
    approximator = NTupleApproximator(board_size=4, patterns=patterns, gamma=1)

    env = Game2048Env()

    print("start training")
    final_scores = td_learning(
        env, approximator, num_episodes=100000, alpha=0.1, gamma=1, epsilon=0.0
    )

    # save
    avg_score = np.mean(final_scores[-100:])
    ckpt_name = f"value_approx-{avg_score}.pkl"
    with open(ckpt_name, "wb") as file:
        pickle.dump(approximator.LUTs, file)


if __name__ == "__main__":
    main()
