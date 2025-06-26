from typing import Any


class OpenGame:
    """
    A class for representing and interacting with an open game."""

    strategy_profile: Any
    play: Any
    coplay: Any
    best_response: Any

    def map(self, observation, state) -> tuple[Any, Any]:
        """
        Plays a turn and returns the choice and outcome.

          Args:
            observation: The current observation of the environment.
            state: The internal state of the agent.

          Returns:
            A tuple containing the chosen action and the resulting outcome.
        """
        choice = self.play(observation, state)
        outcome = self.coplay(choice)
        return (choice, outcome)
