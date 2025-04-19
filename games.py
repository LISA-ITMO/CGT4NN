from typing import Any

class OpenGame:
    strategy_profile: Any
    play: Any
    coplay: Any
    best_response: Any
    
    def map(self, observation, state) -> tuple[Any, Any]:
        choice = self.play(observation, state)
        outcome = self.coplay(choice)
        return (choice, outcome)