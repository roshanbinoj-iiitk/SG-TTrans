"""
Continuous Fatigue Persistence duration tracking (tau_persist).
"""

class FatiguePersistenceTracker:
    """
    Tracks uninterrupted fatigue duration across streaming frames:
        tau_persist(t) = sum_{tau=t-T+1}^t I(y_tau in {Drowsy, Microsleep}) * Delta t
    """
    def __init__(self, sequence_length: int = 60, dt: float = 1.0 / 30.0):
        self.sequence_length = sequence_length
        self.dt = dt
        self.history: list[bool] = []

    def update(self, is_fatigued: bool) -> float:
        """
        Updates tracking buffer with current frame's fatigue classification.
        Returns continuous persistence duration in seconds.
        """
        self.history.append(is_fatigued)
        if len(self.history) > self.sequence_length:
            self.history.pop(0)

        # Count consecutive fatigue frames from most recent backward
        consecutive = 0
        for val in reversed(self.history):
            if val:
                consecutive += 1
            else:
                break

        return float(consecutive * self.dt)

    def reset(self):
        self.history.clear()
