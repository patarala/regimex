"""Dependency-free resource failures shared by backends and computation workers."""


class ResourceBudgetExceeded(RuntimeError):
    """Valid computation inputs cannot be processed within the assigned budget."""


class TimeBudgetExceeded(ResourceBudgetExceeded):
    """The worker's elapsed-time limit was reached."""
