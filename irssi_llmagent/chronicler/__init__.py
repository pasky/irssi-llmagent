"""Chronicler MVP: Chronicle storage and subagent interface."""

from enum import Enum


class QuestStatus(Enum):
    """Status of a quest in the chronicle."""

    ONGOING = "ongoing"
    IN_STEP = "in_step"
    FINISHED = "finished"
