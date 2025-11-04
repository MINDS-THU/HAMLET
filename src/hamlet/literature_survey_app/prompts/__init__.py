"""
Agent Prompts Package

This package contains all agent prompt templates used in the literature survey system.
Each agent has its own dedicated prompt file for better organization and maintainability.
"""

from .writing_agent_prompts import WRITING_AGENT_DESCRIPTION, WRITING_AGENT_TASK_PROMPT
from .survey_writing_agent_prompts import SURVEY_WRITING_AGENT_SYSTEM_PROMPT

__all__ = [
    'WRITING_AGENT_DESCRIPTION',
    'WRITING_AGENT_TASK_PROMPT',
    'SURVEY_WRITING_AGENT_SYSTEM_PROMPT'
] 