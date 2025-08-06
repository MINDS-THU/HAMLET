#!/usr/bin/env python
# coding=utf-8

import os
import json
from typing import Dict, List
from pathlib import Path


class AgentConfigManager:
    """Class for managing agent configurations"""
    
    def __init__(self, config_dir: str = "agent_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.agent_configs_file = self.config_dir / "agent_configs.json"
        # Note: We no longer use tool_configs.json since we have dynamic discovery
        
        # Initialize configuration files
        self._init_config_files()
    
    def _init_config_files(self):
        """Initialize configuration files with default values"""
        if not self.agent_configs_file.exists():
            default_agents = {
                "Literature Survey Agent": {
                    "prompt": "You are a literature survey agent specialized in academic research. You can search for papers, analyze content, and generate comprehensive reports.",
                    "tools": ["web_search", "see_file", "modify_file"],  # Use actual discovered tool names
                    "sub_agents": [],
                    "agent_type": "ToolCallingAgent"
                },
                "Code Assistant": {
                    "prompt": "You are a code assistant that can help with programming tasks, debugging, and code analysis.",
                    "tools": ["see_file", "modify_file", "create_file_with_content", "list_dir"],  # Use actual discovered tool names
                    "sub_agents": [],
                    "agent_type": "CodeAgent"
                },
                "Research Assistant": {
                    "prompt": "You are a research assistant that can help with web search, document analysis, and information gathering.",
                    "tools": ["web_search", "see_file"],  # Use actual discovered tool names
                    "sub_agents": [],
                    "agent_type": "ToolCallingAgent"
                }
            }
            self.save_agent_configs(default_agents)
        
        # Note: We no longer create tool_configs.json as we use dynamic discovery
    
    def get_all_agent_metadata(self) -> Dict[str, Dict]:
        """Get all agent configurations"""
        if self.agent_configs_file.exists():
            with open(self.agent_configs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_all_tool_metadata(self) -> List[str]:
        """Get all tool names using dynamic discovery"""
        try:
            # Import here to avoid circular imports
            from default_tools import get_available_tools
            return get_available_tools()
        except ImportError:
            # Fallback if import fails
            return []
    
    def save_agent_configs(self, configs: Dict[str, Dict]):
        """Save agent configurations"""
        with open(self.agent_configs_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
    
    def add_agent_config(self, name: str, config: Dict):
        """Add new agent configuration"""
        configs = self.get_all_agent_metadata()
        configs[name] = config
        self.save_agent_configs(configs) 