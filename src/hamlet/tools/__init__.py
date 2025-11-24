"""
HAMLET Default Tools Package

This package provides a collection of tools for HAMLET agents, with automatic discovery
and registration of tools from subdirectories.

Usage:
    from default_tools import get_available_tools, create_tool_instance
    
    # Get list of all available tools
    tools = get_available_tools()
    
    # Create a tool instance
    search_tool = create_tool_instance('web_search')
"""

# Import the main functions from tool_registry
from .tool_registry import (
    discover_tools,
    get_available_tools, 
    create_tool_instance,
    get_tool_class,
    get_discovery_errors,
    ToolRegistry
)

# Define what gets imported with "from default_tools import *"
__all__ = [
    'discover_tools',
    'get_available_tools', 
    'create_tool_instance',
    'get_tool_class',
    'get_discovery_errors',
    'ToolRegistry'
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'HAMLET Team'
__description__ = 'Dynamic tool discovery and management for HAMLET agents'