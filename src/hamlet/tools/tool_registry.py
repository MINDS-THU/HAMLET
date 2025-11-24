"""
Tool registry for automatic discovery and loading of HAMLET default tools.
This module provides functionality to automatically discover and manage Tool classes from subdirectories.
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, List, Optional
from hamlet.core.tools import Tool


class ToolRegistry:
    """Registry for managing and discovering tools in the default_tools directory."""
    
    def __init__(self):
        self._tools_cache: Optional[Dict[str, Type[Tool]]] = None
        self._discovery_errors: List[str] = []
    
    def discover_tools(self, force_refresh: bool = False) -> Dict[str, Type[Tool]]:
        """
        Automatically discover all Tool classes in the default_tools subdirectories.
        
        Args:
            force_refresh: If True, force rediscovery even if cache exists
            
        Returns:
            Dict[str, Type[Tool]]: Dictionary mapping tool names to their Tool classes
        """
        if self._tools_cache is not None and not force_refresh:
            return self._tools_cache
            
        tools = {}
        self._discovery_errors = []
        
        # Get the directory containing this file (default_tools directory)
        default_tools_dir = Path(__file__).parent
        
        # Iterate through all subdirectories
        for item in default_tools_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__') and item.name != 'tool_registry':
                tool_dir_name = item.name
                
                # Try to import tools from this directory
                try:
                    # Look for Python files in the tool directory
                    for py_file in item.glob('*.py'):
                        if py_file.name.startswith('__'):
                            continue
                            
                        # Import the module
                        module_name = f"src.hamlet.tools.{tool_dir_name}.{py_file.stem}"
                        try:
                            module = importlib.import_module(module_name)
                            
                            # Find all Tool classes in the module
                            for name, obj in inspect.getmembers(module):
                                if (
                                    inspect.isclass(obj)
                                    and issubclass(obj, Tool)
                                    and obj != Tool
                                    and hasattr(obj, 'name')
                                ):
                                    tool_name = getattr(obj, 'name', name.lower())
                                    if tool_name in tools:
                                        print(
                                            f"[ToolRegistry] WARNING: Duplicate tool name '{tool_name}' from {module_name}. "
                                            "Keeping first definition from "
                                            f"{tools[tool_name].__module__} and skipping this one."
                                        )
                                        continue
                                    tools[tool_name] = obj
                                    print(f"Discovered tool: {tool_name} from {module_name}")
                                    
                        except ImportError as e:
                            error_msg = f"Could not import {module_name}: {e}"
                            print(error_msg)
                            self._discovery_errors.append(error_msg)
                        except Exception as e:
                            error_msg = f"Error processing {module_name}: {e}"
                            print(error_msg)
                            self._discovery_errors.append(error_msg)
                            
                except Exception as e:
                    error_msg = f"Error processing directory {tool_dir_name}: {e}"
                    print(error_msg)
                    self._discovery_errors.append(error_msg)
        
        self._tools_cache = tools
        return tools
    
    def get_available_tools(self) -> List[str]:
        """
        Get a list of all available tool names.
        
        Returns:
            List[str]: List of tool names
        """
        return list(self.discover_tools().keys())
    
    def get_tool_class(self, tool_name: str) -> Type[Tool]:
        """
        Get the Tool class for a given tool name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Type[Tool]: The Tool class
            
        Raises:
            ValueError: If tool_name is not found
        """
        tools = self.discover_tools()
        
        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(tools.keys())}")
        
        return tools[tool_name]
    
    def create_tool_instance(self, tool_name: str, **kwargs) -> Tool:
        """
        Create an instance of a tool by name.
        
        Args:
            tool_name: Name of the tool to create
            **kwargs: Additional arguments to pass to the tool constructor
        
        Returns:
            Tool: Instance of the requested tool
        
        Raises:
            ValueError: If tool_name is not found
        """
        tool_class = self.get_tool_class(tool_name)

        # Tools that require an injected working_dir parameter
        file_tools = [
            'list_dir', 'see_file', 'see_text_file', 'read_binary_as_markdown', 'modify_file',
            'create_file_with_content', 'search_keyword', 'delete_file_or_folder', 'load_object_from_python_file',
            'get_paper_from_url'
        ]
        if tool_name in file_tools:
            kwargs.setdefault('working_dir', kwargs.get('working_dir', 'working_directory'))

        return tool_class(**kwargs)
    
    def get_discovery_errors(self) -> List[str]:
        """
        Get any errors that occurred during tool discovery.
        
        Returns:
            List[str]: List of error messages
        """
        return self._discovery_errors.copy()
    
    def clear_cache(self):
        """Clear the tools cache to force rediscovery on next access."""
        self._tools_cache = None


# Global registry instance
_registry = ToolRegistry()

# Convenience functions for backward compatibility and easier usage
def discover_tools(force_refresh: bool = False) -> Dict[str, Type[Tool]]:
    """Discover all available tools."""
    return _registry.discover_tools(force_refresh=force_refresh)

def get_available_tools() -> List[str]:
    """Get list of available tool names."""
    return _registry.get_available_tools()

def create_tool_instance(tool_name: str, **kwargs) -> Tool:
    """Create a tool instance by name."""
    return _registry.create_tool_instance(tool_name, **kwargs)

def get_tool_class(tool_name: str) -> Type[Tool]:
    """Get a tool class by name."""
    return _registry.get_tool_class(tool_name)

def get_discovery_errors() -> List[str]:
    """Get any discovery errors."""
    return _registry.get_discovery_errors()
