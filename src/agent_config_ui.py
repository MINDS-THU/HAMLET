#!/usr/bin/env python
# coding=utf-8

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path

import gradio as gr
from smolagents.tools import Tool
from smolagents.models import ChatMessage

from .base_agent import MultiStepAgent, ToolCallingAgent, CodeAgent
from .custom_gradio_ui import GradioUI


class AgentConfigManager:
    """Class for managing agent configurations"""
    
    def __init__(self, config_dir: str = "agent_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.agent_configs_file = self.config_dir / "agent_configs.json"
        self.tool_configs_file = self.config_dir / "tool_configs.json"
        
        # Initialize configuration files
        self._init_config_files()
    
    def _init_config_files(self):
        """Initialize configuration files with default values"""
        if not self.agent_configs_file.exists():
            default_agents = {
                "Literature Survey Agent": {
                    "prompt": "You are a literature survey agent specialized in academic research. You can search for papers, analyze content, and generate comprehensive reports.",
                    "tools": ["fetch_arxiv_papers", "open_deep_search", "file_editing"],
                    "sub_agents": [],
                    "agent_type": "ToolCallingAgent"
                },
                "Code Assistant": {
                    "prompt": "You are a code assistant that can help with programming tasks, debugging, and code analysis.",
                    "tools": ["file_editing"],
                    "sub_agents": [],
                    "agent_type": "CodeAgent"
                },
                "Research Assistant": {
                    "prompt": "You are a research assistant that can help with web search, document analysis, and information gathering.",
                    "tools": ["open_deep_search", "text_web_browser", "text_inspector"],
                    "sub_agents": [],
                    "agent_type": "ToolCallingAgent"
                }
            }
            self.save_agent_configs(default_agents)
        
        if not self.tool_configs_file.exists():
            default_tools = {
                "fetch_arxiv_papers": {
                    "name": "fetch_arxiv_papers",
                    "description": "Fetch papers from arXiv based on search query",
                    "category": "research"
                },
                "open_deep_search": {
                    "name": "open_deep_search", 
                    "description": "Perform deep web search for comprehensive information",
                    "category": "search"
                },
                "file_editing": {
                    "name": "file_editing",
                    "description": "Edit and manage local files",
                    "category": "file"
                },
                "text_web_browser": {
                    "name": "text_web_browser",
                    "description": "Browse web pages and extract text content",
                    "category": "web"
                },
                "text_inspector": {
                    "name": "text_inspector",
                    "description": "Analyze and inspect text content",
                    "category": "analysis"
                },
                "kb_repo_management": {
                    "name": "kb_repo_management",
                    "description": "Manage knowledge base repositories",
                    "category": "knowledge"
                },
                "visual_qa": {
                    "name": "visual_qa",
                    "description": "Answer questions about visual content",
                    "category": "vision"
                },
                "talk_to_user": {
                    "name": "talk_to_user",
                    "description": "Direct communication with user",
                    "category": "communication"
                }
            }
            self.save_tool_configs(default_tools)
    
    def get_all_agent_metadata(self) -> Dict[str, Dict]:
        """Get all agent configurations"""
        if self.agent_configs_file.exists():
            with open(self.agent_configs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get_all_tool_metadata(self) -> List[str]:
        """Get all tool names"""
        if self.tool_configs_file.exists():
            with open(self.tool_configs_file, 'r', encoding='utf-8') as f:
                tools = json.load(f)
                return list(tools.keys())
        return []
    
    def save_agent_configs(self, configs: Dict[str, Dict]):
        """Save agent configurations"""
        with open(self.agent_configs_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
    
    def save_tool_configs(self, configs: Dict[str, Dict]):
        """Save tool configurations"""
        with open(self.tool_configs_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
    
    def add_agent_config(self, name: str, config: Dict):
        """Add new agent configuration"""
        configs = self.get_all_agent_metadata()
        configs[name] = config
        self.save_agent_configs(configs)
    
    def add_tool_config(self, name: str, config: Dict):
        """Add new tool configuration"""
        configs = {}
        if self.tool_configs_file.exists():
            with open(self.tool_configs_file, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        configs[name] = config
        self.save_tool_configs(configs)


class ToolFactory:
    """Factory class for creating COOPA tools"""
    
    @staticmethod
    def create_tool(tool_name: str) -> Optional[Tool]:
        """Create corresponding COOPA tool based on tool name"""
        try:
            # Dynamically import tool modules
            if tool_name == "fetch_arxiv_papers":
                from general_tools.fetch_arxiv_papers.fetch_arxiv_papers_tools import ArxivPaperFetcher
                return ArxivPaperFetcher()
            
            elif tool_name == "open_deep_search":
                from general_tools.open_deep_search.ods_tool import OpenDeepSearchTool
                return OpenDeepSearchTool()
            
            elif tool_name == "file_editing":
                from general_tools.file_editing.file_editing_tools import FileEditingTool
                return FileEditingTool()
            
            elif tool_name == "text_web_browser":
                from general_tools.text_web_browser.text_web_browser import TextWebBrowser
                return TextWebBrowser()
            
            elif tool_name == "text_inspector":
                from general_tools.text_inspector.text_inspector_tool import TextInspectorTool
                return TextInspectorTool()
            
            elif tool_name == "kb_repo_management":
                from general_tools.kb_repo_management.kb_repo_retrieval_tools import KBRepoRetrievalTool
                return KBRepoRetrievalTool()
            
            elif tool_name == "visual_qa":
                from general_tools.visual_qa.visual_qa import VisualQATool
                return VisualQATool()
            
            elif tool_name == "talk_to_user":
                from general_tools.talk_to_user.talk_to_user_tool import TalkToUserTool
                return TalkToUserTool()
            
            else:
                # Create default tool
                return Tool(
                    name=tool_name,
                    description=f"Default tool for {tool_name}",
                    func=lambda x: f"[{tool_name}] Executed with: {x}"
                )
                
        except ImportError as e:
            print(f"Warning: Could not import tool {tool_name}: {e}")
            # Return a default tool
            return Tool(
                name=tool_name,
                description=f"Default tool for {tool_name}",
                func=lambda x: f"[{tool_name}] Executed with: {x}"
            )


class AgentFactory:
    """Factory class for creating COOPA agents"""
    
    def __init__(self, model_provider):
        self.model_provider = model_provider
    
    def create_agent(self, name: str, prompt: str, tools: List[str], 
                    agent_type: str = "ToolCallingAgent") -> MultiStepAgent:
        """Create agent instance"""
        # Create tool list
        tool_instances = []
        for tool_name in tools:
            tool = ToolFactory.create_tool(tool_name)
            if tool:
                tool_instances.append(tool)
        
        # Create agent based on agent type
        if agent_type == "CodeAgent":
            return CodeAgent(
                tools=tool_instances,
                model=self.model_provider,
                additional_prompt_variables={"system_prompt": prompt}
            )
        else:  # Default to ToolCallingAgent
            return ToolCallingAgent(
                tools=tool_instances,
                model=self.model_provider,
                additional_prompt_variables={"system_prompt": prompt}
            )


class AgentConfigUI:
    """Agent configuration UI class"""
    
    def __init__(self, model_provider):
        self.config_manager = AgentConfigManager()
        self.agent_factory = AgentFactory(model_provider)
    
    def build_ui(self):
        with gr.Blocks(title="COOPA Agent Configuration") as demo:
            # State for current agent config
            agent_config = gr.State({
                "name": "",
                "tools": [],
                "managed_agents": [],
                "prompt": "",
            })
            all_agents = gr.State(list(self.config_manager.get_all_agent_metadata().keys()))
            all_tools = gr.State(self.config_manager.get_all_tool_metadata())
            all_managed_agents = gr.State(list(self.config_manager.get_all_agent_metadata().keys()))

            # Top: Agent selection and create new agent
            with gr.Row():
                agent_selector = gr.Dropdown(choices=all_agents.value, label="Choose Agent")
                create_agent_btn = gr.Button("Create new agent")

            # Agent config form
            with gr.Column() as config_form:
                agent_name = gr.Textbox(label="Agent Name")

                gr.Markdown("Tools:")
                tools_column = gr.Column()
                add_tool_btn = gr.Button("Add New Tool")

                gr.Markdown("Managed Agents:")
                managed_column = gr.Column()
                add_managed_btn = gr.Button("Add New Managed...")

                gr.Markdown("Prompt template")
                prompt_box = gr.Textbox(label="Prompt", lines=4)

                with gr.Row():
                    save_btn = gr.Button("Save")
                    saveas_btn = gr.Button("Save as")
                launch_btn = gr.Button("Launch Agent", variant="primary")

            # Dummy edit modal (not functional yet)
            with gr.Column(visible=False) as edit_modal:
                gr.Markdown("Edit (dummy modal, not functional yet)")
                close_edit_btn = gr.Button("Close")

            # Conversation page (hidden by default)
            with gr.Column(visible=False) as chat_page:
                gr.Markdown("### Conversation")
                chatbot = gr.Chatbot(height=600, type="messages")
                user_input = gr.Textbox(label="Your Message")
                send_btn = gr.Button("Send")
                back_btn = gr.Button("⬅️ Back to Config")

            # --- Callbacks and dynamic logic ---
            # (Pseudo-code, to be filled in for each button and dynamic list)
            # - agent_selector.change: load agent config into agent_config
            # - create_agent_btn.click: clear agent_config for new agent
            # - add_tool_btn.click: append empty tool to agent_config["tools"]
            # - add_managed_btn.click: append empty managed agent to agent_config["managed_agents"]
            # - For each tool/managed agent: dropdown, edit (open dummy modal), delete
            # - save_btn.click: save agent_config
            # - saveas_btn.click: prompt for new name, save as new agent
            # - launch_btn.click: switch to chat_page
            # - back_btn.click: switch back to config_form

            # (You can use gr.update and gr.State to manage dynamic lists)

        return demo
    
    def launch(self, share: bool = True, **kwargs):
        """Launch UI"""
        demo = self.build_ui()
        return demo.launch(share=share, **kwargs)


def create_agent_config_ui(model_provider):
    """Convenience function to create agent configuration UI"""
    return AgentConfigUI(model_provider) 