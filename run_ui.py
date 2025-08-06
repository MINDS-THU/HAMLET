import sys
from pathlib import Path
import os
import shutil
import time
from typing import Optional

import gradio as gr
import pandas as pd
from gradio_pdf import PDF

# Add project root to sys.path to allow for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_config_ui import AgentConfigManager
from smolagents import LiteLLMModel
from smolagents.tools import Tool
from src.custom_gradio_ui import stream_to_gradio
from src.base_agent import ToolCallingAgent, CodeAgent

# Initialize the configuration manager for agents and tools
config_manager = AgentConfigManager()

class ToolFactory:
    """A factory to create tool instances from their names."""
    @staticmethod
    def create_tool(tool_name: str) -> Optional[Tool]:
        """
        Creates a tool instance based on the provided tool name.
        It also hot-patches the .name attribute if the tool's __init__ doesn't set it.
        """
        tool_instance = None
        try:
            if tool_name == "open_deep_search":
                from default_tools.open_deep_search.ods_tool import OpenDeepSearchTool
                tool_instance = OpenDeepSearchTool()
            elif tool_name == "file_editing":
                from default_tools.file_editing.file_editing_tools import SeeFile
                tool_instance = SeeFile(working_dir="working_directory")
            else:
                print(f"Warning: Unknown tool '{tool_name}'. Creating a default placeholder tool.")
                tool_instance = Tool(name=tool_name, description=f"Placeholder for {tool_name}", func=lambda x: f"[{tool_name}] Executed with: {x}")
        except ImportError as e:
            print(f"Warning: Could not import tool {tool_name}: {e}. Creating a default placeholder tool.")
            tool_instance = Tool(name=tool_name, description=f"Placeholder for {tool_name}", func=lambda x: f"[{tool_name}] Executed with: {x}")

        # Hot-patch the name attribute if it's missing from the tool's __init__
        if tool_instance and not hasattr(tool_instance, 'name'):
            tool_instance.name = tool_name
        
        # Hot-patch the description attribute as well if it's missing
        if tool_instance and not hasattr(tool_instance, 'description'):
            tool_instance.description = f"A tool for {tool_name.replace('_', ' ')}."

        # Hot-patch the inputs attribute, which the template also requires.
        if tool_instance and not hasattr(tool_instance, 'inputs'):
            tool_instance.inputs = {} # Provide a safe default

        # Hot-patch the output_type attribute, the final missing piece.
        if tool_instance and not hasattr(tool_instance, 'output_type'):
            tool_instance.output_type = {} # Provide a safe default

        return tool_instance

def build_agent(name: str, prompt: str, tools: list, sub_agents: list, agent_type: str = "ToolCallingAgent") -> ToolCallingAgent:
    """Recursively builds a multi-step agent from its configuration."""
    tool_objs = [ToolFactory.create_tool(t) for t in tools if t]
    
    sub_agent_objs = []
    for sa_name in sub_agents:
        md = config_manager.get_all_agent_metadata().get(sa_name, {})
        sub_agent_objs.append(build_agent(
            name=sa_name,
            prompt=md.get("prompt", ""),
            tools=md.get("tools", []),
            sub_agents=md.get("sub_agents", []),
            agent_type=md.get("agent_type", "ToolCallingAgent")
        ))
    
    AgentClass = CodeAgent if agent_type == "CodeAgent" else ToolCallingAgent
    
    try:
        # Explicitly set a default model to avoid issues with API key defaults.
        model = LiteLLMModel(model_id="gpt-3.5-turbo")
    except Exception as e:
        raise Exception(f"Failed to instantiate LiteLLMModel. Have you set your LLM API Key (e.g., OPENAI_API_KEY)? Error: {e}")

    # Note: Using `additional_prompt_variables` and `managed_agents` as per the likely correct API
    return AgentClass(
        model=model,
        name=name,
        additional_prompt_variables={"system_prompt": prompt},
        tools=tool_objs,
        managed_agents=sub_agent_objs
    )

def app():
    """Main function to create and configure the Gradio application."""
    working_dir = os.path.abspath("working_directory")
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # --- ALL EVENT HANDLERS DEFINED FIRST ---

    def on_agent_select(agent_name: str):
        """Handles agent selection dropdown change."""
        if agent_name == "Create new agent":
            state = {"name": "", "tools": [], "managed_agents": [], "prompt": ""}
            return state, "", [], [], "", gr.update(visible=True)
        
        md = config_manager.get_all_agent_metadata().get(agent_name, {})
        state = {
            "name": agent_name,
            "tools": md.get("tools", []),
            "managed_agents": md.get("sub_agents", []),
            "prompt": md.get("prompt", "")
        }
        return state, state["name"], state["tools"], state["managed_agents"], state["prompt"], gr.update(visible=True)

    def save_config(name: str, tools: list, sub_agents: list, prompt: str):
        """Saves the agent's configuration."""
        if not name:
            return gr.update(), gr.update(), gr.update(value="Agent name cannot be empty.")
        
        all_configs = config_manager.get_all_agent_metadata()
        all_configs[name] = {"prompt": prompt, "tools": tools, "sub_agents": sub_agents, "agent_type": "ToolCallingAgent"}
        config_manager.save_agent_configs(all_configs)
        
        all_agent_names = list(config_manager.get_all_agent_metadata().keys())
        return gr.update(choices=all_agent_names + ["Create new agent"], value=name), gr.update(choices=all_agent_names), f"Saved '{name}'"

    def launch_agent(state: dict):
        """Launches the agent and switches to the chat UI."""
        if not state.get("name"):
            return gr.update(), gr.update(), None, "Please select, create, and save an agent first.", gr.update()
        try:
            agent_name = state["name"]
            full_config = config_manager.get_all_agent_metadata().get(agent_name, {})
            agent = build_agent(
                name=agent_name,
                prompt=state["prompt"],
                tools=state["tools"],
                sub_agents=state["managed_agents"],
                agent_type=full_config.get("agent_type", "ToolCallingAgent")
            )
            return gr.update(visible=False), gr.update(visible=True), agent, "Agent launched successfully!", gr.update(value=f"## {agent.name}")
        except Exception as e:
            return gr.update(), gr.update(), None, f"Error building agent: {e}", gr.update()

    def handle_upload(file_obj, upload_log: list):
        """Copies uploaded file to the working directory."""
        if not file_obj:
            return gr.update(visible=False), upload_log
        
        shutil.copy(file_obj.name, os.path.join(working_dir, os.path.basename(file_obj.name)))
        return gr.update(value=f"Uploaded {os.path.basename(file_obj.name)}", visible=True), upload_log + [os.path.basename(file_obj.name)]

    def force_refresh_files():
        """Forces the file explorer to refresh by updating its glob pattern."""
        return gr.update(ignore_glob=f"__refresh_{int(time.time() * 1000)}__")

    def show_preview(selection: list):
        """Displays a preview of the selected file in the right-hand pane."""
        hidden = gr.update(visible=False)
        outputs = [hidden] * 5  # pdf, img, df, code, text
        
        if not selection:
            outputs[-1] = gr.update(value="⬅️ Click a file to preview", visible=True)
            return outputs

        rel_path = selection[0]
        abs_path = os.path.join(working_dir, rel_path)

        if os.path.isdir(abs_path):
            outputs[-1] = gr.update(value="📁 Directory (no preview)", visible=True)
            return outputs

        ext = os.path.splitext(abs_path)[1].lower()
        if ext == ".pdf":
            outputs[0] = gr.update(value=abs_path, visible=True)
        elif ext in {".png", ".jpg", ".jpeg", ".gif"}:
            outputs[1] = gr.update(value=abs_path, visible=True)
        elif ext in {".csv", ".xlsx"}:
            try:
                df = pd.read_csv(abs_path) if ext == ".csv" else pd.read_excel(abs_path)
                outputs[2] = gr.update(value=df, visible=True)
            except Exception as e:
                outputs[-1] = gr.update(value=f"⚠️ Cannot read table: {e}", visible=True)
        else:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(20000)
                outputs[3] = gr.update(value=content, visible=True)
            except Exception as e:
                outputs[-1] = gr.update(value=f"⚠️ Cannot display file: {e}", visible=True)
        return outputs

    def interact_with_agent(prompt: str, history: list, agent, uploads: list):
        """Handles the chat interaction with the agent."""
        if not agent:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": "ERROR: Agent not loaded."})
            yield history
            return

        history.append({"role": "user", "content": prompt})
        yield history

        full_prompt = prompt + (f"\n\nFiles provided: {', '.join(uploads)}" if uploads else "")
        
        for message in stream_to_gradio(agent, task=full_prompt, reset_agent_memory=False):
            if isinstance(message, str):
                if history and history[-1].get("role") == "assistant":
                    history[-1]["content"] = message
                else:
                    history.append(gr.ChatMessage(role="assistant", content=message))
            elif isinstance(message, gr.ChatMessage):
                history.append(message)
            yield history

    # --- GRADIO UI LAYOUT AND WIRING ---

    with gr.Blocks(title="COOPA Agent Runner", theme=gr.themes.Default(), fill_height=True) as demo:
        
        # --- State Management ---
        config_state = gr.State({"name": "", "tools": [], "managed_agents": [], "prompt": ""})
        agent_state = gr.State(None)
        file_uploads_log = gr.State([])

        # --- PAGE 1: AGENT CONFIGURATION ---
        with gr.Column(visible=True) as config_page:
            gr.Markdown("# Agent Configuration")
            agent_selector = gr.Dropdown(
                choices=list(config_manager.get_all_agent_metadata().keys()) + ["Create new agent"],
                label="Choose Agent or Create New",
                value="Create new agent"
            )
            launch_status = gr.Textbox(label="Status", interactive=False, lines=1)
            
            with gr.Column(visible=False) as config_form:
                agent_name_input = gr.Textbox(label="Agent Name")
                tool_checkboxes = gr.CheckboxGroup(choices=list(config_manager.get_all_tool_metadata()), label="Select Tools")
                sub_agent_checkboxes = gr.CheckboxGroup(choices=list(config_manager.get_all_agent_metadata().keys()), label="Select Sub-Agents")
                prompt_box = gr.Textbox(lines=5, label="System Prompt")
                with gr.Row():
                    save_button = gr.Button("Save Configuration")
                    close_button = gr.Button("Cancel")
            
            launch_button = gr.Button("Launch Agent", variant="primary")

        # --- PAGE 2: CHAT INTERFACE ---
        with gr.Row(visible=False) as chat_page:
            # Left Sidebar: Controls
            with gr.Column(scale=3, min_width=300):
                agent_title_md = gr.Markdown("## Code Assistant")
                gr.Markdown("An agent that can conduct literature survey.")
                file_uploader = gr.File(label="Upload a file", type="filepath")
                upload_status = gr.Textbox(visible=False, interactive=False)
                back_button = gr.Button("← Back to Config")

            # Middle Area: Chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    label="Agent Chat",
                    avatar_images=(None, "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png"),
                    height="calc(100vh - 160px)", # Adjust height for the input box below
                    show_copy_button=True,
                    type="messages"
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        show_label=False,
                        placeholder="Type your prompt…",
                        lines=4,
                        scale=8,
                    )
                    submit_button = gr.Button("提交", variant="primary", scale=1, min_width=100)
            
            # Right Sidebar: Files and Preview
            with gr.Column(scale=3, min_width=300):
                file_explorer = gr.FileExplorer(root_dir=working_dir, file_count="multiple", interactive=True)
                with gr.Column():
                    pdf_preview = PDF(visible=False, interactive=False)
                    img_preview = gr.Image(interactive=False, visible=False)
                    df_preview = gr.DataFrame(interactive=False, visible=False)
                    code_preview = gr.Code(language="markdown", interactive=False, visible=False)
                    text_preview = gr.Markdown(value="⬅️ Click a file to preview", visible=True)

        # --- Event Wiring ---
        agent_selector.change(
            on_agent_select, 
            [agent_selector], 
            [config_state, agent_name_input, tool_checkboxes, sub_agent_checkboxes, prompt_box, config_form]
        )
        save_button.click(
            save_config, 
            [agent_name_input, tool_checkboxes, sub_agent_checkboxes, prompt_box], 
            [agent_selector, sub_agent_checkboxes, launch_status]
        )
        close_button.click(lambda: gr.update(visible=False), outputs=[config_form])
        launch_button.click(
            launch_agent, 
            [config_state], 
            [config_page, chat_page, agent_state, launch_status, agent_title_md]
        )
        file_explorer.change(
            show_preview, 
            [file_explorer], 
            [pdf_preview, img_preview, df_preview, code_preview, text_preview]
        )
        file_uploader.change(
            handle_upload, 
            [file_uploader, file_uploads_log], 
            [upload_status, file_uploads_log]
        ).then(force_refresh_files, None, [file_explorer])
        
        submit_button.click(
            interact_with_agent, 
            [chat_input, chatbot, agent_state, file_uploads_log], 
            [chatbot]
        ).then(lambda: "", outputs=[chat_input])
        
        chat_input.submit(
            interact_with_agent, 
            [chat_input, chatbot, agent_state, file_uploads_log], 
            [chatbot]
        ).then(lambda: "", outputs=[chat_input])

        back_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[config_page, chat_page])
        
    return demo

if __name__ == "__main__":
    app_instance = app()
    app_instance.launch()
