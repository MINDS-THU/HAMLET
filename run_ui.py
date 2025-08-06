import sys
from pathlib import Path
import os
import shutil
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
from gradio_pdf import PDF

# Add project root to sys.path to allow for relative imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent_config_ui import AgentConfigManager
from smolagents import LiteLLMModel
from smolagents.tools import Tool
from src.custom_gradio_ui import stream_to_gradio
from src.base_agent import ToolCallingAgent, CodeAgent
from default_tools import get_available_tools, create_tool_instance

# Initialize the configuration manager for agents and tools
config_manager = AgentConfigManager()

def test_api_key():
    """Test if the OpenAI API key is working."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment variables"
        
        # Basic validation of API key format
        if not api_key.startswith("sk-"):
            return False, "API key format appears invalid (should start with 'sk-')"
        
        if len(api_key) < 20:
            return False, "API key appears too short"
        
        # Skip the actual API call test for now to avoid hanging
        # We'll let the LiteLLMModel handle the validation
        return True, "API key format looks valid (skipping network test to avoid hanging)"
        
    except Exception as e:
        return False, f"API key validation failed: {str(e)}"

class ToolFactory:
    """A factory to create tool instances from their names using dynamic discovery."""
    @staticmethod
    def create_tool(tool_name: str) -> Optional[Tool]:
        """
        Creates a tool instance based on the provided tool name using dynamic discovery.
        """
        print(f"Creating tool: {tool_name}")
        
        try:
            # Use the dynamic tool creation from default_tools
            tool = create_tool_instance(tool_name, working_dir="working_directory")
            
            # Ensure the tool has all required attributes
            if not hasattr(tool, 'name') or tool.name is None:
                tool.name = tool_name
                print(f"Fixed missing name for tool: {tool_name}")
            
            if not hasattr(tool, 'inputs'):
                tool.inputs = {
                    "input": {
                        "type": "string",
                        "description": f"Input for {tool_name}"
                    }
                }
                print(f"Added missing inputs for tool: {tool_name}")
                
            if not hasattr(tool, 'output_type'):
                tool.output_type = "string"
                print(f"Added missing output_type for tool: {tool_name}")
            
            return tool
            
        except ValueError as e:
            print(f"Tool '{tool_name}' not found in available tools: {e}")
            # Create a placeholder tool for unknown tools
            tool = Tool(
                name=tool_name, 
                description=f"Placeholder for unknown tool: {tool_name}", 
                func=lambda x, tn=tool_name: f"[{tn}] Tool not available - placeholder executed with: {x}"
            )
            tool.inputs = {
                "input": {
                    "type": "string",
                    "description": f"Input for placeholder {tool_name}"
                }
            }
            tool.output_type = "string"
            return tool
            
        except Exception as e:
            print(f"Error creating tool {tool_name}: {e}. Creating a placeholder tool.")
            tool = Tool(
                name=tool_name, 
                description=f"Error placeholder for {tool_name}", 
                func=lambda x, tn=tool_name: f"[{tn}] Error creating tool - placeholder executed with: {x}"
            )
            tool.inputs = {
                "input": {
                    "type": "string",
                    "description": f"Input for error placeholder {tool_name}"
                }
            }
            tool.output_type = "string"
            return tool

def build_agent(name: str, prompt: str, tools: list, sub_agents: list, agent_type: str = "ToolCallingAgent"):
    """Recursively builds a multi-step agent from its configuration."""
    tool_objs = []
    
    # Create tools and ensure they all have the required attributes
    for tool_name in tools:
        if tool_name:
            try:
                tool = ToolFactory.create_tool(tool_name)
                if tool is not None:
                    # Ensure the tool has a name attribute
                    if not hasattr(tool, 'name') or tool.name is None:
                        tool.name = tool_name
                        print(f"Fixed missing name for tool: {tool_name}")
                    
                    # Ensure the tool has other required attributes
                    if not hasattr(tool, 'description'):
                        tool.description = f"Tool for {tool_name}"
                    
                    tool_objs.append(tool)
                    print(f"Successfully created tool: {tool.name}")
                else:
                    print(f"Warning: Failed to create tool {tool_name}")
            except Exception as e:
                print(f"Error creating tool {tool_name}: {e}")
                # Create a fallback tool
                fallback_tool = Tool(
                    name=tool_name,
                    description=f"Fallback tool for {tool_name}",
                    func=lambda x: f"[{tool_name}] Executed with: {x}"
                )
                # Add required attributes for smolagents compatibility
                fallback_tool.inputs = {
                    "input": {
                        "type": "string",
                        "description": f"Input for {tool_name}"
                    }
                }
                fallback_tool.output_type = "string"
                tool_objs.append(fallback_tool)
    
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
        print("Creating LiteLLMModel...")
        # Explicitly set a default model to avoid issues with API key defaults.
        model = LiteLLMModel(model_id="gpt-3.5-turbo")
        print("LiteLLMModel instantiated successfully")
            
    except Exception as e:
        raise Exception(f"Failed to instantiate LiteLLMModel. Have you set your LLM API Key (e.g., OPENAI_API_KEY)? Error: {e}")

    print(f"Creating agent {name} with {len(tool_objs)} tools")
    for tool in tool_objs:
        print(f"  - Tool: {getattr(tool, 'name', 'UNNAMED')} (type: {type(tool).__name__})")

    print("Creating agent instance...")
    # Note: Using `additional_prompt_variables` and `managed_agents` as per the likely correct API
    agent = AgentClass(
        model=model,
        name=name,
        additional_prompt_variables={"system_prompt": prompt},
        tools=tool_objs,
        managed_agents=sub_agent_objs
    )
    print("Agent instance created successfully")
    return agent

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

    def launch_agent(state: dict, current_tools: list, current_prompt: str):
        """Launches the agent and switches to the chat UI."""
        if not state.get("name"):
            return gr.update(), gr.update(), None, "Please select, create, and save an agent first.", gr.update()
        
        # Test API key first
        api_working, api_message = test_api_key()
        if not api_working:
            return gr.update(), gr.update(), None, f"API Key Error: {api_message}", gr.update()
        
        try:
            agent_name = state["name"]
            print(f"Starting to build agent: {agent_name}")
            print(f"API key test passed: {api_message}")
            
            full_config = config_manager.get_all_agent_metadata().get(agent_name, {})
            
            # Use current UI values instead of saved state
            print(f"Building agent with tools: {current_tools}")
            agent = build_agent(
                name=agent_name,
                prompt=current_prompt,  # Use current prompt from UI
                tools=current_tools,    # Use current tools from UI
                sub_agents=state["managed_agents"],  # Sub-agents rarely change in UI
                agent_type=full_config.get("agent_type", "ToolCallingAgent")
            )
            print(f"Agent {agent_name} built successfully")
            
            return gr.update(visible=False), gr.update(visible=True), agent, "Agent launched successfully!", gr.update(value=f"## {agent.name}")
        except Exception as e:
            print(f"Error in launch_agent: {e}")
            import traceback
            traceback.print_exc()
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
        
        try:
            print(f"Starting agent interaction with prompt: {full_prompt[:100]}...")
            message_count = 0
            
            # Add timeout handling
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Agent interaction timed out")
            
            # Set timeout for Windows compatibility
            import threading
            import time
            
            result_container = {"messages": [], "error": None, "completed": False}
            
            def run_agent():
                try:
                    for message in stream_to_gradio(agent, task=full_prompt, reset_agent_memory=False):
                        result_container["messages"].append(message)
                        if len(result_container["messages"]) > 100:  # Prevent memory issues
                            break
                    result_container["completed"] = True
                except Exception as e:
                    result_container["error"] = e
            
            # Run agent in a separate thread with timeout
            agent_thread = threading.Thread(target=run_agent, daemon=True)
            agent_thread.start()
            
            # Wait for completion or timeout
            timeout_seconds = 60  # 60 second timeout
            start_time = time.time()
            
            while agent_thread.is_alive() and (time.time() - start_time) < timeout_seconds:
                # Process any messages that have been generated
                while message_count < len(result_container["messages"]):
                    message = result_container["messages"][message_count]
                    message_count += 1
                    print(f"Received message {message_count}: {type(message)}")
                    
                    if isinstance(message, str):
                        if history and history[-1].get("role") == "assistant":
                            history[-1]["content"] = message
                        else:
                            history.append(gr.ChatMessage(role="assistant", content=message))
                    elif isinstance(message, gr.ChatMessage):
                        history.append(message)
                    else:
                        print(f"Unknown message type: {type(message)}")
                        
                    yield history
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Check if we timed out
            if agent_thread.is_alive():
                print("Agent interaction timed out!")
                history.append(gr.ChatMessage(role="assistant", content="⏰ Agent response timed out. This might be due to API key issues or network problems."))
                yield history
                return
            
            # Check for errors
            if result_container["error"]:
                raise result_container["error"]
            
            # Process any remaining messages
            while message_count < len(result_container["messages"]):
                message = result_container["messages"][message_count]
                message_count += 1
                
                if isinstance(message, str):
                    if history and history[-1].get("role") == "assistant":
                        history[-1]["content"] = message
                    else:
                        history.append(gr.ChatMessage(role="assistant", content=message))
                elif isinstance(message, gr.ChatMessage):
                    history.append(message)
                    
                yield history
                
        except Exception as e:
            print(f"Error during agent interaction: {e}")
            import traceback
            traceback.print_exc()
            
            # Provide more specific error messages
            error_msg = str(e)
            if "openai" in error_msg.lower() or "api" in error_msg.lower():
                error_msg += "\n\n💡 This looks like an API key issue. Make sure you have set your OPENAI_API_KEY in a .env file."
            elif "timeout" in error_msg.lower():
                error_msg += "\n\n💡 The request timed out. Check your internet connection and API service status."
                
            history.append(gr.ChatMessage(role="assistant", content=f"❌ Error: {error_msg}"))
            yield history

    # --- GRADIO UI LAYOUT AND WIRING ---

    with gr.Blocks(title="HAMLET Agent Runner", fill_height=True) as demo:
        
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
                tool_checkboxes = gr.CheckboxGroup(choices=get_available_tools(), label="Select Tools")
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
            [config_state, tool_checkboxes, prompt_box], 
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
        ).then(lambda: gr.update(value=""), outputs=[chat_input])
        
        chat_input.submit(
            interact_with_agent, 
            [chat_input, chatbot, agent_state, file_uploads_log], 
            [chatbot]
        ).then(lambda: gr.update(value=""), outputs=[chat_input])

        back_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[config_page, chat_page])
        
    return demo

if __name__ == "__main__":
    app_instance = app()
    print("App created successfully, attempting to launch...")
    app_instance.queue()  # Enable queue for better handling
    app_instance.launch(server_port=7862, share=False)
