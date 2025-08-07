import sys
from pathlib import Path
import os
import shutil
import time
import threading
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
from gradio_pdf import PDF

# Add project root to sys.path to allow for relative imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent_config_ui import AgentConfigManager, stream_to_gradio
from smolagents import LiteLLMModel
from smolagents.tools import Tool
from smolagents.monitoring import LogLevel
from smolagents import ToolCallingAgent, CodeAgent
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
    def create_tool(tool_name: str, working_dir: str = "working_directory") -> Optional[Tool]:
        """
        Creates a tool instance based on the provided tool name using dynamic discovery.
        """
        print(f"Creating tool: {tool_name} with working_dir: {working_dir}")
        
        try:
            # Use the dynamic tool creation from default_tools
            tool = create_tool_instance(tool_name, working_dir=working_dir)
            
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

def build_agent(name: str, prompt: str, tools: list, sub_agents: list, agent_type: str = "ToolCallingAgent", working_dir: Optional[str] = None):
    """Recursively builds a multi-step agent from its configuration."""
    # Use default working directory if none provided
    if working_dir is None:
        working_dir = os.path.abspath("working_directory")
    
    # Sanitize agent name to be a valid Python identifier
    import re
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = 'agent_' + sanitized_name
    if not sanitized_name or sanitized_name == '_':
        sanitized_name = 'unnamed_agent'
    
    print(f"Original name: '{name}' -> Sanitized name: '{sanitized_name}'")
    print(f"Building agent with working directory: {working_dir}")
    
    tool_objs = []
    
    # Create tools and ensure they all have the required attributes
    for tool_name in tools:
        if tool_name:
            try:
                tool = ToolFactory.create_tool(tool_name, working_dir)
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
            agent_type=md.get("agent_type", "ToolCallingAgent"),
            working_dir=working_dir  # Pass working directory to sub-agents
        ))
    
    AgentClass = CodeAgent if agent_type == "CodeAgent" else ToolCallingAgent
    
    try:
        print("Creating LiteLLMModel...")
        # Use a valid model with timeout settings
        model = LiteLLMModel(
            model_id="gpt-3.5-turbo",
            timeout=30  # 30 second timeout
        )
        print("LiteLLMModel instantiated successfully")
            
    except Exception as e:
        raise Exception(f"Failed to instantiate LiteLLMModel. Network timeout or API key issue. Error: {e}")

    print(f"Creating agent {sanitized_name} (original: {name}) with {len(tool_objs)} tools")
    for tool in tool_objs:
        print(f"  - Tool: {getattr(tool, 'name', 'UNNAMED')} (type: {type(tool).__name__})")

    print("Creating agent instance...")
    
    # Create agent with proper parameters matching the base agent class
    agent = AgentClass(
        tools=tool_objs,
        model=model,
        managed_agents=sub_agent_objs,
        verbosity_level=LogLevel.DEBUG,
        planning_interval=3,
        max_steps=30,
        name=sanitized_name,
        description=prompt or f"A {agent_type} agent with {len(tool_objs)} tools."
    )
    print(f"Agent instance created successfully with name: {sanitized_name}")
    return agent

def app():
    """Main function to create and configure the Gradio application."""
    # Default working directory
    default_working_dir = os.path.abspath("working_d")
    
    def setup_working_directory(path: str):
        """Setup and validate working directory"""
        if not path or path.strip() == "":
            path = default_working_dir
        
        working_dir = os.path.abspath(path)
        if not os.path.exists(working_dir):
            try:
                os.makedirs(working_dir)
                print(f"Created working directory: {working_dir}")
            except Exception as e:
                print(f"Error creating directory {working_dir}: {e}")
                working_dir = default_working_dir
                os.makedirs(working_dir, exist_ok=True)
        else:
            print(f"Using existing working directory: {working_dir}")
        
        return working_dir

    # --- ALL EVENT HANDLERS DEFINED FIRST ---

    def on_working_dir_change(path: str):
        """Handles working directory change."""
        nonlocal current_watching_dir
        if not path:
            path = default_working_dir
        working_dir = setup_working_directory(path)
        
        # Start watching the new directory
        current_watching_dir = working_dir
        # The callback now only logs, the UI refresh is handled by the timer
        file_watcher.start_watching(working_dir, lambda: print("File system change detected by watcher."))
        
        return working_dir, gr.update(root_dir=working_dir)

    def test_button_click():
        """Simple test function to see if button clicks work."""
        print("DEBUG: Button was clicked!")
        return gr.update(visible=True)

    def show_directory_explorer():
        """Show the directory explorer for folder selection."""
        import os
        print("DEBUG: show_directory_explorer function called!")
        
        # Get current directory as default
        current_dir = os.getcwd()
        print(f"DEBUG: Current directory = {current_dir}")
        
        return (
            gr.update(visible=True),      # dir_explorer_column
            gr.update(value=current_dir), # selected_path_display
            gr.update(value="Button clicked!", visible=True)  # test_label
        )

    def select_quick_directory(dir_type: str):
        """Select a common directory quickly."""
        import os
        
        if dir_type == "current":
            path = os.getcwd()
        elif dir_type == "desktop":
            path = os.path.join(os.path.expanduser("~"), "Desktop")
        elif dir_type == "documents":
            path = os.path.join(os.path.expanduser("~"), "Documents")
        elif dir_type == "home":
            path = os.path.expanduser("~")
        else:
            path = os.getcwd()
        
        return gr.update(value=path)

    def create_directory(path: str):
        """Create a new directory if it doesn't exist."""
        if not path or path.strip() == "":
            return gr.update(value=path), gr.update(value="❌ Please enter a valid path", visible=True)
        
        try:
            abs_path = os.path.abspath(path.strip())
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)
                # Return the path and success status
                return gr.update(value=abs_path), gr.update(value=f"✅ Created: {abs_path}", visible=True)
            else:
                # Return the path and already exists status
                return gr.update(value=abs_path), gr.update(value=f"📁 Already exists: {abs_path}", visible=True)
        except Exception as e:
            return gr.update(value=path), gr.update(value=f"❌ Error creating directory: {str(e)}", visible=True)

    def update_manual_path(path: str):
        """Update the selected path display when user types manually."""
        if path and path.strip():
            abs_path = os.path.abspath(path.strip())
            return gr.update(value=abs_path)
        return gr.update(value="")

    def hide_directory_explorer():
        """Hide the directory explorer."""
        return gr.update(visible=False)

    def confirm_directory_selection(selected_path: str):
        """Confirm the selected directory from the manual input or quick selection."""
        if not selected_path or selected_path.strip() == "":
            return gr.update(value="❌ No directory selected"), gr.update(visible=False)
        
        try:
            # Convert to absolute path and validate
            abs_path = os.path.abspath(selected_path.strip())
            
            # Create directory if it doesn't exist
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)
                print(f"Created directory: {abs_path}")
            
            # Update the working directory input
            return gr.update(value=abs_path), gr.update(visible=False)
            
        except Exception as e:
            return gr.update(value=f"❌ Error: {str(e)}"), gr.update(visible=True)

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

    def launch_agent(state: dict, current_tools: list, current_prompt: str, current_working_dir: str):
        """Launches the agent and switches to the chat UI."""
        if not state.get("name"):
            return gr.update(), gr.update(), None, "Please select, create, and save an agent first.", gr.update(), gr.update()
        
        # Test API key first
        api_working, api_message = test_api_key()
        if not api_working:
            return gr.update(), gr.update(), None, f"API Key Error: {api_message}", gr.update(), gr.update()
        
        try:
            agent_name = state["name"]
            print(f"Starting to build agent: {agent_name}")
            print(f"API key test passed: {api_message}")
            print(f"Using working directory: {current_working_dir}")
            
            full_config = config_manager.get_all_agent_metadata().get(agent_name, {})
            
            # Use current UI values instead of saved state
            print(f"Building agent with tools: {current_tools}")
            agent = build_agent(
                name=agent_name,
                prompt=current_prompt,  # Use current prompt from UI
                tools=current_tools,    # Use current tools from UI
                sub_agents=state["managed_agents"],  # Sub-agents rarely change in UI
                agent_type=full_config.get("agent_type", "ToolCallingAgent"),
                working_dir=current_working_dir  # Pass working directory to agent
            )
            print(f"Agent {agent_name} built successfully")
            
            return gr.update(visible=False), gr.update(visible=True), agent, "Agent launched successfully!", gr.update(value=f"## {agent.name}"), gr.update(value=current_working_dir)
        except Exception as e:
            print(f"Error in launch_agent: {e}")
            import traceback
            traceback.print_exc()
            return gr.update(), gr.update(), None, f"Error building agent: {e}", gr.update(), gr.update()

    def handle_upload(file_obj, upload_log: list, working_dir: str):
        """Copies uploaded file to the working directory."""
        if not file_obj:
            return gr.update(visible=False), upload_log, gr.update()
        
        filename = os.path.basename(file_obj.name)
        dest_path = os.path.join(working_dir, filename)
        shutil.copy(file_obj.name, dest_path)
        print(f"File uploaded: {filename} -> {dest_path}")
        
        # Return the status update, updated log, and file explorer refresh
        return (
            gr.update(value=f"Uploaded {filename}", visible=True), 
            upload_log + [filename], 
            force_refresh_files(working_dir)
        )

    def force_refresh_files(working_dir=None):
        """Forces the file explorer to refresh by updating its root directory."""
        # Force refresh by re-setting the root_dir to trigger a re-scan
        if working_dir is None:
            working_dir = current_watching_dir or default_working_dir
        
        print(f"Forcing file explorer refresh for directory: {working_dir}")
        
        # List current files for debugging
        try:
            files = os.listdir(working_dir)
            print(f"Files in directory {working_dir}: {files}")
        except Exception as e:
            print(f"Error listing directory {working_dir}: {e}")
        
        # Force a complete refresh by changing the root_dir
        return gr.update(root_dir=working_dir)
    
    # Global variable for file watcher state
    current_watching_dir = None
    
    def periodic_refresh(working_dir_state):
        """Periodic refresh to catch any missed file changes."""
        # This is a reliable catch-all. It runs every 3 seconds.
        if working_dir_state and os.path.exists(working_dir_state):
            print("Periodic refresh triggered.")
            return force_refresh_files(working_dir_state)
        return gr.update()

    def show_preview(selection: list, working_dir: str):
        """Displays a preview of the selected file in the right-hand pane."""
        hidden = gr.update(visible=False)
        outputs = [hidden] * 5  # pdf, img, df, code, text
        
        if not selection:
            outputs[-1] = gr.update(value="⬅️ Click a file to preview", visible=True)
            return outputs

        rel_path = selection[0]
        
        # Debug: Print selection and working directory
        print(f"DEBUG: File selection: {selection}")
        print(f"DEBUG: Working directory: {working_dir}")
        print(f"DEBUG: Relative path: {rel_path}")
        
        # Handle both relative and absolute paths
        if os.path.isabs(rel_path):
            abs_path = rel_path
        else:
            abs_path = os.path.join(working_dir, rel_path)
        
        # Normalize the path to handle any inconsistencies
        abs_path = os.path.normpath(abs_path)
        
        print(f"DEBUG: Final absolute path: {abs_path}")

        # Check if file exists, if not try some alternatives
        if not os.path.exists(abs_path):
            print(f"DEBUG: File not found at {abs_path}")
            
            # Try to find the file in common locations
            possible_paths = [
                abs_path,
                os.path.join(os.getcwd(), rel_path),
                os.path.join(os.path.dirname(__file__), rel_path),
                os.path.join(os.path.dirname(__file__), "working_directory", rel_path),
                os.path.join(os.path.dirname(__file__), "working_d", rel_path),
            ]
            
            found_path = None
            for path in possible_paths:
                print(f"DEBUG: Trying path: {path}")
                if os.path.exists(path):
                    found_path = path
                    print(f"DEBUG: Found file at: {found_path}")
                    break
            
            if found_path:
                abs_path = found_path
            else:
                # List files in working directory for debugging
                try:
                    files_in_dir = os.listdir(working_dir)
                    print(f"DEBUG: Files in working directory {working_dir}: {files_in_dir}")
                except Exception as e:
                    print(f"DEBUG: Cannot list working directory: {e}")
                
                error_msg = f"⚠️ File not found: {abs_path}\nWorking dir: {working_dir}\nSelection: {selection}"
                print(f"ERROR: {error_msg}")
                outputs[-1] = gr.update(value=error_msg, visible=True)
                return outputs

        if os.path.isdir(abs_path):
            outputs[-1] = gr.update(value="📁 Directory (no preview)", visible=True)
            return outputs

        ext = os.path.splitext(abs_path)[1].lower()
        
        # PDF files
        if ext == ".pdf":
            outputs[0] = gr.update(value=abs_path, visible=True)
        # Image files
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            outputs[1] = gr.update(value=abs_path, visible=True)
        # Spreadsheet files
        elif ext in {".csv", ".xlsx", ".tsv"}:
            try:
                df = pd.read_csv(abs_path) if ext != ".xlsx" else pd.read_excel(abs_path)
                outputs[2] = gr.update(value=df, visible=True)
            except Exception as e:
                outputs[-1] = gr.update(value=f"⚠️ Cannot read table: {e}", visible=True)
        # Code and text files
        else:
            try:
                print(f"DEBUG: Attempting to read text file: {abs_path}")
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(20000)  # Read first 20KB to avoid UI freezing
                
                print(f"DEBUG: Successfully read {len(content)} characters")
                
                # Determine if it's a code file for syntax highlighting
                if ext in {".py", ".js", ".html", ".css", ".json", ".xml", ".yaml", ".yml"}:
                    language = {
                        ".py": "python", ".js": "javascript", ".html": "html", 
                        ".css": "css", ".json": "json", ".xml": "xml", 
                        ".yaml": "yaml", ".yml": "yaml"
                    }.get(ext, "text")
                    print(f"DEBUG: Using code preview with language: {language}")
                    outputs[3] = gr.update(value=content, language=language, visible=True)
                else:
                    print("DEBUG: Using text preview")
                    outputs[-1] = gr.update(value=content, visible=True)
            except Exception as e:
                error_msg = f"⚠️ Cannot display file: {str(e)}"
                print(f"ERROR reading file: {e}")
                import traceback
                traceback.print_exc()
                outputs[-1] = gr.update(value=error_msg, visible=True)
                
        return outputs

    def interact_with_agent(prompt: str, history: list, agent, uploads: list):
        """Handles the chat interaction with the agent."""
        import gradio as gr
        
        if not agent:
            history.append(gr.ChatMessage(role="user", content=prompt))
            history.append(gr.ChatMessage(role="assistant", content="ERROR: Agent not loaded."))
            yield history
            return

        try:
            history.append(gr.ChatMessage(role="user", content=prompt))
            yield history

            full_prompt = prompt + (f"\n\nFiles provided: {', '.join(uploads)}" if uploads else "")
            
            print(f"Starting agent interaction with prompt: {full_prompt[:50]}...")
            
            # Add timeout handling for network issues
            import signal
            import threading
            import time
            
            # Container to hold results from the agent thread
            result = {"messages": [], "error": None, "completed": False}
            
            def run_agent_with_timeout():
                try:
                    for msg in stream_to_gradio(agent, task=full_prompt, reset_agent_memory=False):
                        result["messages"].append(msg)
                        if len(result["messages"]) > 50:  # Prevent memory issues
                            break
                    result["completed"] = True
                except Exception as e:
                    result["error"] = e
            
            # Run agent in background thread with timeout
            agent_thread = threading.Thread(target=run_agent_with_timeout, daemon=True)
            agent_thread.start()
            
            timeout_seconds = 30  # 30 second timeout
            start_time = time.time()
            message_count = 0
            
            while agent_thread.is_alive() and (time.time() - start_time) < timeout_seconds:
                # Process any new messages
                while message_count < len(result["messages"]):
                    msg = result["messages"][message_count]
                    message_count += 1
                    
                    if isinstance(msg, gr.ChatMessage):
                        history.append(msg)
                    elif isinstance(msg, str):
                        if history and getattr(history[-1], 'metadata', {}).get("status") == "pending":
                            history[-1].content = msg
                        else:
                            history.append(
                                gr.ChatMessage(role="assistant", content=msg, metadata={"status": "pending"})
                            )
                    yield history
                
                time.sleep(0.1)  # Small delay
            
            # Check if we timed out
            if agent_thread.is_alive():
                error_msg = """⏰ **Request timed out.** 

This is likely due to network connectivity issues preventing access to OpenAI's API. 

**Possible solutions:**
- Check your internet connection
- Try using a VPN if you're in a restricted region
- Check if your firewall/antivirus is blocking the connection
- Verify OpenAI services are operational at https://status.openai.com"""
                
                history.append(gr.ChatMessage(role="assistant", content=error_msg))
                yield history
                return
            
            # Handle any error that occurred
            if result["error"]:
                raise result["error"]
            
            # Process any remaining messages
            while message_count < len(result["messages"]):
                msg = result["messages"][message_count]
                message_count += 1
                
                if isinstance(msg, gr.ChatMessage):
                    history.append(msg)
                elif isinstance(msg, str):
                    if history and getattr(history[-1], 'metadata', {}).get("status") == "pending":
                        history[-1].content = msg
                    else:
                        history.append(
                            gr.ChatMessage(role="assistant", content=msg, metadata={"status": "pending"})
                        )
                yield history

        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Provide specific error messages for common issues
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "ConnectTimeout" in error_msg:
                error_msg = """🌐 **Network Connection Error**

Cannot connect to OpenAI's servers. This is likely due to:
- Network connectivity issues
- Firewall blocking the connection  
- Geographic restrictions
- VPN required for your location

**Try:**
- Check your internet connection
- Disable firewall temporarily
- Use a VPN if in a restricted region
- Check https://status.openai.com for service status"""
            elif "api" in error_msg.lower() and "key" in error_msg.lower():
                error_msg += "\n\n💡 **API Key Issue:** Check your OPENAI_API_KEY in the .env file"
            
            history.append(gr.ChatMessage(role="assistant", content=f"❌ **Error:** {error_msg}"))
            yield history

    # --- GRADIO UI LAYOUT AND WIRING ---

    with gr.Blocks(title="HAMLET Agent Runner", fill_height=True) as demo:
        
        # --- State Management ---
        config_state = gr.State({"name": "", "tools": [], "managed_agents": [], "prompt": ""})
        agent_state = gr.State(None)
        file_uploads_log = gr.State([])
        working_dir_state = gr.State(setup_working_directory(default_working_dir))
        
        # Add a simple periodic refresh timer
        refresh_timer = gr.Timer(value=3.0)  # Check every 3 seconds

        # --- PAGE 1: AGENT CONFIGURATION ---
        with gr.Column(visible=True) as config_page:
            gr.Markdown("# Agent Configuration")
            
            # Working Directory Selection
            with gr.Row():
                working_dir_input = gr.Textbox(
                    label="Working Directory", 
                    value=default_working_dir,
                    placeholder="Enter path to working directory...",
                    scale=4
                )
                use_explorer_btn = gr.Button("📂 Select Folder", scale=1)
                # Add a test label to see if button clicks work
                test_label = gr.Markdown("", visible=False)
            
            # Directory Explorer (initially hidden)
            with gr.Column(visible=False) as dir_explorer_column:
                gr.Markdown("**Choose Working Directory**")
                
                # Quick directory options
                with gr.Row():
                    gr.Markdown("**Quick Options:**")
                with gr.Row():
                    current_dir_btn = gr.Button("📂 Current Directory", size="sm")
                    desktop_dir_btn = gr.Button("🖥️ Desktop", size="sm")
                    documents_dir_btn = gr.Button("📄 Documents", size="sm")
                    home_dir_btn = gr.Button("🏠 Home", size="sm")
                
                # Manual path entry
                with gr.Row():
                    manual_path_input = gr.Textbox(
                        label="Enter/paste a custom path:",
                        placeholder="e.g., C:\\MyProjects\\Data or D:\\Research",
                        scale=4
                    )
                    create_dir_btn = gr.Button("➕ Create", size="sm", scale=1)
                
                # Status display for directory creation
                dir_status = gr.Markdown("", visible=False)
                
                # Current selection display
                selected_path_display = gr.Textbox(
                    label="Selected Path:",
                    interactive=False,
                    value=""
                )
                
                with gr.Row():
                    confirm_dir_btn = gr.Button("✅ Use This Directory", variant="primary")
                    cancel_dir_btn = gr.Button("❌ Cancel")
            
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
                current_dir_display = gr.Textbox(
                    label="Current Working Directory",
                    value=default_working_dir,
                    interactive=False,
                    lines=1
                )
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
                with gr.Row():
                    gr.Markdown("**Files**")
                    refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0)
                file_explorer = gr.FileExplorer(root_dir=default_working_dir, file_count="multiple", interactive=True)
                with gr.Column():
                    pdf_preview = PDF(visible=False, interactive=False)
                    img_preview = gr.Image(interactive=False, visible=False)
                    df_preview = gr.DataFrame(interactive=False, visible=False)
                    code_preview = gr.Code(language="markdown", interactive=False, visible=False)
                    text_preview = gr.Markdown(value="⬅️ Click a file to preview", visible=True)

        # --- Event Wiring ---
        # Working directory change handler
        working_dir_input.change(
            on_working_dir_change,
            [working_dir_input],
            [working_dir_state, file_explorer]
        )
        
        # Directory explorer functionality
        use_explorer_btn.click(
            show_directory_explorer,
            None,
            [dir_explorer_column, selected_path_display, test_label]
        )
        
        # Quick directory selection buttons
        current_dir_btn.click(
            lambda: select_quick_directory("current"),
            None,
            [selected_path_display]
        )
        
        desktop_dir_btn.click(
            lambda: select_quick_directory("desktop"),
            None,
            [selected_path_display]
        )
        
        documents_dir_btn.click(
            lambda: select_quick_directory("documents"),
            None,
            [selected_path_display]
        )
        
        home_dir_btn.click(
            lambda: select_quick_directory("home"),
            None,
            [selected_path_display]
        )
        
        # Manual path input
        manual_path_input.change(
            update_manual_path,
            [manual_path_input],
            [selected_path_display]
        )
        
        # Create directory button
        create_dir_btn.click(
            create_directory,
            [manual_path_input],
            [selected_path_display, dir_status]
        )
        
        cancel_dir_btn.click(
            hide_directory_explorer,
            None,
            [dir_explorer_column]
        )
        
        confirm_dir_btn.click(
            confirm_directory_selection,
            [selected_path_display],
            [working_dir_input, dir_explorer_column]
        )
        
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
            [config_state, tool_checkboxes, prompt_box, working_dir_state], 
            [config_page, chat_page, agent_state, launch_status, agent_title_md, current_dir_display]
        )
        file_explorer.change(
            show_preview, 
            [file_explorer, working_dir_state], 
            [pdf_preview, img_preview, df_preview, code_preview, text_preview]
        )
        file_uploader.change(
            handle_upload, 
            [file_uploader, file_uploads_log, working_dir_state], 
            [upload_status, file_uploads_log, file_explorer]
        )
        
        submit_button.click(
            interact_with_agent, 
            [chat_input, chatbot, agent_state, file_uploads_log], 
            [chatbot]
        ).then(
            lambda: gr.update(value=""), 
            outputs=[chat_input]
        ).then(
            force_refresh_files, 
            [working_dir_state], 
            [file_explorer]
        )
        
        chat_input.submit(
            interact_with_agent, 
            [chat_input, chatbot, agent_state, file_uploads_log], 
            [chatbot]
        ).then(
            lambda: gr.update(value=""), 
            outputs=[chat_input]
        ).then(
            force_refresh_files, 
            [working_dir_state], 
            [file_explorer]
        )

        back_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[config_page, chat_page])
        
        # Manual refresh button
        refresh_btn.click(
            force_refresh_files,
            [working_dir_state],
            [file_explorer]
        )
        
        # Wire periodic refresh timer
        refresh_timer.tick(
            periodic_refresh,
            [working_dir_state],
            [file_explorer]
        )
        
        # Initialize file watcher for the default working directory
        current_watching_dir = default_working_dir
        file_watcher.start_watching(default_working_dir, lambda: print("File system change detected by watcher."))
        
        # Add cleanup when demo closes (file watcher will auto-cleanup via __del__)
        
    return demo

if __name__ == "__main__":
    app_instance = app()
    print("App created successfully, attempting to launch...")
    app_instance.queue()  # Enable queue for better handling
    app_instance.launch(server_port=7862, share=False)
