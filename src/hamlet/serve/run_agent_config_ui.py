import sys
from pathlib import Path
import re
import os
import shutil
import time
import threading
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import uuid
import tempfile
import atexit
import pandas as pd
from gradio_pdf import PDF

# Add project root to sys.path to allow for relative imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure a writable temp/cache directory for Gradio to avoid /tmp permission issues
try:
    gradio_temp_dir = project_root / "gradio_temp"
    gradio_temp_dir.mkdir(exist_ok=True)
    # Environment variable respected by Gradio for temp storage
    os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)
    # Older versions may also look at GRADIO_CACHE_DIR
    os.environ.setdefault("GRADIO_CACHE_DIR", str(gradio_temp_dir))
    print(f"[GRADIO] Using custom temp dir: {gradio_temp_dir}")
except Exception as e:
    print(f"[GRADIO] Failed to set custom temp dir: {e}")

from hamlet.serve.agent_config_ui import AgentConfigManager, stream_to_gradio
from hamlet.core.models import LiteLLMModel
from hamlet.core.tools import Tool
from hamlet.core.monitoring import LogLevel
from hamlet.core.agents import CodeAgent
from hamlet.tools import get_available_tools, create_tool_instance, discover_tools
from hamlet.tools.text_web_browser.text_web_browser import SimpleTextBrowser

# Initialize the configuration manager for agents and tools
config_manager = AgentConfigManager()

# Retention (in seconds) for temporary ZIP bundles generated for download
BUNDLE_RETENTION_SECONDS = 60  # Adjust if you want bundles to persist longer

class FileWatcher:
    """A simple file system watcher using watchdog."""
    def __init__(self):
        self.observer = None
        self.event_handler = None
        self.callback = None
        
    def start_watching(self, directory: str, callback):
        """Start watching a directory for changes."""
        try:
            # Stop any existing watcher
            self.stop_watching()
            
            self.callback = callback
            self.event_handler = FileChangeHandler(callback)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, directory, recursive=True)
            self.observer.start()
            print(f"Started watching directory: {directory}")
        except Exception as e:
            print(f"Error starting file watcher: {e}")
            # Create a dummy watcher that doesn't crash
            self.observer = None
    
    def stop_watching(self):
        """Stop the file watcher."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join()
            except Exception as e:
                print(f"Error stopping file watcher: {e}")
            finally:
                self.observer = None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_watching()

class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events."""
    def __init__(self, callback):
        self.callback = callback
        self.last_event_time = 0
        
    def on_any_event(self, event):
        """Handle any file system event with debouncing."""
        current_time = time.time()
        # Debounce events - only trigger callback every 0.5 seconds
        if current_time - self.last_event_time > 0.5:
            self.last_event_time = current_time
            if self.callback:
                try:
                    self.callback()
                except Exception as e:
                    print(f"Error in file watcher callback: {e}")

# Initialize the file watcher
file_watcher = FileWatcher()

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
    """A factory to create tool instances from their names using dynamic discovery.

    Improvements:
    - Only pass working_dir to file / KB tools.
    - Provide a shared SimpleTextBrowser instance to browser tools so they are initialized (avoids '[visit_page] Browser not initialized.')
    - Retry constructor without working_dir on TypeError.
    - Inject API keys for search tools automatically.
    """

    _shared_browser = None
    _kb_indexers = {}

    @staticmethod
    def _get_repo_indexer() -> Optional[object]:  # late import to avoid startup cost
        """Return (and lazily create) a singleton RepoIndexer for the global knowledge base.

        Current strategy: one shared knowledge base rooted at ./knowledge_base with its
        vector store kept as a SIBLING directory ./vector_store (NOT nested inside
        knowledge_base) to make it clear the embedding index is an artifact and to
        avoid accidental deletion when users clean or replace the knowledge_base
        folder. This keeps KB content & index stable across user sessions while
        working directories remain per-session.

        If initialization fails (e.g. missing embedding key / network), a None is cached
        so we don't retry repeatedly; tools that depend on the indexer will become
        placeholders with an explanatory message.
        """
        kb_root = os.path.abspath("knowledge_base")
        os.makedirs(kb_root, exist_ok=True)
        key = kb_root
        if key in ToolFactory._kb_indexers:
            return ToolFactory._kb_indexers[key]
        try:
            from hamlet.tools.kb_repo_management.repo_indexer import RepoIndexer  # local import
            # Place vector store as sibling of knowledge_base (project_root/vector_store)
            index_dir = Path(kb_root).parent / "vector_store"
            index_dir.mkdir(exist_ok=True)
            api_key = os.getenv("OPENAI_API_KEY_EMBEDDINGS") or os.getenv("OPENAI_API_KEY")
            ri = RepoIndexer(
                root=kb_root,
                watch=False,
                index_dir=index_dir,
                openai_api_key=api_key or "",  # RepoIndexer internally probes embeddings; empty triggers env fallback
            )
            ToolFactory._kb_indexers[key] = ri
            print(f"[ToolFactory] RepoIndexer initialized for KB root: {kb_root} | index_dir: {index_dir}")
            return ri
        except Exception as e:
            print(f"[ToolFactory] Failed to initialize RepoIndexer (will use stub): {e}")
            class _StubIndexer:
                def __init__(self, root):
                    self.root = Path(root)
                def update_file(self, *_a, **_k):
                    pass
                def get_query_results(self, query: str, k: int = 3):
                    return f"[StubIndexer] Semantic search unavailable (no embeddings). Query='{query}'"
                def get_unique_query_results(self, query: str, k: int = 3):
                    return self.get_query_results(query, k)
            stub = _StubIndexer(kb_root)
            ToolFactory._kb_indexers[key] = stub
            return stub

    @staticmethod
    def create_tool(tool_name: str, working_dir: str = "working_directory") -> Optional[Tool]:
        print(f"Creating tool: {tool_name} with working_dir: {working_dir}")
        # Classification sets
        file_tools = {
            'list_dir', 'see_file', 'see_text_file', 'read_binary_as_markdown', 'modify_file',
            'create_file_with_content', 'search_keyword', 'delete_file_or_folder', 'load_object_from_python_file',
            'get_paper_from_url'
        }
        kb_tools = {
            'delete_from_knowledge_base', 'list_knowledge_base_directory', 'move_or_rename_in_knowledge_base',
            'see_knowledge_base_file', 'append_to_knowledge_base_file', 'copy_to_knowledge_base',
            'write_to_knowledge_base', 'copy_from_knowledge_base', 'keyword_search_knowledge_base',
            'semantic_search_knowledge_base'
        }
        browser_tools = {
            'visit_page', 'page_up', 'page_down', 'find_on_page_ctrl_f', 'find_next',
            'download_file', 'find_archived_url', 'simple_web_search'
        }

        ctor_kwargs: dict = {}
        # working dir only for local file tools; KB tools handle working_dir selectively
        if tool_name in file_tools:
            ctor_kwargs['working_dir'] = working_dir

        # Inject RepoIndexer for KB tools
        if tool_name in kb_tools:
            repo_indexer = ToolFactory._get_repo_indexer()
            if repo_indexer is None:
                print(f"[ToolFactory] WARNING: RepoIndexer unavailable; creating placeholder for {tool_name}.")
            else:
                ctor_kwargs['repo_indexer'] = repo_indexer
                # Only the copy in/out tools also expect a working_dir parameter
                if tool_name in {'copy_to_knowledge_base', 'copy_from_knowledge_base'}:
                    ctor_kwargs['working_dir'] = working_dir

        # Shared browser (lazy)
        if tool_name in browser_tools:
            if ToolFactory._shared_browser is None:
                try:
                    serp_key = os.getenv('SERPAPI_KEY') or os.getenv('SERPAPI_API_KEY') or None
                    ToolFactory._shared_browser = SimpleTextBrowser(
                        downloads_folder=os.path.join(working_dir, 'downloads'),
                        serpapi_key=serp_key
                    )
                    print('[ToolFactory] Created shared SimpleTextBrowser.')
                except Exception as e:
                    print(f"[ToolFactory] Failed to create shared browser: {e}")
            ctor_kwargs['browser'] = ToolFactory._shared_browser

        # API key injections
        if tool_name == 'web_search':
            serper_key = os.getenv('SERPER_API_KEY') or os.getenv('SERPER_APIKEY') or os.getenv('SERPER_KEY')
            if serper_key:
                ctor_kwargs['serper_api_key'] = serper_key
            jina_key = os.getenv('JINA_API_KEY')
            if jina_key and 'reranker' not in ctor_kwargs:
                ctor_kwargs['reranker'] = 'jina'
        elif tool_name == 'simple_web_search':
            serp_key = os.getenv('SERPAPI_KEY') or os.getenv('SERPAPI_API_KEY')
            if serp_key:
                ctor_kwargs['serpapi_key'] = serp_key

        def _attempt(kwargs):
            return create_tool_instance(tool_name, **kwargs)

        try:
            try:
                tool = _attempt(ctor_kwargs)
            except TypeError as te:
                te_msg = str(te)
                # Case 1: working_dir was provided but not accepted -> retry without it
                if 'working_dir' in ctor_kwargs and ('working_dir' in te_msg or 'unexpected keyword argument' in te_msg):
                    removed = ctor_kwargs.pop('working_dir')
                    print(f"[ToolFactory] Retrying {tool_name} without working_dir={removed} due to TypeError: {te_msg}")
                    tool = _attempt(ctor_kwargs)
                # Case 2: missing required working_dir arg (not passed initially) -> inject and retry
                elif 'working_dir' not in ctor_kwargs and ('missing required positional argument' in te_msg and 'working_dir' in te_msg):
                    ctor_kwargs['working_dir'] = working_dir
                    print(f"[ToolFactory] Retrying {tool_name} adding working_dir due to TypeError: {te_msg}")
                    tool = _attempt(ctor_kwargs)
                else:
                    raise

            if hasattr(tool, 'setup'):
                try:
                    tool.setup()
                except Exception as e:
                    print(f"[ToolFactory] Setup for {tool_name} failed: {e}")

            if not hasattr(tool, 'name') or tool.name is None:
                tool.name = tool_name
                print(f"Fixed missing name for tool: {tool_name}")
            if not hasattr(tool, 'inputs'):
                tool.inputs = {"input": {"type": "string", "description": f"Input for {tool_name}"}}
                print(f"Added missing inputs for tool: {tool_name}")
            if not hasattr(tool, 'output_type'):
                tool.output_type = 'string'
                print(f"Added missing output_type for tool: {tool_name}")

            # Extra validation / diagnostics for KB tools producing base Tool behavior
            if tool_name in kb_tools and type(tool).__name__ == 'Tool':
                print(f"[ToolFactory][WARN] KB tool '{tool_name}' instantiated as base Tool (likely constructor mismatch). Kwargs used: {ctor_kwargs}")
            return tool
        except ValueError as e:
            print(f"Tool '{tool_name}' not found: {e}")
            placeholder = Tool(
                name=tool_name,
                description=f"Placeholder for unknown tool: {tool_name}",
                func=lambda x, tn=tool_name: f"[{tn}] Tool not available - placeholder executed with: {x}"
            )
            placeholder.inputs = {"input": {"type": "string", "description": f"Input for placeholder {tool_name}"}}
            placeholder.output_type = 'string'
            return placeholder
        except Exception as e:
            print(f"Error creating tool {tool_name}: {e}. Placeholder created.")
            err_tool = Tool(
                name=tool_name,
                description=f"Error placeholder for {tool_name}",
                func=lambda x, tn=tool_name: f"[{tn}] Error creating tool - placeholder executed with: {x}"
            )
            err_tool.inputs = {"input": {"type": "string", "description": f"Input for error placeholder {tool_name}"}}
            err_tool.output_type = 'string'
            return err_tool

def build_agent(name: str, system_prompt: str, tools: list, sub_agents: list, agent_type: str = "CodeAgent", working_dir: Optional[str] = None, description: str = ""):
    """Recursively builds a multi-step agent from its configuration.

    Parameters:
        name: Display / logical name of the agent.
        system_prompt: User-specified system prompt (will be APPENDED to existing agent.system_prompt after instantiation).
        tools: List of tool names.
        sub_agents: List of sub-agent names.
        agent_type: 'ToolCallingAgent' or 'CodeAgent'.
        working_dir: Working directory path for file operations.
        description: Human readable description to set on the agent (shown in UI).
    """
    if working_dir is None:
        working_dir = os.path.abspath("working_directory")

    # Sanitize agent name
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    if sanitized_name and sanitized_name[0].isdigit():
        sanitized_name = 'agent_' + sanitized_name
    if not sanitized_name or sanitized_name == '_':
        sanitized_name = 'unnamed_agent'

    print(f"Original name: '{name}' -> Sanitized name: '{sanitized_name}'")
    print(f"Building agent with working directory: {working_dir}")

    # Force refresh tool discovery to avoid stale cache after code edits
    try:
        discover_tools(force_refresh=True)
    except Exception as e:
        print(f"[build_agent] Tool rediscovery failed: {e}")

    tool_objs: list[Tool] = []
    for tool_name in tools:
        if not tool_name:
            continue
        try:
            tool = ToolFactory.create_tool(tool_name, working_dir)
            if tool is not None:
                if not hasattr(tool, 'name') or tool.name is None:
                    tool.name = tool_name
                    print(f"Fixed missing name for tool: {tool_name}")
                if not hasattr(tool, 'description'):
                    tool.description = f"Tool for {tool_name}"
                tool_objs.append(tool)
                print(f"Successfully created tool: {tool.name}")
            else:
                print(f"Warning: Failed to create tool {tool_name}")
        except Exception as e:
            print(f"Error creating tool {tool_name}: {e}")
            fallback_tool = Tool(
                name=tool_name,
                description=f"Fallback tool for {tool_name}",
                func=lambda x, tn=tool_name: f"[{tn}] Executed with: {x}"
            )
            fallback_tool.inputs = {
                "input": {"type": "string", "description": f"Input for {tool_name}"}
            }
            fallback_tool.output_type = "string"
            tool_objs.append(fallback_tool)

    sub_agent_objs = []
    for sa_name in sub_agents:
        md = config_manager.get_all_agent_metadata().get(sa_name, {})
        sub_agent_objs.append(build_agent(
            name=sa_name,
            system_prompt=md.get("prompt", ""),
            tools=md.get("tools", []),
            sub_agents=md.get("sub_agents", []),
            agent_type=md.get("agent_type", "CodeAgent"),
            working_dir=working_dir,
            description=md.get("description", "")
        ))

    AgentClass = CodeAgent
    try:
        print("Creating LiteLLMModel...")
        model = LiteLLMModel(
            model_id="gpt-4.1",
            timeout=30
        )
        print("LiteLLMModel instantiated successfully")
    except Exception as e:
        raise Exception(f"Failed to instantiate LiteLLMModel. Network timeout or API key issue. Error: {e}")

    print(f"Creating agent {sanitized_name} (original: {name}) with {len(tool_objs)} tools")
    for tool in tool_objs:
        print(f"  - Tool: {getattr(tool, 'name', 'UNNAMED')} (type: {type(tool).__name__})")

    print("Creating agent instance...")
    agent = AgentClass(
        tools=tool_objs,
        model=model,
        managed_agents=sub_agent_objs,
        verbosity_level=LogLevel.DEBUG,
        planning_interval=3,
        max_steps=30,
        name=sanitized_name,
        description=description or f"A {agent_type} agent with {len(tool_objs)} tools."
    )
    print(f"Agent instance created successfully with name: {sanitized_name}")
    # Preserve original (may contain non-ASCII characters) for UI display
    try:
        agent.display_name = name  # type: ignore[attr-defined]
    except Exception:
        pass

    # Append user system prompt AFTER instantiation
    try:
        if system_prompt:
            base_sp = getattr(agent, 'system_prompt', '') or ''
            new_sp = (base_sp.rstrip() + "\n" + system_prompt.strip()).strip()
            agent.system_prompt = new_sp
            print(f"Appended system prompt for agent {sanitized_name}. Length now: {len(agent.system_prompt)}")
    except Exception as e:
        print(f"Failed to append system prompt for {sanitized_name}: {e}")

    # Ensure description attribute matches provided description (redundant safety)
    try:
        if description:
            agent.description = description
    except Exception as e:
        print(f"Failed to set description for {sanitized_name}: {e}")

    # (Removed) previously appended sub-agent descriptions manually; now rely on underlying framework behavior.
    # Inject simple callable wrappers for each tool so that within code execution
    # a user can call tool_name(arg1, ...) instead of needing agent.run or manual forward.
    try:
        wrapper_count = 0
        for t in tool_objs:
            tname = getattr(t, 'name', None)
            if not tname or not isinstance(tname, str):
                continue
            # Skip if name not a valid identifier (avoid exec issues)
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', tname):
                continue
            # Create a small wrapper function capturing the tool
            def _make_wrapper(tool_ref):  # closure factory
                def _wrapper(*args, **kwargs):
                    # Prefer forward if available, else fallback to callable interface
                    if hasattr(tool_ref, 'forward'):
                        try:
                            return tool_ref.forward(*args, **kwargs)
                        except TypeError:
                            # Some tools expect named args; fallback to passing first positional as 'input'
                            if len(args) == 1 and not kwargs and 'input' in getattr(tool_ref, 'inputs', {}):
                                return tool_ref.forward(args[0])  # type: ignore
                            raise
                    # Fallback: if Tool base exposes a func attribute
                    func_attr = getattr(tool_ref, 'func', None)
                    if callable(func_attr):
                        return func_attr(*args, **kwargs)
                    return f"[{tname}] Tool not callable."
                return _wrapper
            wrapper = _make_wrapper(t)
            # Attach to agent namespace (if CodeAgent supports custom locals)
            try:
                # For CodeAgent, maintain an execution namespace attribute
                existing = getattr(agent, '_exec_namespace', None)
                if existing is None:
                    existing = {}
                    setattr(agent, '_exec_namespace', existing)
                existing[tname] = wrapper
                wrapper_count += 1
            except Exception as e:
                print(f"Failed to inject wrapper for tool {tname}: {e}")
        if wrapper_count:
            print(f"Injected {wrapper_count} tool wrappers into agent execution namespace.")
    except Exception as e:
        print(f"Wrapper injection error: {e}")

    return agent
    
    # Unreachable (return above) -- keep logic below if future refactor moves return.

    # Inject callable wrappers (this block currently unreachable due to early return)
    # Keeping for reference.


def app():
    """Main function to create and configure the Gradio application.

    Updated behavior:
    - Each user/session receives an isolated temporary working directory that is NOT selectable.
    - When the browser session ends (or app exits), the directory is cleaned up.
    - Agent configurations are per-user: a user identifier is derived from Gradio's request headers / cookies.
    - The previous manual directory picker UI is removed.
    """
    # Mapping session -> temp directory for cleanup
    session_temp_dirs = {}

    # Create per-session working directories inside project root so Gradio FileExplorer can access them.
    base_workspace_dir = os.path.join(project_root, "session_workspaces")
    os.makedirs(base_workspace_dir, exist_ok=True)

    def _create_temp_working_dir(session_id: str):
        if session_id in session_temp_dirs:
            return session_temp_dirs[session_id]
        # Use deterministic path inside project to avoid /tmp visibility issues in FileExplorer
        temp_dir = os.path.join(base_workspace_dir, f"ws_{session_id[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        session_temp_dirs[session_id] = temp_dir
        print(f"[SESSION {session_id}] Created session working dir: {temp_dir}")
        return temp_dir

    def _cleanup_temp_dir(path: str):
        if os.path.exists(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
                print(f"Cleaned temp dir: {path}")
            except Exception as e:
                print(f"Failed to clean temp dir {path}: {e}")

    # Ensure global cleanup at process exit
    atexit.register(lambda: [
        _cleanup_temp_dir(p) for p in list(session_temp_dirs.values())
    ])

    # Helper to derive user id from request (very lightweight). If none, anonymous session id used.
    def _derive_user_id(request: gr.Request | None) -> str:
        """Derive a stable user id prioritizing a persistent cookie.

        Order:
        1. Valid 'hamlet_user_id' cookie (UUID v4 format or UUID.signature)
        2. Fallback hash(IP + UA)
        3. 'anonymous' if request unavailable
        """
        if request is None:
            return "anonymous"
        cookie_id = None
        try:
            cookie_id = request.cookies.get("hamlet_user_id") if request.cookies else None
        except Exception:
            cookie_id = None
        if cookie_id:
            import re
            # Accept either plain UUID v4 or signed variant uuid.hex.signature
            uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(\.[0-9a-f]{64})?$")
            if uuid_re.match(cookie_id):
                # If signed form present, strip signature for logical user id
                base_uuid = cookie_id.split('.', 1)[0]
                return base_uuid
        # Fallback: hash IP + user-agent
        import hashlib
        raw = f"{request.client.host}|{request.headers.get('user-agent','')}"  # type: ignore[attr-defined]
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]

    # Generate / ensure a persistent cookie value for user id
    def _ensure_cookie_user_id(request: gr.Request | None):
        user_id = _derive_user_id(request)
        config_manager.set_user(user_id)
        return user_id
    
    # Removed manual directory setup logic; now per-session temp directories are used.

    # --- ALL EVENT HANDLERS DEFINED FIRST ---

    # Removed: on_working_dir_change (no longer exposed to UI)

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
            return gr.update(value=f"❌ 错误：{str(e)}"), gr.update(visible=True)

    def on_agent_select(agent_name: str):
        """Handles agent selection dropdown change."""
        if agent_name in ("Create new agent", "创建新智能体"):
            # Build current agent list for sub-agent choices
            all_agent_names = sorted(list(config_manager.get_all_agent_metadata().keys()))
            sub_agent_choices = all_agent_names  # none excluded since we're creating new
            state = {"name": "", "tools": [], "managed_agents": [], "prompt": "", "description": ""}
            return (
                state,            # config_state
                "",              # agent_name_input
                "",              # description input
                [],               # tool selection (hidden internal)
                gr.update(choices=sub_agent_choices, value=[]),  # sub_agent_checkboxes sanitized
                "",              # prompt
                gr.update(visible=True)  # show form
            )
        md = config_manager.get_all_agent_metadata().get(agent_name, {})
        state = {
            "name": agent_name,
            "tools": md.get("tools", []),
            "managed_agents": md.get("sub_agents", []),
            "prompt": md.get("prompt", ""),
            "description": md.get("description", "")
        }
        # Recompute valid sub-agent choices (exclude self to prevent recursion)
        all_agent_names = sorted(list(config_manager.get_all_agent_metadata().keys()))
        sub_agent_choices = [a for a in all_agent_names if a != agent_name]
        valid_selected_subs = [sa for sa in state["managed_agents"] if sa in sub_agent_choices]
        return (
            state,                      # config_state
            state["name"],              # agent_name_input
            state["description"],       # description input
            state["tools"],             # tool selection (hidden internal)
            gr.update(choices=sub_agent_choices, value=valid_selected_subs),  # sanitized sub-agents
            state["prompt"],
            gr.update(visible=True)
        )

    def save_config(name: str, description: str, tools: list, sub_agents: list, prompt: str):
        """Saves the agent's configuration."""
        if not name:
            return gr.update(), gr.update(), gr.update(value="智能体的名字不能为空")
        
        all_configs = config_manager.get_all_agent_metadata()
        all_configs[name] = {"prompt": prompt, "description": description, "tools": tools, "sub_agents": sub_agents, "agent_type": "CodeAgent"}
        config_manager.save_agent_configs(all_configs)
        
        # Re-read to ensure persistence and build sorted list for stable UX
        all_agent_names = sorted(list(config_manager.get_all_agent_metadata().keys()))
        # Sub-agent choices should not include the sentinel
        sub_agent_choices = [a for a in all_agent_names if a != name]
        sentinel = "创建新智能体"
        return (
            gr.update(choices=all_agent_names + [sentinel], value=name if name else sentinel),  # Update main selector
            gr.update(choices=sub_agent_choices, value=[sa for sa in sub_agents if sa in sub_agent_choices]),  # Update sub-agent checkbox group
            f"已保存 '{name}'"
        )

    def delete_agent(name: str):
        """Delete an existing agent configuration and remove it from any sub-agent lists.

        Returns updates for: agent_selector, sub_agent_checkboxes, launch_status
        Follow-up chain will refresh the form via on_agent_select.
        """
        sentinel = "创建新智能体"
        if not name or name == sentinel or name in ("Create new agent"):
            return gr.update(), gr.update(), "⚠️ 请选择要删除的已存在智能体。"
        all_configs = config_manager.get_all_agent_metadata()
        if name not in all_configs:
            # Nothing to delete
            all_agent_names = sorted(list(all_configs.keys()))
            return (
                gr.update(choices=all_agent_names + [sentinel], value=sentinel),
                gr.update(choices=all_agent_names, value=[]),
                f"❌ 未找到智能体 '{name}'"
            )
        # Remove the agent
        del all_configs[name]
        # Remove from any sub-agent lists
        changed = False
        for ag_name, meta in all_configs.items():
            subs = meta.get("sub_agents", [])
            if name in subs:
                meta["sub_agents"] = [s for s in subs if s != name]
                changed = True
        if changed:
            config_manager.save_agent_configs(all_configs)
        else:
            # Save after deletion even if no sub-agent change
            config_manager.save_agent_configs(all_configs)
        # Recompute choices
        remaining = sorted(list(all_configs.keys()))
        return (
            gr.update(choices=remaining + [sentinel], value=sentinel),
            gr.update(choices=remaining, value=[]),
            f"🗑️ 已删除智能体 '{name}'"
        )

    def launch_agent(state: dict, current_description: str, current_tools: list, current_prompt: str, current_working_dir: str):
        """Launches the agent and switches to the chat UI."""
        if not state.get("name"):
            return gr.update(), gr.update(), None, "⚠️ 请先选择、创建并保存一个智能体。", gr.update(), gr.update(), gr.update()
        
        # Test API key first
        api_working, api_message = test_api_key()
        if not api_working:
            return gr.update(), gr.update(), None, f"🔑 API 密钥错误：{api_message}", gr.update(), gr.update()
        
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
                system_prompt=current_prompt,  # Append to base system prompt
                tools=current_tools,
                sub_agents=state["managed_agents"],
                agent_type=full_config.get("agent_type", "CodeAgent"),
                working_dir=current_working_dir,
                description=current_description or full_config.get("description", "")
            )
            print(f"Agent {agent_name} built successfully")
            # Override description field if provided
            try:
                agent.description = getattr(agent, 'description', '')
                # Attach session working dir for later file operations
                setattr(agent, '_session_working_dir', current_working_dir)
                # --- Debug prints for system prompt & description ---
                try:
                    sp = getattr(agent, 'system_prompt', '(no system_prompt attribute)')
                except Exception as _e_sp:
                    sp = f"<error retrieving system_prompt: {_e_sp}>"
                try:
                    desc_dbg = getattr(agent, 'description', '(no description attribute)')
                except Exception as _e_desc:
                    desc_dbg = f"<error retrieving description: {_e_desc}>"
                print("[AGENT DEBUG] ================= SYSTEM PROMPT BEGIN ================")
                print(sp)
                print("[AGENT DEBUG] ================= SYSTEM PROMPT END ==================")
                print("[AGENT DEBUG] Description:", desc_dbg if desc_dbg else "(empty description)")
                print(f"[AGENT DEBUG] System prompt length: {len(sp) if isinstance(sp, str) else 'N/A'} chars")
            except Exception:
                pass
            desc_display = current_description or "（无描述）"
            # Ensure file explorer will point to the working directory
            file_explorer_update = gr.update(root_dir=current_working_dir)
            return (
                gr.update(visible=False),  # hide config page
                gr.update(visible=True),   # show chat page
                agent,                     # agent state
                "✅ 智能体启动成功！",  # status (localized)
                # Prefer original display_name (supports中文) if present
                gr.update(value=f"## {getattr(agent, 'display_name', agent.name)}"),  # title markdown
                gr.update(value=desc_display),         # description markdown
                file_explorer_update                   # file explorer root refresh
            )
        except Exception as e:
            print(f"Error in launch_agent: {e}")
            import traceback
            traceback.print_exc()
            return gr.update(), gr.update(), None, f"❌ 构建智能体失败：{e}", gr.update(), gr.update(), gr.update()

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
            gr.update(value=f"已上传 {filename}", visible=True), 
            upload_log + [filename], 
            force_refresh_files(working_dir)
        )

    def force_refresh_files(_=None):
        """Force refresh FileExplorer by tweaking ignore_glob with a random pattern (matches nothing)."""
        dummy_glob = f"__refresh_{int(time.time()*1000)}__"
        return gr.update(ignore_glob=dummy_glob)

    # --- Added helper functions for debug directory listing ---
    # Removed alternate listings & fallback dropdown per user request; rely solely on FileExplorer + ignore_glob refresh.

    # Removed flat file list/dropdown helpers to simplify UI; FileExplorer alone will be used.
    
    # Global variable for file watcher state
    current_watching_dir = None
    
    def periodic_refresh(working_dir_state):
        """Periodic refresh to catch any missed file changes."""
        # This is a reliable catch-all. It runs every 3 seconds.
        if working_dir_state and os.path.exists(working_dir_state):
            print("Periodic refresh triggered.")
            return force_refresh_files(working_dir_state)
        return gr.update()

    # Simplified preview (adapted from agent_conversation_ui.py)
    def show_preview(selection: list, working_dir: str):
        """Return component updates so the appropriate viewer shows the file + a proper download component."""
        hidden = gr.update(visible=False)
        hidden_file = gr.update(value=None, visible=False)
        if not selection:
            return hidden, hidden, hidden, gr.update(value="⬅️ 点击文件进行预览", visible=True), hidden_file

        # Normalize to list
        if not isinstance(selection, list):
            selection = [selection]
        # Remove duplicates, preserve order
        seen = set()
        selection = [s for s in selection if not (s in seen or seen.add(s))]

        # Sanitize: convert any absolute path under working_dir to relative; ignore paths outside
        sanitized = []
        for p in selection:
            if os.path.isabs(p):
                try:
                    rel = os.path.relpath(p, working_dir)
                    # Disallow escaping via ..
                    if rel.startswith('..'):
                        # Skip anything outside working_dir
                        continue
                    sanitized.append(rel)
                except Exception:
                    continue
            else:
                # Normalize path (remove leading ./)
                norm = p[2:] if p.startswith('./') else p
                sanitized.append(norm)
        selection = sanitized
        if not selection:
            return hidden, hidden, hidden, gr.update(value="⚠️ 未选择有效的文件", visible=True), hidden_file
        # If multiple items or any directory selected, just show a summary message (do NOT auto-zip)
        if len(selection) > 1 or any(os.path.isdir(os.path.join(working_dir, p)) for p in selection):
            summary_lines = ["📂 已选择以下项目（尚未打包，点击‘打包/下载所选’按钮进行下载）："] + [f"- {p}" for p in selection]
            msg = "\n".join(summary_lines)
            return hidden, hidden, hidden, gr.update(value=msg, visible=True), hidden_file

        # Single selection case
        rel_path = selection[0]
        abs_path = os.path.join(working_dir, rel_path)
        if os.path.isdir(abs_path):  # single directory; show message, wait for user to click download
            return hidden, hidden, hidden, gr.update(value=f"📁 已选择目录: {rel_path}（点击‘打包/下载所选’按钮进行打包下载）", visible=True), hidden_file

        download_update = gr.update(value=abs_path, visible=True) if os.path.exists(abs_path) else hidden_file
        ext = os.path.splitext(abs_path)[1].lower()
        # Images
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            return hidden, gr.update(value=abs_path, visible=True), hidden, hidden, download_update
        # PDF
        if ext == ".pdf":
            return gr.update(value=abs_path, visible=True), hidden, hidden, hidden, download_update
        # Tables
        if ext in {".csv", ".tsv", ".xlsx"}:
            try:
                df = (pd.read_csv if ext != ".xlsx" else pd.read_excel)(abs_path)
                return hidden, hidden, gr.update(value=df, visible=True), hidden, download_update
            except Exception as e:
                return hidden, hidden, hidden, gr.update(value=f"⚠️ 无法读取文件: {e}", visible=True), download_update
        # Text fallback
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read(20000)
        except Exception as e:
            txt = f"⚠️ 无法显示文件: {e}"
        return hidden, hidden, hidden, gr.update(value=txt, visible=True), download_update

    def interact_with_agent(prompt: str, history: list, agent, uploads: list):
        """Handles the chat interaction with the agent."""
        import gradio as gr
        import os
        prev_cwd = None
        session_dir = getattr(agent, '_session_working_dir', None) if agent else None
        # We avoid global chdir unless really needed; but if user wants relative writes to appear, temporarily chdir
        if session_dir and os.path.isdir(session_dir):
            try:
                prev_cwd = os.getcwd()
                os.chdir(session_dir)
                print(f"[INTERACT] Temporarily switched cwd to {session_dir}")
            except Exception as _e:
                print(f"[INTERACT] Failed to chdir to session dir: {_e}")
        
        if not agent:
            history.append(gr.ChatMessage(role="user", content=prompt))
            history.append(gr.ChatMessage(role="assistant", content="❌ 错误：智能体尚未加载。"))
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
            
            timeout_seconds = 1200  # Extended overall timeout for slower networks / longer first token latency
            inactivity_grace = 120   # If no new message for this many seconds, warn user
            start_time = time.time()
            last_progress_time = start_time
            message_count = 0
            warned_inactivity = False
            
            while agent_thread.is_alive() and (time.time() - start_time) < timeout_seconds:
                # Process any new messages
                new_data = False
                while message_count < len(result["messages"]):
                    msg = result["messages"][message_count]
                    message_count += 1
                    new_data = True
                    last_progress_time = time.time()
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

                # Inactivity warning (no new tokens/messages)
                now = time.time()
                if not new_data and (now - last_progress_time) > inactivity_grace and not warned_inactivity:
                    warned_inactivity = True
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=(
                            "⏳ **正在等待模型响应...**\n\n已超过 "
                            f"{int(inactivity_grace)} 秒没有新输出。可能是：\n"
                            "- 网络较慢或首次响应延迟\n"
                            "- 模型端正在排队\n"
                            "- 即将触发超时 (仍在尝试)\n\n"
                            "如果持续无响应，可稍后重试或简化问题。"
                        ),
                        metadata={"status": "pending"}
                    ))
                    yield history

                time.sleep(0.15)  # Slightly longer sleep reduces CPU load
            
            # Check if we timed out
            if agent_thread.is_alive():
                elapsed = int(time.time() - start_time)
                error_msg = (
                    "⏰ **请求超时**\n\n"
                    f"在 {elapsed} 秒内未完成响应。\n\n"
                    "**可能原因**：\n"
                    "- 首字节延迟过长（模型/网络阻塞）\n"
                    "- 当前网络出口限制或不稳定\n"
                    "- API 服务端排队时间过长\n\n"
                    "**建议操作**：\n"
                    "1. 检查/切换网络或代理 (VPN)\n"
                    "2. 简化问题或减少上下文再试\n"
                    "3. 若仍失败，查看服务状态：https://status.openai.com\n\n"
                    "*(English)* Request timed out after extended waiting. Consider retrying with a simpler query or improved connectivity."
                )
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
            if "timeout" in error_msg.lower() or "connecttimeout" in error_msg.lower():
                error_msg = """🌐 **网络连接错误**

无法连接到 OpenAI 服务器，可能原因：
- 网络不稳定或被阻断
- 防火墙/安全软件拦截
- 地域访问限制
- 需要使用 / 切换 VPN

**建议：**
- 检查网络与代理设置
- 放行或暂时关闭防火墙阻断规则
- 在受限地区使用合规 VPN
- 查看服务状态：https://status.openai.com

*(English)* Network Connection Error: Cannot reach OpenAI servers."""
            elif "api" in error_msg.lower() and "key" in error_msg.lower():
                error_msg += "\n\n💡 **API Key Issue:** Check your OPENAI_API_KEY in the .env file"
            
            history.append(gr.ChatMessage(role="assistant", content=f"❌ **错误：** {error_msg}"))
            yield history
        finally:
            if prev_cwd:
                try:
                    os.chdir(prev_cwd)
                    print(f"[INTERACT] Restored cwd to {prev_cwd}")
                except Exception as _e:
                    print(f"[INTERACT] Failed to restore cwd: {_e}")
            # After interaction, list session dir for debugging visibility issues
            if session_dir and os.path.isdir(session_dir):
                try:
                    print(f"[INTERACT] Post-interaction dir listing for {session_dir}: {os.listdir(session_dir)}")
                except Exception as e:
                    print(f"[INTERACT] Could not list session dir {session_dir}: {e}")

    def prepare_download(selection: list, working_dir: str):
        """On-demand packaging logic. If multiple files or any directory in selection -> zip now, else return single file.

        Returns a gr.File update (value=path, visible=True) or a status message in text preview if nothing selected.
        """
        if not selection:
            return gr.update(value=None, visible=False), gr.update(value="⚠️ 未选择任何文件或目录", visible=True)
        if not isinstance(selection, list):
            selection = [selection]
        # Deduplicate
        seen = set(); selection = [s for s in selection if not (s in seen or seen.add(s))]
        # Sanitize as in preview
        sanitized = []
        for p in selection:
            if os.path.isabs(p):
                try:
                    rel = os.path.relpath(p, working_dir)
                    if rel.startswith('..'):
                        continue
                    sanitized.append(rel)
                except Exception:
                    continue
            else:
                sanitized.append(p[2:] if p.startswith('./') else p)
        selection = sanitized
        if not selection:
            return gr.update(value=None, visible=False), gr.update(value="⚠️ 未找到可打包的文件", visible=True)
        # Any directory OR more than one item -> zip
        multi_or_dir = len(selection) > 1 or any(os.path.isdir(os.path.join(working_dir, p)) for p in selection)
        if multi_or_dir:
            try:
                import zipfile, hashlib, time
                ts = int(time.time()*1000)
                key = "|".join(sorted(selection))
                digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]
                zip_dir = os.path.join(working_dir, "_bundles"); os.makedirs(zip_dir, exist_ok=True)
                zip_path = os.path.join(zip_dir, f"bundle_{digest}_{ts}.zip")
                with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for rel in selection:
                        abs_p = os.path.join(working_dir, rel)
                        if not os.path.exists(abs_p):
                            continue
                        if os.path.isdir(abs_p):
                            for root, dirs, files in os.walk(abs_p):
                                for f in files:
                                    fp = os.path.join(root, f)
                                    arc = os.path.relpath(fp, working_dir)
                                    try: zf.write(fp, arc)
                                    except Exception as ie: print(f"[ZIP] skip {fp}: {ie}")
                        else:
                            try: zf.write(abs_p, rel)
                            except Exception as ie: print(f"[ZIP] skip {abs_p}: {ie}")
                # Schedule deletion after retention period so download can complete
                def _del_later(path=zip_path):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                            print(f"[BUNDLE] Auto-deleted bundle: {path}")
                    except Exception as de:
                        print(f"[BUNDLE] Failed to auto-delete {path}: {de}")
                try:
                    t = threading.Timer(BUNDLE_RETENTION_SECONDS, _del_later)
                    t.daemon = True
                    t.start()
                    print(f"[BUNDLE] Scheduled deletion in {BUNDLE_RETENTION_SECONDS}s: {zip_path}")
                except Exception as se:
                    print(f"[BUNDLE] Failed to schedule deletion: {se}")
                info_msg = gr.update(value=f"🔐 ZIP 文件已生成（将在 {BUNDLE_RETENTION_SECONDS} 秒后自动删除）", visible=True)
                return gr.update(value=zip_path, visible=True), info_msg
            except Exception as e:
                return gr.update(value=None, visible=False), gr.update(value=f"⚠️ 打包失败: {e}", visible=True)
        # Single file (not directory)
        rel = selection[0]
        abs_path = os.path.join(working_dir, rel)
        if os.path.isdir(abs_path):
            return gr.update(value=None, visible=False), gr.update(value="⚠️ 目录为空或无法访问", visible=True)
        if not os.path.exists(abs_path):
            return gr.update(value=None, visible=False), gr.update(value="⚠️ 文件不存在", visible=True)
        return gr.update(value=abs_path, visible=True), gr.update()

    # --- GRADIO UI LAYOUT AND WIRING ---

    with gr.Blocks(title="工业工程学科引擎-构建专属智能体", fill_height=True) as demo:
        # Use load event to initialize per-user temp directory
        user_id_state = gr.State("")
        session_dir_state = gr.State("")
        
        # --- State Management ---
        config_state = gr.State({"name": "", "tools": [], "managed_agents": [], "prompt": "", "description": ""})
        agent_state = gr.State(None)
        file_uploads_log = gr.State([])
        working_dir_state = gr.State("")  # Will be set on load
        
        # Add a simple periodic refresh timer
        refresh_timer = gr.Timer(value=3.0)  # Check every 3 seconds

        # --- PAGE 1: AGENT CONFIGURATION ---
        with gr.Column(visible=True) as config_page:
            # Inject a small script to set a persistent cookie-based UUID on first visit
            gr.HTML(
                """
<script>
(function(){
    function getCookie(name){
        const m = document.cookie.match(new RegExp('(^| )'+name+'=([^;]+)'));
        return m?decodeURIComponent(m[2]):null;
    }
    if(!getCookie('hamlet_user_id')){
        const uuid = (self.crypto && crypto.randomUUID) ? crypto.randomUUID() : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c){const r=Math.random()*16|0,v=c==='x'?r:(r&0x3|0x8);return v.toString(16);});
        const oneYear = 60*60*24*365;
        document.cookie = 'hamlet_user_id=' + encodeURIComponent(uuid) + '; Path=/; Max-Age='+oneYear+'; SameSite=Lax';
        // Optional: trigger a soft notice (no auto-reload to avoid loop)
        console.log('Assigned new hamlet_user_id cookie', uuid);
    }
})();
 </script>
"""
            )
        with gr.Column(visible=True) as config_page:
            # 顶部帮助按钮与弹窗
            import pathlib
            help_md_path = pathlib.Path("GUI_USAGE_CN.md")
            if help_md_path.exists():
                help_content = help_md_path.read_text(encoding="utf-8")
            else:
                help_content = "### 使用帮助\n文档缺失：GUI_USAGE_CN.md 未找到。"
            with gr.Row():
                gr.Markdown("# 工业工程学科引擎-构建专属智能体")
            # 直接使用可折叠说明，无需额外按钮；用户点击标题即可展开/收起
            help_box = gr.Accordion("📘 使用说明（点击展开）", open=False)
            with help_box:
                help_md = gr.Markdown(help_content)
            user_id_display = gr.Textbox(label="用户标识（自动生成）", interactive=False)
            # Removed directory selection UI entirely
            # gr.Markdown("每个会话中智能体会使用新创建的临时工作空间，关闭页面后会自动删除。")
            gr.Markdown("## 选择要启动或修改的智能体")
            agent_selector = gr.Dropdown(
                choices=["创建新智能体"] + list(config_manager.get_all_agent_metadata().keys()),
                label="可选智能体",
                value="创建新智能体"
            )
            launch_status = gr.Textbox(label="状态", interactive=False, lines=1)
            
            gr.Markdown("## 智能体配置信息")            
            with gr.Column(visible=False) as config_form:
                agent_name_input = gr.Textbox(label="智能体名称")
                agent_description_input = gr.Textbox(label="智能体描述", lines=3, placeholder="请输入该智能体的用途或简介…")
                prompt_box = gr.Textbox(lines=5, label="系统提示词", placeholder="明确该智能体的角色、目标、可调用工具、何时调用子智能体等…")
                # --- 分组工具选择区（按子目录折叠显示，支持组全选） ---
                def _compute_grouped_tools():
                    cls_map = discover_tools()
                    grouped: dict[str, list[tuple[str,str]]] = {}
                    for tname, cls in cls_map.items():
                        module = getattr(cls, '__module__', '')
                        parts = module.split('.')
                        # group = parts[-1] if len(parts) > 2 and parts[0] == 'src.hamlet.tools' else '其他'
                        group = parts[-1] if len(parts) > 2 and parts[0] == 'src' else '其他'
                        desc = getattr(cls, 'description', '') or ''
                        grouped.setdefault(group, []).append((tname, desc))
                    for g in grouped:
                        grouped[g].sort(key=lambda x: x[0])
                    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))

                grouped_tools = _compute_grouped_tools()
                master_components = {}
                tool_components: dict[str, list[gr.Checkbox]] = {}

                # Hidden unified selection state for persistence / saving
                all_tool_names = [t for grp in grouped_tools.values() for t,_ in grp]
                tool_checkboxes = gr.CheckboxGroup(choices=all_tool_names, label="选择工具 (内部)", visible=False)

                gr.Markdown("### 可选的工具分组（展开后可逐条勾选，左侧为工具勾选，右侧为简介）")
                for group, tool_list in grouped_tools.items():
                    with gr.Accordion(f"{group}（{len(tool_list)}）工具", open=False):
                        master_cb = gr.Checkbox(False, label="全选/取消本组", info="勾选后选中或清除本组所有工具")
                        per_tool_cbs = []
                        for t, d in tool_list:
                            with gr.Row():
                                cb = gr.Checkbox(False, label=t)
                                # description markdown truncated for readability
                                short = (d or '').strip() if isinstance(d, str) else ''
                                if len(short) > 300:
                                    short = short[:300] + '…'
                                gr.Markdown(short if short else '（无描述）')
                                per_tool_cbs.append(cb)
                        tool_components[group] = per_tool_cbs
                        master_components[group] = master_cb

                def _compute_selected_from_components(current_hidden: list[str], updates: dict[str, bool]):
                    base = set(current_hidden or [])
                    for name, val in updates.items():
                        if val:
                            base.add(name)
                        else:
                            base.discard(name)
                    return sorted(base)

                def toggle_group(master_value: bool, current_selected: list[str], group: str):
                    # Set all tool checkboxes in group to master_value
                    names = [t for t,_ in grouped_tools.get(group, [])]
                    if master_value:
                        new_selected = sorted(set(current_selected or []) | set(names))
                    else:
                        new_selected = [t for t in (current_selected or []) if t not in set(names)]
                    # Return hidden unified update plus each tool checkbox update
                    tool_updates = [gr.update(value=master_value) for _ in names]
                    return [gr.update(value=new_selected)] + tool_updates

                def toggle_single(tool_name: str, tool_value: bool, current_selected: list[str], group: str):
                    if tool_value:
                        new_selected = sorted(set(current_selected or []) | {tool_name})
                    else:
                        new_selected = [t for t in (current_selected or []) if t != tool_name]
                    # Determine master state
                    group_names = [t for t,_ in grouped_tools.get(group, [])]
                    master_state = all(n in new_selected for n in group_names) if group_names else False
                    return gr.update(value=new_selected), gr.update(value=master_state)

                def _sync_wrapper(selected_list: list[str]):
                    """Produce updates for every per-tool checkbox then every master checkbox (order must match wiring)."""
                    selected = set(selected_list or [])
                    updates = []
                    # Per-tool checkboxes (in group order)
                    for group, tool_list in grouped_tools.items():
                        for t,_ in tool_list:
                            updates.append(gr.update(value=(t in selected)))
                    # Master checkboxes
                    for group, tool_list in grouped_tools.items():
                        names = [t for t,_ in tool_list]
                        master_state = all(t in selected for t in names) and len(names) > 0
                        updates.append(gr.update(value=master_state))
                    return updates

                # Bind events for master checkboxes
                for group, master_cb in master_components.items():
                    # Provide hidden textbox carrying group name for arg passing
                    hidden_group = gr.Textbox(value=group, visible=False)
                    names = [t for t,_ in grouped_tools[group]]
                    # Outputs: hidden unified selection + each per-tool checkbox in this group
                    per_tool_cbs = tool_components[group]
                    master_cb.change(
                        toggle_group,
                        [master_cb, tool_checkboxes, hidden_group],
                        [tool_checkboxes] + per_tool_cbs
                    )
                    # Bind each tool checkbox
                    for cb, (tname, _) in zip(per_tool_cbs, grouped_tools[group]):
                        hidden_tool = gr.Textbox(value=tname, visible=False)
                        hidden_group2 = gr.Textbox(value=group, visible=False)
                        cb.change(
                            toggle_single,
                            [hidden_tool, cb, tool_checkboxes, hidden_group2],
                            [tool_checkboxes, master_cb]
                        )
                gr.Markdown("### 可选的子智能体（用户先前构建的其他智能体）")
                sub_agent_checkboxes = gr.CheckboxGroup(choices=list(config_manager.get_all_agent_metadata().keys()))
                with gr.Row():
                    save_button = gr.Button("保存配置")
                    close_button = gr.Button("取消")
                    delete_button = gr.Button("删除智能体", variant="stop")
            
            launch_button = gr.Button("启动智能体", variant="primary")

        # --- PAGE 2: CHAT INTERFACE ---
        with gr.Row(visible=False) as chat_page:
            # Left Sidebar: Controls
            with gr.Column(scale=3, min_width=300):
                agent_title_md = gr.Markdown("## 代码助手")
                agent_desc_md = gr.Markdown("（无描述）")
                # Removed显示实际服务器路径的文本框以避免泄露目录结构
                file_uploader = gr.File(label="上传文件", type="filepath")
                upload_status = gr.Textbox(visible=False, interactive=False)
                back_button = gr.Button("← 返回配置")

            # Middle Area: Chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    label="智能体对话",
                    avatar_images=(None, "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png"),
                    height="calc(100vh - 160px)", # Adjust height for the input box below
                    show_copy_button=True,
                    type="messages"
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        show_label=False,
                        placeholder="请输入你的问题…",
                        lines=4,
                        scale=8,
                    )
                    submit_button = gr.Button("提交", variant="primary", scale=1, min_width=100)
            # Right Sidebar: Files and Preview (adapted)
            with gr.Column(scale=3, min_width=300, elem_classes="vscode-pane"):
                gr.Markdown("### 智能体工作目录")
                gr.Markdown("每次对话都会使用一个新的临时工作目录，智能体可以读写该目录下的文件。请注意，关闭或刷新页面后目录会被自动删除。")
                refresh_btn = gr.Button("🔄 刷新", size="sm")
                file_explorer = gr.FileExplorer(root_dir=".", file_count="multiple", interactive=True, height=220)
                # Removed debug markdown and fallback dropdown
                # Fullscreen overlay container
                gr.HTML("""
<div id='preview-overlay' style='display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9999;align-items:center;justify-content:center;'>
    <div id='overlay-content' style='background:#fff;width:95vw;height:95vh;position:relative;overflow:hidden;border-radius:8px;'>
        <span id='overlay-close' style='position:absolute;top:8px;right:16px;font-size:32px;color:#333;cursor:pointer'>&times;</span>
    </div>
</div>
""")
                gr.Markdown("### 预览 <span id='preview_fs' style='cursor:pointer;font-size:0.9em;'>🗖</span>")
                pdf_preview = PDF(visible=False, interactive=False, elem_classes="preview-box")
                img_preview = gr.Image(interactive=False, visible=False, elem_classes="preview-box")
                table_preview = gr.DataFrame(interactive=False, visible=False, elem_classes="preview-box")
                text_preview = gr.Code(interactive=False, visible=True, lines=20, elem_classes="preview-box")
                # Button to trigger on-demand zipping/downloading
                download_selected_btn = gr.Button("⬇️ 打包/下载所选", variant="secondary")
                download_file = gr.File(label="下载结果", visible=False)
                # JS for fullscreen toggle
                demo.load(None, None, None, js="""
() => {
    const btn = document.getElementById('preview_fs');
    const overlay = document.getElementById('preview-overlay');
    const closeBtn = document.getElementById('overlay-close');
    const content = document.getElementById('overlay-content');
    if (!btn || btn.dataset.bound) return; btn.dataset.bound=1;
    function activePreview(){
        return Array.from(document.querySelectorAll('.preview-box')).find(el=>el.offsetParent!==null);
    }
    btn.addEventListener('click', ()=>{
        const box = activePreview(); if(!box) return;
        const placeholder=document.createElement('div'); placeholder.style.display='none';
        box.parentNode.insertBefore(placeholder, box);
        content.innerHTML=''; content.appendChild(box);
        overlay.style.display='flex';
        function close(){ if(placeholder.parentNode) placeholder.parentNode.replaceChild(box, placeholder); overlay.style.display='none'; }
        closeBtn.onclick=close; overlay.onclick=(e)=>{ if(e.target===overlay) close(); };
    });
}
""")
        # --- 事件绑定 ---
        # First: when selecting an agent, load its config (including hidden unified tool list)
        # Then: synchronize visible grouped checkboxes using the updated hidden list.
        # We build _sync_outputs first so we can chain correctly (patching prior placeholder approach)
        # 新结构：_sync_wrapper 返回顺序 = 全部工具复选框(逐个) + 全部组 master 复选框
        _sync_outputs = []
        per_tool_flat = []
        for g, tool_list in grouped_tools.items():
            for cb in tool_components[g]:
                per_tool_flat.append(cb)
        _sync_outputs.extend(per_tool_flat)
        for g in grouped_tools.keys():
            _sync_outputs.append(master_components[g])

        agent_selector.change(
            on_agent_select,
            [agent_selector],
            [config_state, agent_name_input, agent_description_input, tool_checkboxes, sub_agent_checkboxes, prompt_box, config_form]
        ).then(
            _sync_wrapper,
            [tool_checkboxes],
            _sync_outputs
        )
        # (Removed placeholder sync rebuilding block — outputs are already wired in chain above.)

        save_button.click(
            save_config,
            [agent_name_input, agent_description_input, tool_checkboxes, sub_agent_checkboxes, prompt_box],
            [agent_selector, sub_agent_checkboxes, launch_status]
        )
        delete_button.click(
            delete_agent,
            [agent_name_input],
            [agent_selector, sub_agent_checkboxes, launch_status]
        ).then(
            on_agent_select,
            [agent_selector],
            [config_state, agent_name_input, agent_description_input, tool_checkboxes, sub_agent_checkboxes, prompt_box, config_form]
        )
        close_button.click(lambda: gr.update(visible=False), outputs=[config_form])
        launch_button.click(
            launch_agent, 
            [config_state, agent_description_input, tool_checkboxes, prompt_box, working_dir_state], 
            [config_page, chat_page, agent_state, launch_status, agent_title_md, agent_desc_md, file_explorer]
        )
        file_explorer.change(
            show_preview,
            [file_explorer, working_dir_state],
            [pdf_preview, img_preview, table_preview, text_preview, download_file]
        )
        download_selected_btn.click(
            prepare_download,
            [file_explorer, working_dir_state],
            [download_file, text_preview]
        )
        file_uploader.change(
            handle_upload,
            [file_uploader, file_uploads_log, working_dir_state],
            [upload_status, file_uploads_log, file_explorer]
        ).then(
            force_refresh_files,
            None,
            [file_explorer]
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
            None,
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
            None,
            [file_explorer]
        )

        back_button.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[config_page, chat_page])

        # 手动刷新按钮
        refresh_btn.click(
            force_refresh_files,
            None,
            [file_explorer]
        )

        # 定时刷新
        refresh_timer.tick(
            force_refresh_files,  # function accepts optional param, will be called without args
            None,
            [file_explorer]
        )

        # 初始化：为每个客户端创建会话临时工作目录
        def _on_app_load(request: gr.Request):  # type: ignore[name-defined]
            user_id = _ensure_cookie_user_id(request)
            session_id = str(uuid.uuid4())
            temp_dir = _create_temp_working_dir(session_id)
            nonlocal current_watching_dir
            current_watching_dir = temp_dir
            file_watcher.start_watching(temp_dir, lambda: print(f"[SESSION {session_id}] FS change."))
            user_agents = sorted(list(config_manager.get_all_agent_metadata().keys()))
            # If the user already has agents, pre-select the first one; otherwise default to sentinel and creation form will appear.
            initial_value = user_agents[0] if user_agents else "创建新智能体"
            selector_update = gr.update(choices=user_agents + ["创建新智能体"], value=initial_value)
            sub_agent_update = gr.update(choices=user_agents, value=[])
            return (
                user_id,
                session_id,
                temp_dir,  # working_dir_state
                gr.update(root_dir=temp_dir),
                gr.update(value=user_id),
                selector_update,
                sub_agent_update
            )

        demo.load(
            _on_app_load,
            inputs=None,
            outputs=[user_id_state, session_dir_state, working_dir_state, file_explorer, user_id_display, agent_selector, sub_agent_checkboxes]
        )
        # Trigger population of form fields (and make config_form visible) based on the initial selector value.
        demo.load(
            on_agent_select,
            inputs=[agent_selector],
            outputs=[config_state, agent_name_input, agent_description_input, tool_checkboxes, sub_agent_checkboxes, prompt_box, config_form]
        )

        def _on_app_unload():
            for d in list(session_temp_dirs.values()):
                _cleanup_temp_dir(d)
        demo.unload(_on_app_unload)

        # Build list button manual trigger
        # Removed fallback list events
        
        # Add cleanup when demo closes (file watcher will auto-cleanup via __del__)
        
    return demo

if __name__ == "__main__":
    app_instance = app()
    print("App created successfully, attempting to launch...")
    app_instance.queue()  # Enable queue for better handling
    app_instance.launch(server_name='0.0.0.0', server_port=4000, share=False)
    # app_instance.launch(server_port=7280, share=False)