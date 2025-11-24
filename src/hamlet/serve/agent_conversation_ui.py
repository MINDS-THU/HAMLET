#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import re
import shutil
import tempfile
import zipfile
from typing import Optional

from hamlet.core.agent_types import AgentAudio, AgentImage, AgentText
from hamlet.core.agents import MultiStepAgent, PlanningStep
from hamlet.core.memory import ActionStep, FinalAnswerStep, MemoryStep
from hamlet.core.utils import _is_package_available
from hamlet.core.models import ChatMessageStreamDelta  # NEW


def get_step_footnote_content(step_log: MemoryStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
        token_str = f" | Input tokens:{step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration"):
        step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
        step_footnote += step_duration
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content


def pull_messages_from_step(step_log: MemoryStep, skip_model_outputs: bool = False):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        # Output the step number  show only once if we aren’t already streaming
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Step"
        if not skip_model_outputs:
            yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if (not skip_model_outputs) and hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "done",
                },
            )
            yield parent_message_tool

        # Display execution logs if they exist
        if hasattr(step_log, "observations") and (
            step_log.observations is not None and step_log.observations.strip()
        ):  # Only yield execution logs if there's actual content
            log_content = step_log.observations.strip()
            if log_content:
                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                yield gr.ChatMessage(
                    role="assistant",
                    content=f"```bash\n{log_content}\n",
                    metadata={"title": "📝 Execution Logs", "status": "done"},
                )

        # Display any errors
        if hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Error", "status": "done"},
            )

        # Update parent message metadata to done status without yielding a new message
        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                path_image = AgentImage(image).to_string()
                yield gr.ChatMessage(
                    role="assistant",
                    content={"path": path_image, "mime_type": f"image/{path_image.split('.')[-1]}"},
                    metadata={"title": "🖼️ Output Image", "status": "done"},
                )

        # Handle standalone errors but not from tool calls
        if hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        yield gr.ChatMessage(role="assistant", content=get_step_footnote_content(step_log, step_number))
        yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})

    elif isinstance(step_log, PlanningStep):
        if not skip_model_outputs:               # ★ guard duplication
            yield gr.ChatMessage(role="assistant", content="**Planning step**")
            yield gr.ChatMessage(role="assistant", content=step_log.plan)
            yield gr.ChatMessage(role="assistant", content=get_step_footnote_content(step_log, "Planning step"))
            yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = getattr(step_log, "output", None)
        if isinstance(final_answer, AgentText):
            yield gr.ChatMessage(
                role="assistant",
                content=f"**Final answer:**\n{final_answer.to_string()}\n",
            )
        elif isinstance(final_answer, AgentImage):
            yield gr.ChatMessage(
                role="assistant",
                content={"path": final_answer.to_string(), "mime_type": "image/png"},
            )
        elif isinstance(final_answer, AgentAudio):
            yield gr.ChatMessage(
                role="assistant",
                content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            )
        else:
            yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")

    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")

def stream_to_gradio(
    agent,
    task: str,
    task_images: list | None = None,
    reset_agent_memory: bool = False,
    additional_args: dict | None = None,
):
    """Runs an agent with the given task and streams the messages as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError("Install with: pip install 'smolagents[gradio]'")

    import gradio as gr

    intermediate_text = ""
    for step_log in agent.run(
        task, images=task_images, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        # copy token counts onto the step, if they exist
        if getattr(agent.model, "last_input_token_count", None) is not None and isinstance(
            step_log, (ActionStep, PlanningStep)
        ):
            step_log.input_token_count = agent.model.last_input_token_count
            step_log.output_token_count = agent.model.last_output_token_count

        # ───────────────────────────────── Memory steps ─────────────────────────
        if isinstance(step_log, MemoryStep):
            intermediate_text = ""  # reset buffer
            for msg in pull_messages_from_step(
                step_log,
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield msg

        # ───────────────────────────────── Streaming deltas ─────────────────────
        elif isinstance(step_log, ChatMessageStreamDelta):
            intermediate_text += step_log.content or ""
            yield intermediate_text



class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(
        self,
        agent: MultiStepAgent,
        file_upload_folder: str | None = None,
        readme_md_path: str | None = None,
    ):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder: str
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        self._set_file_upload_folder(file_upload_folder)
        self._readme_path = readme_md_path
        self._readme_markdown = self._load_agent_readme(readme_md_path)

    def _set_file_upload_folder(self, folder: Optional[os.PathLike[str] | str]) -> None:
        """Ensure the upload folder exists, defaulting to a temporary directory."""
        if folder is None:
            folder_path = tempfile.mkdtemp(prefix="hamlet_gradio_upload_")
        else:
            folder_path = os.fspath(folder)
            os.makedirs(folder_path, exist_ok=True)
        self.file_upload_folder = folder_path

    def _load_agent_readme(self, readme_path: Optional[str], asset_base_url: Optional[str] = None) -> Optional[str]:
        """Read optional markdown file that describes the agent, fixing relative assets if needed."""
        if not readme_path:
            return None
        import urllib.parse

        parsed = urllib.parse.urlparse(readme_path)
        if parsed.scheme in {"http", "https"}:
            # handle GitHub "blob" URLs by converting to raw content
            if parsed.netloc in {"github.com", "www.github.com"}:
                parts = parsed.path.strip("/").split("/")
                if len(parts) >= 5 and parts[2] == "blob":
                    owner, repo, _blob, branch, *rest = parts
                    asset_dir = "/".join(rest[:-1])
                    if asset_dir:
                        asset_dir += "/"
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{'/'.join(rest)}"
                    base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{asset_dir}"
                    return self._load_agent_readme(raw_url, base)

            if parsed.netloc == "raw.githubusercontent.com":
                parts = parsed.path.strip("/").split("/")
                if len(parts) >= 4:
                    asset_dir = "/".join(parts[:-1]) + "/"
                    asset_base_url = f"https://raw.githubusercontent.com/{asset_dir}"

            try:
                import urllib.request

                request = urllib.request.Request(readme_path, headers={"User-Agent": "HAMLET-Agent/1.0"})
                with urllib.request.urlopen(request) as response:
                    raw_bytes = response.read()
                    detected_charset = response.headers.get_content_charset()
                    try:
                        content = raw_bytes.decode(detected_charset or "utf-8", errors="strict")
                    except (LookupError, UnicodeDecodeError):
                        content = raw_bytes.decode("utf-8", errors="replace")
                    if asset_base_url:
                        content = self._rewrite_relative_asset_paths(content, asset_base_url)
                    return content
            except Exception as exc:  # broad to keep UI resilient
                print(f"Warning: could not fetch agent readme '{readme_path}': {exc}")
                return None

        abs_path = os.path.abspath(readme_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as readme_file:
                content = readme_file.read()
                if asset_base_url:
                    content = self._rewrite_relative_asset_paths(content, asset_base_url)
                return content
        except OSError as exc:
            print(f"Warning: could not load agent readme '{abs_path}': {exc}")
            return None

    def _rewrite_relative_asset_paths(self, markdown_text: str, base_url: str) -> str:
        """Convert relative image paths to absolute URLs so Gradio can render them."""
        import re
        import urllib.parse

        if not base_url.endswith("/"):
            base_url = base_url + "/"

        def _absolutize(url: str) -> str:
            url = url.strip()
            if not url or url.startswith(("http://", "https://", "data:")) or url.startswith("#"):
                return url
            return urllib.parse.urljoin(base_url, url)

        def _replace_markdown_image(match: re.Match) -> str:
            alt = match.group(1)
            target = match.group(2).strip()
            title = match.group(3)
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            abs_url = _absolutize(target)
            title_part = f' "{title}"' if title is not None else ""
            return f"![{alt}]({abs_url}{title_part})"

        def _replace_html_image(match: re.Match) -> str:
            quote = match.group(1)
            url = match.group(2)
            rest = match.group(3)
            abs_url = _absolutize(url)
            return f"<img src={quote}{abs_url}{quote}{rest}>"

        md_image_pattern = re.compile(r"!\[([^\]]*)\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+\"([^\"]*)\")?\s*\)")
        html_image_pattern = re.compile(r"<img\s+[^>]*src=([\"\'])\s*([^\"\'>]+)\1([^>]*)>")

        markdown_text = md_image_pattern.sub(_replace_markdown_image, markdown_text)
        markdown_text = html_image_pattern.sub(_replace_html_image, markdown_text)
        return markdown_text

    def interact_with_agent(self, prompt, messages, session_state):
        import gradio as gr

        # Get the agent type from the template agent
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            for msg in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                if isinstance(msg, gr.ChatMessage):
                    messages.append(msg)  # finished step
                elif isinstance(msg, str):                      # ← live delta
                    if messages and messages[-1].metadata.get("status") == "pending":
                        messages[-1].content = msg              # update the same bubble
                    else:
                        messages.append(
                            gr.ChatMessage(role="assistant", content=msg, metadata={"status": "pending"})
                        )
                yield messages

            yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            messages.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield messages

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".py", ".md", ".csv", ".json", ".xlsx", ".html", ".xls", ".wav", ".mp3"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        import gradio as gr
        # Show only relative paths under self.file_upload_folder
        if self.file_upload_folder:
            rel_paths = [os.path.relpath(f, self.file_upload_folder) for f in file_uploads_log]
        else:
            rel_paths = file_uploads_log

        return (
            text_input
            + (
                f"\nFile uploaded: {rel_paths}"
                if len(rel_paths) > 0
                else ""
            ),
            "",
            gr.Button(interactive=False),
        )

    def launch(self, share: bool = True, file_upload_folder: Optional[os.PathLike[str] | str] = None, **kwargs):
        if file_upload_folder is not None:
            self._set_file_upload_folder(file_upload_folder)
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        custom_css = r"""
        /* --------------------------------------------------------------
        1. Global tweaks – remove Gradio’s 80 rem cap
        ----------------------------------------------------------------*/
        html, body{
            width:100%;
            min-height:100vh;
            height:100%;
            margin:0;
            padding:0;
        }

        #hamlet-root{
            min-height:100vh;
            display:flex;
            flex-direction:column;
        }
        #hamlet-root > .gr-block{
            flex:1 1 auto;
            display:flex;
            flex-direction:column;
            max-width:100%!important;
            width:100%!important;
        }

        /* make the central Row stretch full width */
        .gr-block.gr-row{width:100%!important;}

        /* --------------------------------------------------------------
        2. Layout – chat column and VS‑Code‑like right column
        ----------------------------------------------------------------*/

        .vscode-pane{
            border-left:1px solid #e0e0e0;
            display:flex; flex-direction:column;
            height:100%; overflow:hidden;
        }
                #hamlet-main{
                    display:flex;
                    flex:1 1 auto;
                    min-height:0;
                    align-items:stretch;
                    flex-wrap:wrap;
                }
                #hamlet-main > .gr-column{
                    height:100%;
                    display:flex;
                    flex-direction:column;
                }
        .vscode-header{background:#f6f6f6;font-weight:600;padding:4px 8px;}

        .vscode-pane .gr-file-explorer{flex:0 0 25vh;overflow:auto;}

        .vscode-pane .gr-code,
        .vscode-pane .gr-image,
        .vscode-pane .gr-dataframe,
        .vscode-pane .gr-pdf{
            flex:1 1 auto;min-height:0;overflow:auto;
            width:100%!important;box-sizing:border-box;
        }

        .chat-pane{
            display:flex;
            flex-direction:column;
            height:100%;
            gap:.5rem;
        }
        .tsinghua-banner-wrapper{
            flex:1 0 100%;
            width:100%;
            box-sizing:border-box;
            margin-bottom:.75rem;
        }
        .tsinghua-banner{
            background:#602460;
            border-radius:8px;
            padding:.75rem;
            display:flex;
            align-items:center;
            justify-content:center;
            width:100%;
            position:relative;
        }
        .tsinghua-banner-logo{
            position:absolute;
            left:1.25rem;
            display:flex;
            align-items:center;
        }
        .tsinghua-banner-logo img{
            max-height:56px;
            width:auto;
            display:block;
        }
        .tsinghua-banner-title{
            font-size:1.4rem;
            font-weight:600;
            color:#ffffff;
            letter-spacing:.08rem;
            text-align:center;
            width:100%;
        }
        .chat-pane .gr-chatbot{
            flex:1 1 auto;
            min-height:0;
        }
        .chat-input-row{
            flex:0 0 auto;
            align-items:flex-end;
            gap:.5rem;
        }

        @media (min-width:1024px){
        .gr-block.gr-row{display:flex;gap:1rem;}
        .gr-block.gr-column{flex-grow:1;min-width:300px;}
        }
        .gr-column:nth-child(1){flex:3;min-width:450px;}   /* chat */
        .gr-column:nth-child(2){flex:2;min-width:350px;}   /* files+preview */

        /* --------------------------------------------------------------
        3. Preview pane inside the right column
        ----------------------------------------------------------------*/
        .preview-box{
            max-height:60vh;overflow:auto;flex:1 1 auto;min-height:0;
        }
        .preview-box .gr-pdf,
        .preview-box .gr-pdf object,
        .preview-box .gr-pdf iframe{
            width:100%!important;height:100%!important;
        }

        .preview-header-row{
            align-items:center;
            justify-content:space-between;
            margin-top:.25rem;
            margin-bottom:.25rem;
            gap:.5rem;
        }
        .preview-header-title{flex:1; margin:0;}
        .preview-toolbar{display:flex;gap:.35rem;align-items:center;justify-content:flex-end;}
        .preview-toolbar button{
            min-width:34px;
            height:32px;
            padding:0;
            font-size:1rem;
            border-radius:4px;
        }
        .preview-toolbar button:disabled{opacity:0.45;cursor:not-allowed;}
        .preview-toolbar button.copied{background:#dff5ec;border-color:#8fd3b7;}

        .files-header-row{
            align-items:center;
            justify-content:space-between;
            margin-bottom:1.00rem;
            gap:.5rem;
        }
        .files-header-title{flex:1; margin:0;}
        .files-toolbar{display:flex;gap:.35rem;align-items:center;justify-content:flex-end;}
        .files-toolbar button,
        .files-toolbar label{
            min-width:34px;
            height:32px;
            padding:0 .6rem;
            font-size:.9rem;
            border-radius:4px;
            display:flex;
            align-items:center;
            justify-content:center;
            cursor:pointer;
        }
        .files-toolbar button:disabled,
        .files-toolbar label[aria-disabled="true"]{
            opacity:0.45;
            cursor:not-allowed;
        }

        .preview-wrapper{position:relative;}

        .vscode-pane{
            gap:0 !important;
            row-gap:0 !important;      /* Safari */
        }
        .vscode-pane > .gr-block{
            margin-top:0 !important;
        }

        /* --------------------------------------------------------------
        4. FULL‑SCREEN overlay
        ----------------------------------------------------------------*/
        .preview‑overlay{
            position:fixed;inset:0;background:rgba(0,0,0,.75);
            display:none;align-items:center;justify-content:center;
            z-index:9999;
        }

        /* white panel that holds the live preview */
        .overlay‑content{
            background:#fff;
            padding:0;
            border-radius:8px;
            width:95vw;height:95vh;        /* almost the whole viewport */
            overflow:hidden;               /* the PDF itself will scroll */
            position:relative;             /* keeps the “×” inside      */
        }

        /* single close button – moved 1 rem inwards so it doesn’t overlap
        with the PDF viewer’s own “×” */
        .overlay‑close{
            position:absolute;top:1rem;right:1rem;     /* <‑ no overlap now */
            color:#fff;font-size:2.4rem;cursor:pointer;user-select:none;
        }

        /* --------------------------------------------------------------
        5. Stretch whatever we moved into the overlay
        ----------------------------------------------------------------*/
        /* ——— make the moved preview fill the whole panel ——— */
        .overlay‑content .preview-box,
        .overlay‑content .gr-code,
        .overlay‑content .gr-image,
        .overlay‑content .gr-dataframe,
        .overlay‑content .gr-pdf{
            width:100%!important;height:100%!important;
            max-width:none!important;max-height:none!important;
            overflow:auto;
        }

        .overlay‑content canvas,
        .overlay‑content .page,
        .overlay‑content .pdf-page{
            display:block;
            margin:0 auto;                 /* centre horizontally          */
            max-width:100%;                /* don’t overflow the panel     */
            height:auto;                   /* keep aspect ratio            */
        }
        /* if an <embed>/<iframe> is used instead, stretch that too */
        .overlay‑content .gr-pdf object,
        .overlay‑content .gr-pdf iframe,
        .overlay‑content embed[type="application/pdf"]{
            width:100%!important;height:100%!important;
        }

        /* centre the navigation bar that gradio‑pdf injects */
        .overlay‑content .swiper-pagination,
        .overlay‑content .swiper-button-prev,
        .overlay‑content .swiper-button-next{
            position:relative!important;
        }
        }

        /* --------------------------------------------------------------
        Header look‑&‑feel  (add just after the existing .vscode-header
        declaration so it overrides browser defaults)
        ----------------------------------------------------------------*/
        .vscode-header{                    /* unify both headers         */
            font-size: .95rem;             /* same text size             */
            line-height: 1.35;             /* compact vertical rhythm    */
            margin: 0   !important;        /* kill <h3>’s huge margins   */
        }

        .vscode-header h1,
        .vscode-header h2,
        .vscode-header h3{                 /* when one is produced by
            margin:0; font:inherit; }      /* gr.Markdown(“### …”)       */

        /* every direct .gr-block child in the VS‑Code column      */
        .vscode-pane > .gr-block{
            margin-top:.25rem !important;      /* same 4 px gap for all */
        }

        /* keep the very first child (the “Files” header) flush    */
        .vscode-pane > .gr-block:first-child{
            margin-top:0 !important;
        }

        /* the invisible overlay’s wrapper must not push anything  */
        .vscode-pane > .overlay-wrapper{
            margin-top:0 !important;           /* kill the gap         */
            height:0 !important;               /* removes extra space  */
        }

        /* --------------------------------------------------------------
        6. Brand credit badge
        ----------------------------------------------------------------*/
        .minds-credit{
            margin-top:.6rem;
            display:flex;
            justify-content:flex-end;
            --credit-avatar-size:28px;
        }
        .minds-credit .credit-pill{
            background:rgba(255,255,255,0.95);
            border:1px solid #dcdcdc;
            border-radius:12px;
            padding:.45rem .75rem;
            display:flex;
            align-items:center;
            gap:.65rem;
            box-shadow:0 4px 12px rgba(0,0,0,0.08);
            pointer-events:auto;
            max-width:320px;
        }
        .minds-credit .credit-pill img{
            width:var(--credit-avatar-size)!important;
            height:var(--credit-avatar-size)!important;
            max-width:none!important;
            max-height:none!important;
            border-radius:10px;
            border:1px solid #e5e5e5;
            object-fit:cover;
            flex-shrink:0;
            display:block;
        }
        .minds-credit .credit-text{
            display:block;
            font-size:.82rem;
            line-height:1.15;
            text-align:left;
        }
        .minds-credit .credit-line{
            display:block;
            margin-bottom:.2rem;
        }
        .minds-credit .credit-line:last-child{margin-bottom:0;}
        .minds-credit .credit-text a{
            font-size:.79rem;
            color:#1769aa;
            text-decoration:none;
        }
        .minds-credit .credit-text a:hover{
            text-decoration:underline;
        }

        """

        LIGHTBOX_HTML = """
        <div id="preview-overlay" class="preview‑overlay">
        <span class="overlay‑close"
                onclick="event.stopPropagation();"
                >&times;</span>
        <div id="overlay‑content" class="overlay‑content"></div>
        </div>
        """

        import os, time
        import gradio as gr
        import pandas as pd
        from gradio_pdf import PDF          #  <- NEW (pip install gradio-pdf)

        folder = os.path.abspath(self.file_upload_folder)

        # ---------- helpers --------------------------------------------------
        def list_folder():
            """Return sorted list of every file/dir (relative paths) inside folder."""
            paths = []
            for root, dirs, files in os.walk(folder):
                for n in dirs + files:
                    paths.append(os.path.relpath(os.path.join(root, n), folder))
            return sorted(paths)

        # def show_preview(selection):
        #     """
        #     `selection` is a list of checked items.
        #     If nothing or a directory is selected, return a helpful message.
        #     For files, return up to 20 kB of text so huge files don’t freeze the UI.
        #     """
        #     if not selection:
        #         return "⬅️  Click a file to preview"

        #     # if multi‑select, look at the first item
        #     rel_path = selection[0] if isinstance(selection, list) else selection
        #     abs_path = os.path.join(folder, rel_path)

        #     if os.path.isdir(abs_path):
        #         return "📁 This is a directory."
        #     try:
        #         with open(abs_path, "r", encoding="utf‑8", errors="ignore") as f:
        #             return f.read(20_000)   # read first 20 kB
        #     except Exception as e:
        #         return f"⚠️  Cannot display file: {e}"


        def preview_message(text: str):
            hidden = gr.update(visible=False)
            download_reset = gr.update(value=None, visible=True, interactive=False)
            button_disabled = gr.update(interactive=False)
            return (
                hidden,
                hidden,
                hidden,
                gr.update(value=text, visible=True),
                download_reset,
                button_disabled,
                button_disabled,
                button_disabled,
                "",
            )

        def _pick_valid_selection(selection):
            """Return the most recently clicked existing path."""
            if not selection:
                return None
            if isinstance(selection, list):
                for candidate in reversed(selection):
                    if not candidate:
                        continue
                    candidate_path = os.path.join(folder, candidate)
                    if os.path.exists(candidate_path):
                        return candidate
                # fallback to last entry even if missing so we can show an error
                return selection[-1]
            return selection

        def show_preview(selection):
            """Return component updates so the right viewer shows the selected file."""
            hidden = gr.update(visible=False)
            copy_disabled = gr.update(interactive=False)

            rel_path = _pick_valid_selection(selection)
            if not rel_path:
                return preview_message("⬅️ Click a file to preview")

            abs_path = os.path.join(folder, rel_path)
            if not os.path.exists(abs_path):
                return preview_message("⚠️ File not found. Refresh and try again.")

            if os.path.isdir(abs_path):
                return preview_message("📁 Directory (no preview)")

            ext = os.path.splitext(abs_path)[1].lower()
            download_update = gr.update(value=abs_path, visible=True, interactive=True)
            delete_enabled = gr.update(interactive=True)
            fullscreen_enabled = gr.update(interactive=True)

            # ---------- IMAGES ----------
            if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                return (
                    hidden,
                    gr.update(value=abs_path, visible=True),
                    hidden,
                    hidden,
                    download_update,
                    copy_disabled,
                    delete_enabled,
                    fullscreen_enabled,
                    abs_path,
                )

            # ---------- PDF -------------
            if ext == ".pdf":
                return (
                    gr.update(value=abs_path, visible=True),
                    hidden,
                    hidden,
                    hidden,
                    download_update,
                    copy_disabled,
                    delete_enabled,
                    fullscreen_enabled,
                    abs_path,
                )

            # ---------- SPREADSHEETS ----
            if ext in {".csv", ".tsv", ".xlsx"}:
                try:
                    df = (pd.read_csv if ext != ".xlsx" else pd.read_excel)(abs_path)
                    return (
                        hidden,
                        hidden,
                        gr.update(value=df, visible=True),
                        hidden,
                        download_update,
                        copy_disabled,
                        delete_enabled,
                        fullscreen_enabled,
                        abs_path,
                    )
                except Exception as e:
                    return preview_message(f"⚠️ Cannot read file: {e}")

            # ---------- TEXT / fallback -
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(20_000)
            except Exception as e:
                txt = f"⚠️ Cannot display file: {e}"
                copy_state = copy_disabled
            else:
                copy_state = gr.update(interactive=True)

            return (
                hidden,
                hidden,
                hidden,
                gr.update(value=txt, visible=True),
                download_update,
                copy_state,
                delete_enabled,
                fullscreen_enabled,
                abs_path,
            )

        def delete_selected_file(current_file: str | None):
            if not current_file:
                return preview_message("⚠️ No file selected to delete.")

            abs_path = os.path.abspath(current_file)
            try:
                common = os.path.commonpath([folder, abs_path])
            except ValueError:
                return preview_message("⚠️ Invalid file path.")

            if common != folder:
                return preview_message("⚠️ Cannot delete files outside the workspace.")
            if not os.path.exists(abs_path):
                return preview_message("⚠️ File already removed.")
            if os.path.isdir(abs_path):
                return preview_message("⚠️ Deleting directories is not supported.")

            try:
                os.remove(abs_path)
                message = f"🗑️ Deleted {os.path.relpath(abs_path, folder)}"
            except Exception as e:
                message = f"⚠️ Delete failed: {e}"

            return preview_message(message)


        def zip_workspace():
            tmp_dir = tempfile.mkdtemp(prefix="hamlet_zip_")
            zip_path = os.path.join(tmp_dir, "workspace.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for root, _dirs, files in os.walk(folder):
                    for file in files:
                        abs_file = os.path.join(root, file)
                        rel_name = os.path.relpath(abs_file, folder)
                        archive.write(abs_file, rel_name)
            return zip_path


        def reset_workspace(current_log: list[str]):
            message = "The working directory has been reset."
            try:
                for entry in os.listdir(folder):
                    abs_entry = os.path.join(folder, entry)
                    if os.path.isdir(abs_entry):
                        shutil.rmtree(abs_entry)
                    else:
                        os.remove(abs_entry)
            except Exception as e:
                message = f"Failed to reset the working directory: {e}"
                new_log = current_log
            else:
                new_log = []

            preview_outputs = preview_message(message)
            status_update = gr.update(value=message, visible=True)
            return (*preview_outputs, new_log, status_update)



        def force_refresh():
            """
            Force‑refresh the FileExplorer by tweaking ignore_glob
            with a random pattern that matches nothing.
            """
            dummy_glob = f"__refresh_{int(time.time()*1000)}__"
            return gr.update(ignore_glob=dummy_glob)

        # ---------- UI -------------------------------------------------------
        with gr.Blocks(theme="ocean", fill_height=True, css=custom_css, elem_id="hamlet-root") as demo:
            session_state    = gr.State({})
            stored_messages  = gr.State([])
            file_uploads_log = gr.State([])
            selected_file    = gr.State("")
            upload_status_box = None

            # ----- sidebar ---------------------------------------------------
            with gr.Sidebar():
                gr.Markdown(f"# {self.name.title()}")
                if self.description:
                    gr.Markdown(f"**Agent:** {self.description}")
                if self._readme_markdown:
                    gr.Markdown(self._readme_markdown)

            # ----- main area -------------------------------------------------
            with gr.Row(scale=12, elem_classes="main-layout", elem_id="hamlet-main"):
                gr.HTML(
                    """
                    <div class="tsinghua-banner">
                        <div class="tsinghua-banner-logo">
                            <img src="https://www.ie.tsinghua.edu.cn/images/logo.png" alt="Tsinghua University IE Department logo" />
                        </div>
                        <div class="tsinghua-banner-title">学科知识引擎</div>
                    </div>
                    """,
                    visible=True,
                    elem_classes="tsinghua-banner-wrapper",
                )

                # ---- (A) Chat column ---------------------------------------
                with gr.Column(scale=7, min_width=500, elem_classes="chat-pane"):
                    chatbot = gr.Chatbot(
                        label="Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                        ),
                        resizeable=True,
                    )

                    with gr.Row(elem_classes="chat-input-row", equal_height=True):
                        text_input = gr.Textbox(
                            lines=1,
                            placeholder="Type your prompt…",
                            show_label=False,
                            scale=4,
                        )
                        submit_btn = gr.Button(
                            "提交",
                            variant="primary",
                            scale=1,
                        )

                # ---- (B) Explorer + preview column -------------------------
                with gr.Column(scale=5, min_width=350, elem_classes="vscode-pane"):
                    
                    # 1️⃣ FILE‑EXPLORER + toolbar
                    with gr.Row(elem_classes="files-header-row"):
                        gr.HTML("<div class='vscode-header files-header-title'>工作区</div>")
                        with gr.Row(elem_classes="files-toolbar"):
                            upload_button = gr.UploadButton(
                                "📤",
                                file_types=["file"],
                                file_count="single",
                                size="sm",
                                elem_id="files_upload",
                                variant="secondary",
                            )
                            zip_download_button = gr.DownloadButton(
                                label="🗜",
                                size="sm",
                                variant="secondary",
                                elem_id="files_zip",
                            )
                            reset_button = gr.Button(
                                "♻",
                                size="sm",
                                variant="secondary",
                                elem_id="files_reset",
                            )

                    upload_status_box = gr.Textbox(visible=False, interactive=False)
                    file_explorer = gr.FileExplorer(
                        root_dir=folder, file_count="single", interactive=True, value=None,
                        height="30vh",
                    )

                    with gr.Row(elem_classes="preview-header-row"):
                        gr.HTML("<div class='vscode-header preview-header-title'>预览</div>")
                        with gr.Row(elem_classes="preview-toolbar"):
                            copy_button = gr.Button(
                                "⧉",
                                elem_id="preview_copy",
                                interactive=False,
                                size="sm",
                                variant="secondary",
                            )
                            preview_download = gr.DownloadButton(
                                label="⬇",
                                visible=True,
                                interactive=False,
                                elem_id="preview_download",
                                size="sm",
                                variant="secondary",
                            )
                            fullscreen_button = gr.Button(
                                "🗖",
                                elem_id="preview_fs",
                                interactive=False,
                                size="sm",
                                variant="secondary",
                            )
                            delete_button = gr.Button(
                                "🗑",
                                elem_id="preview_delete",
                                interactive=False,
                                size="sm",
                                variant="secondary",
                            )

                    # put ONE overlay somewhere in the page (fixed‑position → location irrelevant)
                    gr.HTML(LIGHTBOX_HTML, visible=True, elem_classes="overlay-wrapper")

                    # 3️⃣ PREVIEW widgets
                    with gr.Group(elem_classes="preview-wrapper"):
                        pdf_preview   = PDF(
                            visible=False,
                            interactive=False,
                            elem_classes="preview-box"
                        )
                        img_preview   = gr.Image(interactive=False, visible=False, elem_classes="preview-box")
                        table_preview = gr.Dataframe(interactive=False, visible=False, elem_classes="preview-box")
                        text_preview  = gr.Code(
                            interactive=True,
                            lines=20,
                            visible=True,
                            elem_classes="preview-box",
                            elem_id="preview-text",
                        )

                    gr.HTML(
                        """
                        <style>
                            .minds-credit {
                                display: flex;
                                align-items: center;
                                gap: 16px;
                                justify-content: flex-end;   /* ← pushes items to the right */
                                text-align: middle;           /* ← aligns text inside */
                            }

                            .credit-text .credit-line {
                                display: block;
                            }
                        </style>

                        <div class="minds-credit">
                            <div class="credit-text">
                                <span class="credit-line">
                                    工业工程学科引擎底层智能体框架由 <strong><a href="https://github.com/MINDS-THU" target="_blank" rel="noopener noreferrer">MINDS-THU</a></strong> 设计、实现与维护
                                </span>
                                <span class="credit-line">
                                    如需构建与部署您自己的智能体，请访问我们的项目
                                    <a href="https://github.com/MINDS-THU/HAMLET" target="_blank" rel="noopener noreferrer">
                                        HAMLET
                                    </a>
                                </span>
                                <span class="credit-line">
                                    如有Bug报告或交流需求，欢迎发邮件至
                                    <a href="mailto:li.chuanhao@outlook.com">li.chuanhao@outlook.com</a>
                                </span>
                            </div>

                            <img src="https://avatars.githubusercontent.com/u/224852121?s=400&u=9715939e7952f8ac9b2f5806bdc185c14c3b5376&v=4"
                                alt="MINDS-THU logo"
                                style="width: 55px; height: auto;" />
                        </div>

                        """,
                        visible=True,
                        padding=True,
                    )

                    file_explorer.change(
                        show_preview,
                        [file_explorer],
                        [
                            pdf_preview,
                            img_preview,
                            table_preview,
                            text_preview,
                            preview_download,
                            copy_button,
                            delete_button,
                            fullscreen_button,
                            selected_file,
                        ],
                    )

                    delete_button.click(
                        delete_selected_file,
                        [selected_file],
                        [
                            pdf_preview,
                            img_preview,
                            table_preview,
                            text_preview,
                            preview_download,
                            copy_button,
                            delete_button,
                            fullscreen_button,
                            selected_file,
                        ],
                    ).then(
                        force_refresh,
                        None,
                        [file_explorer],
                    )

                    upload_button.upload(
                        self.upload_file,
                        [upload_button, file_uploads_log],
                        [upload_status_box, file_uploads_log],
                    ).then(
                        force_refresh,
                        None,
                        [file_explorer],
                    )

                    zip_download_button.click(
                        zip_workspace,
                        None,
                        [zip_download_button],
                    )

                    reset_button.click(
                        reset_workspace,
                        [file_uploads_log],
                        [
                            pdf_preview,
                            img_preview,
                            table_preview,
                            text_preview,
                            preview_download,
                            copy_button,
                            delete_button,
                            fullscreen_button,
                            selected_file,
                            file_uploads_log,
                            upload_status_box,
                        ],
                    ).then(
                        force_refresh,
                        None,
                        [file_explorer],
                    )

                    copy_button.click(
                        None,
                        [text_preview],
                        None,
                        js="""
                        (text) => {
                            if (!text || !navigator.clipboard) return;
                            navigator.clipboard.writeText(text).then(() => {
                                const btn = document.getElementById('preview_copy');
                                if (!btn) return;
                                btn.classList.add('copied');
                                setTimeout(() => btn.classList.remove('copied'), 700);
                            }).catch(err => console.warn('Clipboard copy failed', err));
                        }
                        """,
                    )

                    # ──────────────────────────────────────────────────────────────
                    # 4️⃣  JS hook – paste this whole block over the one you have now
                    # ──────────────────────────────────────────────────────────────
                    demo.load(
                        None, None, None,
                        js="""
                        () => {
                        if (window.__hamletPreviewToolbarBound) return;
                        window.__hamletPreviewToolbarBound = true;

                        const overlay = document.getElementById('preview-overlay');
                        const content = document.getElementById('overlay‑content');
                        const closeBtn = overlay?.querySelector('.overlay‑close');

                        const tooltipMap = {
                            preview_copy: '复制',
                            preview_download: '下载',
                            preview_fs: '最大化',
                            preview_delete: '删除',
                            files_upload: '上传文件',
                            files_zip: '打包下载工作区',
                            files_reset: '重置',
                        };

                        const findTarget = (node) => {
                            if (!node) return null;
                            const selector = 'button,label,[role="button"]';
                            if (node.matches?.(selector)) return node;
                            return node.querySelector?.(selector) || null;
                        };

                        const applyTooltips = () => {
                            Object.entries(tooltipMap).forEach(([id, text]) => {
                                const node = document.getElementById(id);
                                const target = findTarget(node);
                                if (!target) return;
                                target.setAttribute('title', text);
                                target.setAttribute('aria-label', text);
                            });
                        };
                        applyTooltips();

                        const previewToolbar = document.querySelector('.preview-toolbar');
                        if (previewToolbar) {
                            new MutationObserver(() => applyTooltips()).observe(previewToolbar, { childList: true, subtree: true });
                        }
                        const filesToolbar = document.querySelector('.files-toolbar');
                        if (filesToolbar) {
                            new MutationObserver(() => applyTooltips()).observe(filesToolbar, { childList: true, subtree: true });
                        }

                        const getVisibleBox = () =>
                            Array.from(document.querySelectorAll('.preview-box'))
                                .find(el => el.offsetParent !== null);
                        const enterFullscreen = () => {
                            const box = getVisibleBox();
                            if (!box || !overlay || !content) return;

                            const placeholder = document.createElement('div');
                            placeholder.style.display = 'none';
                            box.parentNode.insertBefore(placeholder, box);

                            content.innerHTML = '';
                            content.appendChild(box);

                            box.querySelectorAll('embed,iframe,object').forEach(el => {
                                el.style.width = '100%';
                                el.style.height = '100%';
                            });

                            overlay.style.display = 'flex';
                            window.dispatchEvent(new Event('resize'));

                            const close = () => {
                                if (placeholder.parentNode) {
                                    placeholder.parentNode.replaceChild(box, placeholder);
                                }
                                overlay.style.display = 'none';
                            };

                            overlay.onclick = null;
                            if (closeBtn) closeBtn.onclick = null;

                            overlay.addEventListener('click', (ev) => {
                                if (ev.target === overlay) close();
                            }, { once: true });

                            if (closeBtn) {
                                closeBtn.addEventListener('click', (ev) => {
                                    ev.stopPropagation();
                                    close();
                                }, { once: true });
                            }
                        };

                        document.addEventListener('click', (event) => {
                            const fsButton = event.target.closest('#preview_fs button, #preview_fs');
                            if (fsButton) {
                                event.preventDefault();
                                enterFullscreen();
                                return;
                            }
                        });
                        }
                        """
                    )


            # ----- chat helpers & events ------------------------------------
            def handle_prompt(prompt, uploads, history):
                """Log user message, clear textbox, disable button."""
                new_hist, cleared, _ = self.log_user_message(prompt, uploads)
                return new_hist, "", gr.update(interactive=False)

            def reenable_ui():
                """Re‑enable textbox & button after agent is done."""
                return gr.update(interactive=True), gr.update(interactive=True)

            # textbox ↩️
            text_input.submit(
                handle_prompt,
                [text_input, file_uploads_log, stored_messages],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                force_refresh,            # LLM might have touched the FS
                None,
                [file_explorer],
            ).then(
                reenable_ui,
                None,
                [text_input, submit_btn],
            )

            # submit button 🖱️
            submit_btn.click(
                handle_prompt,
                [text_input, file_uploads_log, stored_messages],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_state],
                [chatbot],
            ).then(
                force_refresh,
                None,
                [file_explorer],
            ).then(
                reenable_ui,
                None,
                [text_input, submit_btn],
            )



        return demo





__all__ = ["stream_to_gradio", "GradioUI"]
