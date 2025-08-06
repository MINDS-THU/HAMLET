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
import os
import re
import shutil
from typing import Optional

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from smolagents.utils import _is_package_available
from smolagents.models import ChatMessageStreamDelta  # NEW


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
        final_answer = step_log.final_answer
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

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

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
            allowed_file_types = [".pdf", ".docx", ".txt", ".md", ".py", ".csv", ".tsv", ".xlsx", ".json", ".jsonl", ".yaml", ".yml"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores
        print("DEBUG – sanitized_name:", sanitized_name)
        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        # return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]
        relative_path = os.path.relpath(file_path, self.file_upload_folder)
        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [relative_path]


    def log_user_message(self, text_input, file_uploads_log):
        import gradio as gr

        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
            gr.Button(interactive=False),
        )

    def launch(self, share: bool = True, **kwargs):
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        custom_css = r"""
        /* --------------------------------------------------------------
        1. Global tweaks – remove Gradio’s 80 rem cap
        ----------------------------------------------------------------*/
        html, body, #root, .gradio-container,
        [class*="gradio-container"]{
            max-width:100%!important;
            width:100%!important;
            margin:0!important;
            padding:0!important;
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
        .vscode-header{background:#f6f6f6;font-weight:600;padding:4px 8px;}

        .vscode-pane .gr-file-explorer{flex:0 0 25vh;overflow:auto;}

        .vscode-pane .gr-code,
        .vscode-pane .gr-image,
        .vscode-pane .gr-dataframe,
        .vscode-pane .gr-pdf{
            flex:1 1 auto;min-height:0;overflow:auto;
            width:100%!important;box-sizing:border-box;
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

        def show_preview(selection):
            """Return component updates so the right viewer shows the file."""
            hidden = gr.update(visible=False)

            if not selection:
                return hidden, hidden, hidden, gr.update(
                    value="⬅️ Click a file to preview", visible=True
                )

            rel_path = selection[0] if isinstance(selection, list) else selection
            abs_path = os.path.join(folder, rel_path)

            if os.path.isdir(abs_path):
                return hidden, hidden, hidden, gr.update(
                    value="📁 Directory (no preview)", visible=True
                )

            ext = os.path.splitext(abs_path)[1].lower()

            # ---------- IMAGES ----------
            if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                return hidden, gr.update(value=abs_path, visible=True), hidden, hidden

            # ---------- PDF -------------
            if ext == ".pdf":
                return gr.update(value=abs_path, visible=True), hidden, hidden, hidden

            # ---------- SPREADSHEETS ----
            if ext in {".csv", ".tsv", ".xlsx"}:
                try:
                    df = (pd.read_csv if ext != ".xlsx" else pd.read_excel)(abs_path)
                    return hidden, hidden, gr.update(value=df, visible=True), hidden
                except Exception as e:
                    msg = f"⚠️ Cannot read file: {e}"
                    return hidden, hidden, hidden, gr.update(value=msg, visible=True)

            # ---------- TEXT / fallback -
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(20_000)
            except Exception as e:
                txt = f"⚠️ Cannot display file: {e}"
            return hidden, hidden, hidden, gr.update(value=txt, visible=True)


        def force_refresh():
            """
            Force‑refresh the FileExplorer by tweaking ignore_glob
            with a random pattern that matches nothing.
            """
            dummy_glob = f"__refresh_{int(time.time()*1000)}__"
            return gr.update(ignore_glob=dummy_glob)

        # ---------- UI -------------------------------------------------------
        with gr.Blocks(theme="ocean", fill_height=True, css=custom_css) as demo:
            session_state    = gr.State({})
            stored_messages  = gr.State([])
            file_uploads_log = gr.State([])

            # ----- sidebar ---------------------------------------------------
            with gr.Sidebar():
                gr.Markdown(f"# {self.name.title()}")
                if self.description:
                    gr.Markdown(f"**Agent:** {self.description}")

                text_input = gr.Textbox(lines=3, placeholder="Type your prompt…")
                submit_btn = gr.Button("Submit", variant="primary")

                # file‑upload widget
                if self.file_upload_folder:
                    upload_file   = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(visible=False, interactive=False)

            # ----- main area -------------------------------------------------
            with gr.Row(scale=12):                 # full width minus sidebar
                # ---- (A) Chat column ---------------------------------------
                with gr.Column(scale=7, min_width=590):
                    chatbot = gr.Chatbot(
                        label="Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                        ),
                        height="100vh",     # ← responsive: 70 % of the browser window
                        resizeable=True,
                    )

                # ---- (B) Explorer + preview column -------------------------
                with gr.Column(scale=5, min_width=350, elem_classes="vscode-pane"):
                    
                    # 1️⃣ FILE‑EXPLORER
                    gr.HTML("<div class='vscode-header'>Files</div>")
                    file_explorer = gr.FileExplorer(
                        root_dir=folder, file_count="multiple", interactive=True, value=None,
                        height="25vh",
                    )

                    # 2️⃣ PREVIEW header + working full‑screen button
                    gr.HTML(
                        """<div class='vscode-header'>
                            Preview <span class='fullscreen-btn' id='preview_fs'>🗖</span>
                        </div>"""
                    )

                    # put ONE overlay somewhere in the page (fixed‑position → location irrelevant)
                    # gr.HTML(LIGHTBOX_HTML, visible=True)
                    gr.HTML(LIGHTBOX_HTML, visible=True, elem_classes="overlay-wrapper")

                    # 3️⃣ PREVIEW widgets
                    pdf_preview   = PDF(
                        visible=False,
                        interactive=False,
                        # height=400,           # ← ADD THIS LINE
                        elem_classes="preview-box"
                    )
                    img_preview   = gr.Image(interactive=False, visible=False, elem_classes="preview-box")
                    table_preview = gr.Dataframe(interactive=False, visible=False, elem_classes="preview-box")
                    text_preview  = gr.Code(interactive=False, lines=20, visible=True, elem_classes="preview-box")

                    file_explorer.change(
                        show_preview,
                        [file_explorer],
                        [pdf_preview, img_preview, table_preview, text_preview],
                    )

                    # ──────────────────────────────────────────────────────────────
                    # 4️⃣  JS hook – paste this whole block over the one you have now
                    # ──────────────────────────────────────────────────────────────
                    demo.load(
                        None, None, None,
                        js="""                // ← this is the block you already pasted
                        () => {
                        const fsBtn    = document.getElementById('preview_fs');
                        const overlay  = document.getElementById('preview-overlay');
                        const content  = document.getElementById('overlay‑content');
                        const closeBtn = overlay.querySelector('.overlay‑close');

                        if (!fsBtn || fsBtn.dataset.bound) return;
                        fsBtn.dataset.bound = 1;

                        fsBtn.addEventListener('click', () => {
                            const box = Array.from(document.querySelectorAll('.preview-box'))
                                            .find(el => el.offsetParent !== null);
                            if (!box) return;

                            const placeholder = document.createElement('div');
                            placeholder.style.display = 'none';
                            box.parentNode.insertBefore(placeholder, box);

                            content.innerHTML = '';
                            content.appendChild(box);

                            box.querySelectorAll('embed,iframe,object').forEach(el => {
                            el.style.width  = '100%';
                            el.style.height = '100%';
                            });

                            overlay.style.display = 'flex';
                            window.dispatchEvent(new Event('resize'));
                            /* ---------- close logic ------------------------------------ */
                            const close = () => {
                            // put the live box back where it came from
                            if (placeholder.parentNode)
                                placeholder.parentNode.replaceChild(box, placeholder);

                            overlay.style.display = 'none';
                            };

                            /* ‑‑‑ make sure we don’t pile up multiple listeners ‑‑‑ */
                            overlay.onclick  = null;
                            closeBtn.onclick = null;

                            /* grey background click */
                            overlay.addEventListener('click', (ev) => {
                            if (ev.target === overlay) close();
                            }, { once:true });

                            /* little × click */
                            closeBtn.addEventListener('click', (ev) => {
                            ev.stopPropagation();       // don’t bubble to overlay
                            close();
                            }, { once:true });
                            /* ----------------------------------------------------------- */
                        });
                        }
                        """
                    )





            # now that file_explorer exists, wire **upload** → refresh
            if self.file_upload_folder:
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                ).then(
                    force_refresh,          # immediately refresh explorer
                    None,
                    [file_explorer],
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
