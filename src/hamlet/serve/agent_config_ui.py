#!/usr/bin/env python
# coding=utf-8

import os
import json
import tempfile
import threading
from typing import Dict, List
from pathlib import Path

#!/usr/bin/env python
# coding=utf-8

import os
import re
from typing import Optional

from hamlet.core.agent_types import AgentAudio, AgentImage, AgentText
from hamlet.core.agents import MultiStepAgent, PlanningStep
from hamlet.core.memory import ActionStep, FinalAnswerStep, MemoryStep
from hamlet.core.utils import _is_package_available
from hamlet.core.models import ChatMessageStreamDelta


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
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Step"
        if not skip_model_outputs:
            yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if (not skip_model_outputs) and hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content)
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
        ):
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
        if not skip_model_outputs:
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


class AgentConfigManager:
    """Class for managing agent configurations"""
    
    def __init__(self, config_dir: str = "agent_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        # Default (global) config file; may be overridden per user
        self.agent_configs_file = self.config_dir / "agent_configs.json"
        self._user_id = None
        # In‑process lock to avoid concurrent writers clobbering temp file
        self._lock = threading.Lock()
        # Note: We no longer use tool_configs.json since we have dynamic discovery
        
        # Initialize configuration files
        self._init_config_files()

    # ---------------- Per-User Support ----------------
    def set_user(self, user_id: str):
        """Set the active user so configs are isolated per student.

        A sanitized and hashed filename is created to avoid exposing raw identifiers
        in directory listings while still being deterministic for returning students.
        """
        import re, hashlib
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty.")
        raw = user_id.strip()
        # Sanitize (allow alnum, dash, underscore)
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', raw)
        if not safe:
            safe = 'user'
        # Short hash to avoid collisions / leaking full ID
        short_hash = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
        self._user_id = safe
        self.agent_configs_file = self.config_dir / f"{safe}_{short_hash}_agent_configs.json"
        if not self.agent_configs_file.exists():
            # Initialize with defaults for brand-new user
            self._init_config_files(force=True)
        else:
            # Do nothing; existing file will be used
            pass

    def get_user(self) -> Optional[str]:  # type: ignore[name-defined]
        return self._user_id
    
    def _init_config_files(self, force: bool = False):
        """Initialize configuration files with default values"""
        if force or not self.agent_configs_file.exists():
            self.save_agent_configs(self._build_example_seed())
        
        # Note: We no longer create tool_configs.json as we use dynamic discovery
    
    def get_all_agent_metadata(self) -> Dict[str, Dict]:
        """Get all agent configurations"""
        if self.agent_configs_file.exists():
            try:
                with open(self.agent_configs_file, 'r', encoding='utf-8') as f:
                    data = f.read().strip()
                    if not data:
                        raise ValueError("empty config file")
                    return json.loads(data)
            except Exception as e:
                print(f"[AgentConfigManager] Failed to read config '{self.agent_configs_file}' ({e}); attempting recovery.")
                self._init_config_files(force=True)
                try:
                    with open(self.agent_configs_file, 'r', encoding='utf-8') as f2:
                        return json.load(f2)
                except Exception:
                    return {}
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
        # Use unique temp file + lock to prevent race where two writers move the same temp file.
        with self._lock:
            # Ensure target directory still exists (may have been removed externally).
            self.config_dir.mkdir(exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.config_dir),
                prefix=self.agent_configs_file.stem + "_",
                suffix=".tmp"
            )
            tmp_file = Path(tmp_path)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(configs, f, indent=2, ensure_ascii=False)
                try:
                    os.replace(tmp_file, self.agent_configs_file)
                except FileNotFoundError:
                    # Rare case: tmp file vanished (external cleanup) OR directory removed.
                    # Fall back to a direct write.
                    with open(self.agent_configs_file, 'w', encoding='utf-8') as f:
                        json.dump(configs, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    # As a fallback to preserve user data, attempt direct write.
                    with open(self.agent_configs_file, 'w', encoding='utf-8') as f:
                        json.dump(configs, f, indent=2, ensure_ascii=False)
                    print(f"[AgentConfigManager] Warning: atomic replace failed ({e}); used direct write.")
            finally:
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass
    
    def add_agent_config(self, name: str, config: Dict):
        """Add new agent configuration"""
        configs = self.get_all_agent_metadata()
        configs[name] = config
        self.save_agent_configs(configs) 
    
    def _build_example_seed(self) -> Dict[str, Dict]:
        """构建文献搜索智能体（用于首次初始化或损坏恢复）。"""
        example_name = "文献搜索智能体"
        tools = [
            "web_search",                 # 检索
            "get_paper_from_url",         # 批量抓取论文全文
            "create_file_with_content",   # 初次写入 Markdown
            "see_text_file",              # 查看文件内容
            "modify_file",                # 局部修改
            "list_dir",                   # 目录 / 去重 / 避免覆盖
            "delete_file_or_folder"       # 仅在用户明确要求删除时使用
        ]
        description = (
            "根据用户提供的研究主题自动检索相关论文，下载正文，抽取与整理关键信息，"
            "并生成包含每篇论文标题、原始链接及结构化总结的报告 (literature_report.md)。"
            "支持增量追加、主题澄清与文件化存档。"
        )
        prompt = (
            "你是一名面向科研人员的“文献搜索与综述助手”。你的核心任务：根据用户给出的研究主题或问题，自动检索、获取、整理并生成结构化的多篇论文综述报告。\n\n"
            "【总体流程】\n"
            "1. 任务理解与澄清：主题过宽或含糊先提出 1~2 个澄清问题（领域 / 年限 / 期望篇数）。默认 5 篇。\n"
            "2. 初步检索 (web_search)：生成 2~3 组多样化查询（同义词/方法/场景）；去重筛选学术来源。\n"
            "3. 论文获取 (get_paper_from_url)：收集前 N 条候选 PDF/页面，一次批量抓取；失败记录但不阻塞整体。\n"
            "4. 本地文件：为每篇创建 papers/<slug>.md，结构含：标题、Source、摘要清洗、研究问题、数据/场景、方法要点、主要结果(定量优先)、优势与创新、局限性、与主题相关性。\n"
            "5. 生成 literature_report.md：目录 + 每篇 100~180 字精炼总结 + 一句话启示。\n"
            "6. 对话输出：简要预览统计与后续可选操作（追加检索 / 缩小范围 / 生成对比表）。\n\n"
            "【工具使用准则】\n"
            "- web_search：若结果不足或用户要求追加再调用；避免无节制循环。\n"
            "- get_paper_from_url：批量一次；失败不多于 1 次重试。\n"
            "- create_file_with_content：首次写文件；改动用 modify_file。\n"
            "- modify_file：仅改最小必要行；前置 see_text_file 定位。\n"
            "- list_dir：防止覆盖 / 去重 / 统计。\n"
            "- delete_file_or_folder：除非用户明确说明删除对象。\n\n"
            "【质量与安全】\n"
            "- 不杜撰未出现的数据；缺失写“(原文未提供)”。\n"
            "- 摘要不足时可用正文开头补齐并标注“(补充)”。\n"
            "- 中文总结；英文标题可保留。\n"
            "- 相似方法多篇出现时在报告中做简洁对比。\n"
            "- 控制下载数量：若候选 > 需求 2 倍先本地筛选再抓取。\n\n"
            "【回答格式（对话）】\n"
            "- 需澄清：列出澄清点并等待用户。\n"
            "- 已完成检索抓取：说明成功篇数、报告文件名称、后续可选操作菜单。\n"
            "- 部分失败：列出失败 URL + 原因摘要。\n\n"
            "现在等待用户输入的研究主题。"
        )
        return {example_name: {"prompt": prompt, "description": description, "tools": tools, "sub_agents": [], "agent_type": "CodeAgent"}}