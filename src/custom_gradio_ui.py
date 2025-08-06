#!/usr/bin/env python
# coding=utf-8

import os
import re
from typing import Optional

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from smolagents.utils import _is_package_available
from smolagents.models import ChatMessageStreamDelta


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


__all__ = ["stream_to_gradio"]
