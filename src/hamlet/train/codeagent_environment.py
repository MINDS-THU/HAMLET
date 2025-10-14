import inspect
import logging
import time
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List
import json

from openai import AsyncOpenAI

from src.hamlet.core.agents import CodeAgent
from src.hamlet.core.models import (
    MessageRole,
    Model,
    OpenAIServerModel,
)
from src.hamlet.core.utils import (
    parse_code_blobs, 
    extract_code_from_text,
)
from src.hamlet.core.local_python_executor import fix_final_answer_code

from src.hamlet.train.environment import Environment
from src.hamlet.train.utils.async_utils import maybe_await
from src.hamlet.train.utils.types import (
    Info,
    Messages,
    SamplingArgs,
    State,
)
from src.hamlet.train.rubric import Rubric

logger = logging.getLogger("src.hamlet.train.codeagent_environment")

class MultiTurnEnv(Environment):
    def __init__(self, max_turns: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns

    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    async def max_turns_reached(self, state: State) -> bool:
        """Check if the maximum number of turns has been reached."""
        return state["turn"] >= self.max_turns and self.max_turns > 0

    async def setup_state(self, state: State, **kwargs) -> State:
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """When overriding, call self.max_turns_reached(state) to check if turn limit reached."""
        max_turns_reached = await self.max_turns_reached(state)
        prompt_too_long = await self.prompt_too_long(state)
        return max_turns_reached or prompt_too_long

    @abstractmethod
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        pass

class CodeAgentRubric(Rubric):
    def __init__(
        self,
        agent: CodeAgent | None,
    ):
        super().__init__(parser=None)
        self.agent = agent
        self.reward_funcs = [
            self.successful_parsing_reward_func,
            self.successful_code_execution_reward_func,
            # self.correct_answer_reward_func,
        ]
        self.reward_weights = [
            0.5,
            0.5,
            # 1.0,
        ]

    def successful_parsing_reward_func(
        self, completion: List[Dict[str, str]], **kwargs
    ) -> float:
        assert self.agent is not None
        total_attempts = 0
        successful_parsing = 0
        for i, msg in enumerate(completion):
            if msg["role"] == "assistant" or msg["role"] == MessageRole.ASSISTANT:
                model_output = msg["content"]
                total_attempts += 1
                # check if we can parse the code block
                try:
                    if not self.agent._use_structured_outputs_internally:
                        if model_output and not model_output.strip().endswith(self.agent.code_block_tags[1]):
                            model_output += self.agent.code_block_tags[1]
                        code_action = parse_code_blobs(model_output, self.agent.code_block_tags)
                    else:
                        code_action = json.loads(model_output)["code"]
                        code_action = extract_code_from_text(code_action, self.agent.code_block_tags) or code_action
                    code_action = fix_final_answer_code(code_action)

                    successful_parsing += 1
                except Exception:
                    pass

        # Calculate reward
        if total_attempts == 0:
            return 1.0
        return successful_parsing / total_attempts

    def successful_code_execution_reward_func(
        self, completion: List[Dict[str, str]], **kwargs
    ) -> float:
        assert self.agent is not None
        total_attempts = 0
        successful_executions = 0
        for i, msg in enumerate(completion):
            if msg["role"] == "assistant" or msg["role"] == MessageRole.ASSISTANT:
                model_output = msg["content"]
                # check if we can parse the code block
                try:
                    if not self.agent._use_structured_outputs_internally:
                        if model_output and not model_output.strip().endswith(self.agent.code_block_tags[1]):
                            model_output += self.agent.code_block_tags[1]
                        code_action = parse_code_blobs(model_output, self.agent.code_block_tags)
                    else:
                        code_action = json.loads(model_output)["code"]
                        code_action = extract_code_from_text(code_action, self.agent.code_block_tags) or code_action
                    code_action = fix_final_answer_code(code_action)
                except Exception:
                    continue
                total_attempts += 1
                # Execute
                try:
                    code_output = self.agent.python_executor(
                        code_action
                    )
                    successful_executions += 1
                except Exception as e:
                    print("Execution error:", e)
                    pass

        # Calculate reward
        if total_attempts == 0:
            return 0.0
        return successful_executions / total_attempts

    # def correct_answer_reward_func(self, completion, answer, **kwargs) -> float:
    #     """Reward function that checks if the final answer matches the expected answer."""
    #     # get the last assistant message, note that complete[-1] might be user message
    #     model_output = (
    #         completion[-1]["content"]
    #         if completion[-1]["role"] == "assistant"
    #         else completion[-2]["content"]
    #     )
    #     print("====== correct answer reward func ======")
    #     print("completion:", completion)
    #     # Parse
    #     try:
    #         code_action = fix_final_answer_code(parse_code_blobs(model_output))
    #     except Exception:
    #         return 0.0
    #     # Execute
    #     try:
    #         output, execution_logs, terminate = self.python_executor(
    #             code_action, terminal_tools=self.terminal_tools + ["final_answer"]
    #         )
    #     except Exception:
    #         return 0.0

    #     if terminate:
    #         print("Parsed final response:", output)
    #         print("Expected answer:", answer)
    #         return 1.0 if abs(float(answer) - float(output)) <= 1e-3 else 0.0
    #     else:
    #         return 0.0

class CodeAgentEnv(MultiTurnEnv):
    def __init__(
        self,
        agent_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        model_class: type[Model] | None = None,
        **kwargs,
    ):
        """Create an environment that wraps a CodeAgent for multi-turn rollouts.

                This environment adapts the training pipeline's chat-style prompts to a
                CodeAgent run. It constructs the underlying Model and CodeAgent lazily on the
                first rollout and reuses them across calls unless the (model_id, base_url,
                api_key) signature changes.

                Parameters
                - agent_kwargs: dict | None
                        Keyword arguments forwarded to the CodeAgent constructor, e.g.:
                        - tools: list[Tool] (optional). You can pass tools here or via the top-level
                            ``tools`` kwarg below. If both are provided, a ValueError is raised.
                        - Other CodeAgent options such as max_steps, instructions, planning_interval,
                            managed_agents, verbosity, etc. (any valid CodeAgent kwargs).

                - model_kwargs: dict | None
                        Keyword arguments forwarded to the model_class constructor. The environment
                        will automatically inject:
                        - model_id: set per rollout from the ``model`` argument passed to ``rollout``.
                        - base_url and api_key: if present on the AsyncOpenAI client passed to
                            ``rollout``, and if the selected model_class accepts these parameters.
                        Note: generation parameters such as temperature, max_tokens, etc. should be
                        provided via ``sampling_args`` (Environment API), not model_kwargs. Those
                        are normalized and pushed into the model's runtime kwargs at rollout time.

                - model_class: type[Model] | None
                        The concrete model class to instantiate. Defaults to OpenAIServerModel.
                        Any subclass of ``hamlet.core.models.Model`` is supported as long as its
                        constructor matches the provided model_kwargs and the auto-injected fields
                        described above.

                - **kwargs: Any
                        Forwarded to the parent Environment and MultiTurnEnv initializers. Common
                        fields include:
                        - dataset, eval_dataset: datasets.Dataset, at least one must be provided or
                            the base Environment will raise a ValueError.
                        - system_prompt: Optional[str]
                        - few_shot: Optional[list[ChatMessage]]
                        - parser: Optional[Parser]
                        - rubric: Optional[Rubric] (will be overridden at rollout by CodeAgentRubric)
                        - sampling_args: Optional[SamplingArgs] (merged and applied at rollout)
                        - message_type: Literal["chat", "completion"]. Must be "chat" for CodeAgentEnv;
                            using "completion" will fail in rollout.
                        - oai_tools: Optional[list[ChatCompletionToolParam]]
                        - max_workers: int
                        - max_turns: int (from MultiTurnEnv) to cap number of turns in this environment
                        - tools: Optional[list[Tool]] — convenience alias for agent tools. If provided
                            here, they are moved into agent_kwargs["tools"]. Do not pass tools in both
                            places.

                Behavior and constraints
                - Tools source of truth: Either pass tools via ``tools`` in **kwargs or inside
                    ``agent_kwargs['tools']``. Passing both raises a ValueError.
                - Message type: Only chat-style prompts are supported. ``rollout`` raises if
                    message_type != "chat".
                - Lazy initialization: The model and agent are created on first rollout and
                    recreated only when (model_id, base_url, api_key) changes.
                - Sampling args handling: At rollout, ``sampling_args`` are merged with the
                    environment defaults and normalized (e.g., ``max_completion_tokens`` vs
                    ``max_tokens``). These are pushed into the model's runtime kwargs so agent
                    generations respect them.
                - Rubric: The rubric is replaced per rollout with a CodeAgent-specific rubric
                    that rewards successful parsing and code execution.

    Example
    -------
    >>> from datasets import Dataset
    >>> data = Dataset.from_dict({"question": ["What is 2+2?"], "answer": ["4"]})
    >>> env = CodeAgentEnv(
    ...     agent_kwargs={"tools": []},
    ...     model_kwargs={"timeout": 30},
    ...     dataset=data,
    ...     message_type="chat",
    ...     max_turns=3,
    ... )
    """
        # Extract agent-specific kwargs without leaking them into the base Environment.
        agent_kwargs = deepcopy(agent_kwargs) if agent_kwargs is not None else {}
        tools = kwargs.pop("tools", None)
        if tools is not None and "tools" in agent_kwargs:
            raise ValueError(
                "Tools were provided both at the top-level and inside 'agent_kwargs'. Please choose one location."
            )
        if tools is not None:
            agent_kwargs["tools"] = tools
        agent_kwargs.setdefault("tools", [])
        self.agent_kwargs = agent_kwargs

        self.model_kwargs = deepcopy(model_kwargs) if model_kwargs is not None else {}
        self.model_class: type[Model] = model_class or OpenAIServerModel
        self._model: Model | None = None
        self._agent: CodeAgent | None = None
        # CodeAgent(model=OpenAIServerModel(model_id="gpt-5"), **agent_kwargs)
        self._model_signature: tuple[str, str | None, str | None] | None = None
        super().__init__(**kwargs)
        self.rubric = CodeAgentRubric(agent=self._agent)  # will be reset when rollout is called

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        if self.message_type != "chat":
            raise ValueError("CodeAgentEnv currently only supports 'chat' message_type.")
        # Prepare/merge sampling args and apply them to the underlying model so
        # the agent's LLM calls honor generation parameters (temperature, max_tokens, etc.).
        merged_sampling: dict[str, Any] = {}
        if hasattr(self, "sampling_args") and isinstance(getattr(self, "sampling_args"), dict):  # from Environment
            merged_sampling.update(getattr(self, "sampling_args"))
        if sampling_args is not None:
            merged_sampling.update(sampling_args)

        # Translate/clean sampling args for OpenAI Chat Completions style models
        def _prepare_model_kwargs_from_sampling(s_args: dict[str, Any]) -> dict[str, Any]:
            params: dict[str, Any] = {k: v for k, v in (s_args or {}).items() if v is not None}
            # Normalize token limit naming for OpenAI chat.completions
            if "max_completion_tokens" in params and "max_tokens" not in params:
                params["max_tokens"] = params.pop("max_completion_tokens")
            # Keep extra_body if provided (supported by openai-python client)
            extra_body = params.pop("extra_body", None)
            if extra_body is not None:
                params["extra_body"] = extra_body
            # Avoid passing fields that the agent manages explicitly
            params.pop("tools", None)
            params.pop("response_format", None)  # agent may set structured outputs internally
            return params

        model_generation_kwargs = _prepare_model_kwargs_from_sampling(merged_sampling)

        # Ensure model/agent are initialized, then inject generation kwargs into the model
        self._ensure_agent(client=client, model_id=model)
        self.rubric = CodeAgentRubric(agent=self._agent)
        if self._model is None:
            raise RuntimeError("Model initialisation failed.")
        # Update model kwargs used by Model.generate; do not include None values
        try:
            self._model.kwargs.update(model_generation_kwargs)
        except Exception:
            # Be resilient: if a specific model doesn't accept some keys, ignore silently here
            for k, v in model_generation_kwargs.items():
                try:
                    self._model.kwargs[k] = v
                except Exception:
                    pass

        info = info or {}
        state: State = {
            "id": 0,  # TODO: add id
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],
            "turn": 0,
            "timing": {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
            },
        }
        start_time = time.time()
        state = await maybe_await(self.setup_state, state, **kwargs)
        if not isinstance(prompt, list):
            raise ValueError("CodeAgentEnv expects chat prompts as a list of messages.")

        completion = []
        rollout = list(prompt) if not isinstance(prompt, str) else prompt

        task_text = self._extract_task_from_prompt(prompt, default=task)
        if self._agent is None:
            raise RuntimeError("CodeAgent has not been initialised correctly.")
        full_result = self._agent.run(task_text, return_full_result=True, reset=True)
        end_time = time.time()
        state["timing"]["generation_ms"] = (end_time - start_time) * 1000
        state["timing"]["total_ms"] = (end_time - start_time) * 1000
        state["turn"] = full_result.steps[-1]["step_number"]
        state["full_steps"] = full_result.steps
        # Convert the agent's full_result steps into format compatible with the training pipeline
        messages = full_result.steps[-1]["model_input_messages"] + [full_result.steps[-1]["model_output_message"]]
        for ind, message in enumerate(messages):
            # skip system prompt and the first user message, which are already stored in rollout
            if message["role"] == "system" or message["role"] == "tool-call":
                continue
            elif ind == 1:
                assert message["role"] == "user"
                continue
            else:
                if message["role"] == "assistant":
                    state["responses"].append(message)
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                if message["role"] == "tool-response":
                    if isinstance(message["content"], str):
                        rollout.append({"role": "user", "content": message["content"]})
                        completion.append({"role": "user", "content": message["content"]})
                    elif isinstance(message["content"], dict) and "text" in message["content"]:
                        rollout.append({"role": "user", "content": message["content"]["text"]})
                        completion.append({"role": "user", "content": message["content"]["text"]})
                    elif isinstance(message["content"], list) and len(message["content"]) > 0 and isinstance(message["content"][0], dict) and "text" in message["content"][0]:
                        rollout.append({"role": "user", "content": message["content"][0]["text"]})
                        completion.append({"role": "user", "content": message["content"][0]["text"]})
                    else:
                        raise ValueError(f"Unexpected tool-response content format: {message['content']}")
                else:
                    if isinstance(message["content"], str):
                        rollout.append({"role": message["role"], "content": message["content"]})
                        completion.append({"role": message["role"], "content": message["content"]})
                    elif isinstance(message["content"], dict) and "text" in message["content"]:
                        rollout.append({"role": message["role"], "content": message["content"]["text"]})
                        completion.append({"role": message["role"], "content": message["content"]["text"]})
                    elif isinstance(message["content"], list) and len(message["content"]) > 0 and isinstance(message["content"][0], dict) and "text" in message["content"][0]:
                        rollout.append({"role": message["role"], "content": message["content"][0]["text"]})
                        completion.append({"role": message["role"], "content": message["content"][0]["text"]})
                    else:
                        raise ValueError(f"Unexpected tool-response content format: {message['content']}")

        state["completion"] = completion
        return completion, state
        # Convert memory steps into chat messages, excluding the initial TaskStep that mirrors the user input.
        # new_messages: list[ChatCompletionMessageParam] = []
        # for memory_step in self._agent.memory.steps:
        #     if isinstance(memory_step, TaskStep):
        #         continue
        #     step_messages = memory_step.to_messages(summary_mode=False)
        #     if not step_messages:
        #         continue
        #     converted = self._convert_messages(step_messages)
        #     new_messages.extend(converted)

        # if full_result.output is not None:
        #     final_message: ChatCompletionMessageParam = {
        #         "role": "assistant",
        #         "content": str(full_result.output),
        #     }
        #     if not new_messages or cast(dict[str, Any], new_messages[-1]).get("content") != final_message["content"]:
        #         new_messages.append(final_message)

        # Update rollout and completion buffers
        # rollout_messages.extend(new_messages)
        # completion_messages = new_messages

        # assistant_messages = [msg for msg in new_messages if msg.get("role") == "assistant"]
        # state["responses"].extend(assistant_messages)
        # state["completion"] = completion_messages
        # state["full_steps"] = full_result.steps
        # if full_result.token_usage is not None:
        #     state["token_usage"] = {
        #         "input_tokens": full_result.token_usage.input_tokens,
        #         "output_tokens": full_result.token_usage.output_tokens,
        #     }

        # return completion_messages, state

    def _ensure_agent(self, client: AsyncOpenAI, model_id: str) -> None:
        """Initialise or refresh the underlying CodeAgent and model."""

        connection = self._extract_client_connection(client)
        signature = (model_id, connection.get("base_url"), connection.get("api_key"))

        model_needs_init = self._model_signature != signature
        if model_needs_init:
            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None
            model_kwargs = {**self.model_kwargs}
            init_params = inspect.signature(self.model_class.__init__).parameters
            if "base_url" in init_params and connection.get("base_url") is not None:
                model_kwargs["base_url"] = connection["base_url"]
            if "api_key" in init_params and connection.get("api_key") is not None:
                model_kwargs.setdefault("api_key", connection["api_key"])
            model_kwargs["model_id"] = model_id
            self._model = self.model_class(**model_kwargs)
            self._model_signature = signature

        if self._agent is None:
            agent_kwargs = {**self.agent_kwargs}
            assert self._model is not None, "Model initialisation failed."
            self._agent = CodeAgent(model=self._model, **agent_kwargs)

    @staticmethod
    def _extract_task_from_prompt(prompt: Messages, default: str | None = None) -> str:
        if isinstance(prompt, str):
            return prompt
        user_messages: list[str] = []
        for message in prompt:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                user_messages.append("\n".join(part for part in text_parts if part))
            else:
                user_messages.append(str(content))
        if user_messages:
            return user_messages[-1]
        return default or ""

    # @staticmethod
    # def _convert_messages(messages: Iterable[ChatMessage]) -> list[ChatCompletionMessageParam]:
    #     # print("========= messages ===============")
    #     # print(messages)
    #     # print("========= end messages ===============")
    #     cleaned = get_clean_message_list(
    #         list(messages),
    #         role_conversions=tool_role_conversions,
    #         flatten_messages_as_text=True,
    #     )
    #     # print("========= cleaned ===============")
    #     # print(cleaned)
    #     # print("========= end cleaned ===============")
    #     formatted: list[ChatCompletionMessageParam] = []
    #     for message in cleaned:
    #         role = message.get("role", "assistant")
    #         content = message.get("content")
    #         if content is None or not isinstance(content, str):
    #             continue
    #         formatted.append({"role": role, "content": content})
    #     return formatted

    @staticmethod
    def _extract_client_connection(client: AsyncOpenAI) -> dict[str, str | None]:
        base_url = getattr(client, "base_url", None)
        if base_url is None:
            inner_client = getattr(client, "_client", None)
            base_url_obj = getattr(inner_client, "_base_url", None)
            if base_url_obj is not None:
                base_url = str(base_url_obj)
        elif hasattr(base_url, "__str__") and not isinstance(base_url, str):
            base_url = str(base_url)

        api_key = getattr(client, "api_key", None)
        if api_key is None:
            inner_client = getattr(client, "_client", None)
            api_key = getattr(inner_client, "_api_key", None)

        return {"base_url": base_url, "api_key": api_key}