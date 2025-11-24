from typing import Callable, Any, Tuple, Mapping, Optional
from wrapt import wrap_function_wrapper
from opentelemetry import trace as trace_api, context as context_api
from opentelemetry.context import Context
from openinference.instrumentation import OITracer, TraceConfig, get_attributes_from_context
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from openinference.instrumentation.smolagents._wrappers import (
    _bind_arguments,
    _get_input_value,
    _smolagent_run_attributes,
    _tools,
    _output_value_and_mime_type_for_tool_span,
    _ModelWrapper,
    _StepWrapper,
)
from hamlet.core import models
from hamlet.core.agents import CodeAgent, ToolCallingAgent, MultiStepAgent
from hamlet.core.tools import Tool
import smolagents

# Constants
AGENT = OpenInferenceSpanKindValues.AGENT.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE


class CustomRunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        print("✅ CustomRunWrapper is ACTIVE")

        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        agent_name = getattr(instance, "name", instance.__class__.__name__)
        span_name = f"{instance.__class__.__name__}.run"
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            context=Context(),
            attributes={
                "openinference.agent.name": agent_name,
                SpanAttributes.OPENINFERENCE_SPAN_KIND: AGENT,
                INPUT_VALUE: _get_input_value(wrapped, *args, **kwargs),
                **dict(_smolagent_run_attributes(instance, arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            result = wrapped(*args, **kwargs)
            span.set_attribute(OUTPUT_VALUE, str(result))
            return result


class CustomToolCallWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        print("✅ CustomToolCallWrapper is ACTIVE")

        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        tool_name = getattr(instance, "name", instance.__class__.__name__)
        span_name = f"{instance.__class__.__name__}"
        with self._tracer.start_as_current_span(
            span_name,
            context=Context(),
            attributes={
                "openinference.tool.name": tool_name,
                SpanAttributes.OPENINFERENCE_SPAN_KIND: TOOL,
                INPUT_VALUE: _get_input_value(wrapped, *args, **kwargs),
                **dict(_tools(instance)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = wrapped(*args, **kwargs)
            span.set_attributes(
                dict(
                    _output_value_and_mime_type_for_tool_span(
                        response=response,
                        output_type=getattr(instance, "output_type", "string"),
                    )
                )
            )
            return response


class CustomSmolagentsInstrumentor:
    _already_instrumented = False

    def __init__(self) -> None:
        self._original_run_method = None
        self._original_step_methods = None
        self._original_model_call_methods = None
        self._original_tool_call_method = None

    def instrument(self, tracer_provider=None, config: Optional[TraceConfig] = None) -> None:
        if CustomSmolagentsInstrumentor._already_instrumented:
            return
        CustomSmolagentsInstrumentor._already_instrumented = True

        if tracer_provider is None:
            tracer_provider = trace_api.get_tracer_provider()
        if config is None:
            config = TraceConfig()

        self._tracer = OITracer(
            tracer_provider.get_tracer(__name__, "custom"),
            config=config,
        )

        run_wrapper = CustomRunWrapper(tracer=self._tracer)
        step_wrapper = _StepWrapper(tracer=self._tracer)
        tool_wrapper = CustomToolCallWrapper(tracer=self._tracer)

        # List of modules to support (original + test)
        ### Very important to add the modules here, otherwise the traces will not be grouped together
        instrumented_modules = ["smolagents", "src.base_agent"]

        # === Instrument run() on CodeAgent and ToolCallingAgent ===
        for mod in instrumented_modules:
            for cls_name in ["ToolCallingAgent", "CodeAgent"]:  # <-- add CodeAgent
                wrap_function_wrapper(mod, f"{cls_name}.run", run_wrapper)

        # === Instrument step() on CodeAgent and ToolCallingAgent ===
        for mod in instrumented_modules:
            for cls_name in ["CodeAgent", "ToolCallingAgent"]:
                wrap_function_wrapper(mod, f"{cls_name}.step", step_wrapper)

        # === Instrument model.__call__() ===
        self._original_model_call_methods = {}
        exported_model_subclasses = [
            attr for _, attr in vars(smolagents).items()
            if isinstance(attr, type) and issubclass(attr, models.Model)
        ]
        for model_subclass in exported_model_subclasses:
            model_wrapper = _ModelWrapper(tracer=self._tracer)
            self._original_model_call_methods[model_subclass] = getattr(model_subclass, "__call__")
            wrap_function_wrapper("smolagents", model_subclass.__name__ + ".__call__", model_wrapper)

        # === Instrument Tool.__call__() ===
        wrap_function_wrapper("smolagents", "Tool.__call__", tool_wrapper)
