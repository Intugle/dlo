from typing import Any, Awaitable, Callable, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langgraph.runtime import Runtime


class StateSchema(AgentState):
    custom: dict


class CustomAgentMiddleware(AgentMiddleware):
    def __init__(self):
        self._state: Optional[StateSchema] = None

    def set_state(self, state: StateSchema) -> None:
        self._state = state

    @property
    def state(self) -> Optional[StateSchema]:
        return self._state

    # def find_graph_interrupt(self, exc: BaseException) -> Optional[GraphInterrupt]:
    #     if isinstance(exc, GraphInterrupt):
    #         return exc
    #
    #     if isinstance(exc, ExceptionGroup):
    #         for e in exc.exceptions:
    #             gi = self.find_graph_interrupt(e)
    #             if gi is not None:
    #                 return gi
    #
    #     return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        print("In wrap_model_call")
        breakpoint()
        return handler(request)

    async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        print("In awrap_model_call")
        breakpoint()
        return await handler(request)

    def before_agent(
            self,
            state: StateSchema,
            runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        print("In before_agent")
        breakpoint()
        print(messages)
        return state

    async def abefore_agent(
            self,
            state: StateSchema,
            runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        print("In abefore_agent")
        # Delegate to sync implementation
        return self.before_agent(state, runtime)

    def after_model(
        self,
        state: StateSchema,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        print("In after_model")
        breakpoint()
        return state

    async def aafter_model(
            self,
            state: StateSchema,
            runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        print("In aafter_model")
        # Delegate to sync implementation
        return self.after_model(state, runtime)

    def after_agent(
            self,
            state: StateSchema,
            runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        print("In after_agent")
        breakpoint()
        return state

    async def aafter_agent(
            self,
            state: StateSchema,
            runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        print("In aafter_agent")
        # Delegate to sync implementation
        return self.after_agent(state, runtime)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ):
        print("in wrap_tool_call")
        breakpoint()
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ):
        print("in awrap_tool_call")
        breakpoint()
        return await handler(request)
