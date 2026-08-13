"""
Agent
"""

import logging

from functools import cached_property
from typing import Callable, Optional

from copilotkit import CopilotKitMiddleware
from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from dlo.agents.llm import ChatModelFactory
from dlo.agents.tool import ToolRegistry
from dlo.common.exception import errors
from dlo.core.compiler.graph import Graph
from dlo.core.config import Profile, Project
from dlo.core.constants import COMPILED_GRAPH_FIG_PATH_AGENTS
from dlo.core.models.agent import Agent, AgentManifest, AgentMode, AgentType, ToolConfig

log = logging.getLogger("__name__")


def get_weather(location: str):
    """Get weather for a location"""
    return f"The weather in {location} is sunny."


class AgentBuilder:
    def __init__(
        self,
        project: Project,
        profile: Profile,
        agent: Agent,
        tool_registry: ToolRegistry,
        checkpointer,
        agent_manifest: AgentManifest,
        compiled_agents: Optional[dict[str, CompiledStateGraph]] = None
    ):
        self.project = project
        self.profile = profile
        self.agent = agent
        self.tool_registry = tool_registry
        self.compiled_agents = compiled_agents
        self.checkpointer = checkpointer
        self.agent_manifest = agent_manifest

    def get_model(self, model: str):
        model_provider, model = model.split("/")

        provider = self.profile.providers.get(model_provider)
        if provider is None:
            raise errors.DloParseError(f"Provider `{model_provider}` not found in the profile")

        config = {
            **provider.config,
            "provider": provider.provider,
            "model": model,
        }
        if self.agent.temperature is not None:
            config["temperature"] = self.agent.temperature
        if self.agent.reasoning_effort is not None:
            config["reasoning_effort"] = self.agent.reasoning_effort

        return ChatModelFactory.create(**config)

    @cached_property
    def model(self):
        model = self.get_model(self.agent.primary_model)
        if fallback_models := self.agent.fallback_models:
            compiled_fallback_models = []
            for fallback_model in fallback_models:
                fm = self.get_model(fallback_model)
                compiled_fallback_models.append(fm)

            model = model.with_fallbacks(compiled_fallback_models)
        return model

    def _normalize_tool(tool):
        return tool.name if isinstance(tool, ToolConfig) else tool

    @cached_property
    def tools(self):
        tools = self.agent.normalized_tools
        return [
            self.tool_registry.get_structured_tool(tool.name)
            for tool in tools
        ]

    async def create_agent(self):
        async def _create_deep_agent(agent: Agent):
            subagents = []
            for subagent in agent.subagents:
                subagent_manifest = self.agent_manifest.agents[subagent]
                subagents.append(
                    CompiledSubAgent(
                        name=subagent_manifest.name,
                        description=subagent_manifest.description,
                        runnable=self.compiled_agents[subagent],
                    )
                )

            return create_deep_agent(
                model=self.model,
                middleware=[CopilotKitMiddleware()],  # for frontend tools and context
                system_prompt=agent.prompt,
                tools=self.tools,
                checkpointer=self.checkpointer,
                subagents=subagents,
                # permissions=agent.permissions,
                backend=FilesystemBackend(
                    root_dir=self.project.project_root_path, virtual_mode=True
                ),
                skills=agent.skills,
            )

        async def _create_standard_agent(agent: Agent):
            custom_graph = create_agent(
                model=self.model,
                system_prompt=agent.prompt,
                tools=self.tools,
                checkpointer=self.checkpointer,
            )

            return custom_graph

        # Agent factory
        agent_map: dict[AgentMode, Callable] = {
            AgentType.deepagent: _create_deep_agent,
            AgentType.standard: _create_standard_agent,
            None: _create_standard_agent,
        }

        return await agent_map[self.agent.agent_type](self.agent)


class AgentCompiler:
    def __init__(
        self,
        project: Project,
        profile: Profile,
        agent_manifest: AgentManifest,
        checkpointer,
    ):
        self.agent_manifest = agent_manifest
        self.profile = profile
        self.project = project
        self.compiled_agents: dict[str, CompiledStateGraph] = {}
        self.checkpointer = checkpointer

        # Per-compiler tool registry — isolated
        self.tool_registry = ToolRegistry(tools_meta=self.agent_manifest.tools_meta)
        self.tool_registry.discover_and_register("dlo.agents.tools")
        self.register_users_tools()

    def register_users_tools(self):
        tools_dir = self.agent_manifest.root_dir / "tools"

        if tools_dir.exists():
            self.tool_registry.discover_and_register_from_dir(tools_dir)

    @cached_property
    def graph(self) -> Graph:
        graph = Graph()

        agents_name = list(self.agent_manifest.agents.keys())
        agents = self.agent_manifest.agents.values()

        for agent in agents:
            graph.add_node(agent.name)

        # Check for missing subagents
        missing = [
            (agent.name, subagent)
            for agent in agents
            for subagent in agent.subagents
            if subagent not in agents_name
        ]
        if missing:
            details = ", ".join(f"`{sub}` (in `{agent}`)" for agent, sub in missing)
            raise errors.DloParseError(f"Subagents not found: {details}")

        graph.add_edges_from(
            (subagent, agent.name) for agent in agents for subagent in agent.subagents
        )
        return graph

    def draw_layer(self) -> None:
        graph = self.graph
        figure_name = self.project.project_root_path / COMPILED_GRAPH_FIG_PATH_AGENTS

        graph.draw_layer(nodes=self.agent_manifest.agents, figure_name=figure_name)

    async def create_agent(self, agent: Agent):
        agent_builder = AgentBuilder(
            project=self.project,
            profile=self.profile,
            agent=agent,
            compiled_agents=self.compiled_agents,
            tool_registry=self.tool_registry,
            checkpointer=self.checkpointer,
            agent_manifest=self.agent_manifest,
        )
        return await agent_builder.create_agent()

    async def compile_agent(self, agent_name: str) -> None:
        agent = self.agent_manifest.agents[agent_name]

        compiled_agent = await self.create_agent(agent)

        self.compiled_agents[agent_name] = compiled_agent

    async def compile(self) -> None:
        self.draw_layer()

        for agent_name in self.graph.topoligical_sort:
            await self.compile_agent(agent_name)
