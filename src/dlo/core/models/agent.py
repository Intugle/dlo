import logging

from dataclasses import dataclass, field
from enum import auto
from pathlib import Path
from typing import Any, Optional

# from deepagents.middleware.filesystem import FilesystemPermission
from dlo.common.exception import errors
from dlo.common.schema import EnumBase, SchemaMixin
from dlo.core.config import Project
from dlo.core.constants import AGENT_FILE_NAME, DEFAULT_AGENT_RECURSION_LIMIT, TARGET_DIR
from dlo.core.models.resources import BaseResource, ResourceTypes

# Configure module logger
log = logging.getLogger(__name__)


class AgentMode(EnumBase):
    primary = auto()
    subagent = auto()


class AgentType(EnumBase):
    deepagent = auto()
    standard = auto()


class Hitl(EnumBase):
    allow = auto()
    deny = auto()
    edit = auto()


@dataclass
class ToolConfig():
    name: str
    hitl: Optional[bool | list[Hitl]] = field(default=None)
    call_limit: Optional[int] = field(default=None)


@dataclass(kw_only=True)
class Agent(BaseResource):
    prompt: str = field()

    primary_model: str = field(metadata={"alias": "model"})
    fallback_models: list[str] = field(default_factory=list)
    agent_type: Optional[AgentType] = field(default=None)

    mode: AgentMode = field(default=AgentMode.primary)

    temperature: Optional[float] = field(default=None)
    reasoning_effort: Optional[str] = field(default=None)

    filesystem_permissions: list[Any] = field(default_factory=list)
    subagents: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tools: list[str | ToolConfig] = field(default_factory=list)
    mcp: list[str] = field(default_factory=list)

    session_memory: bool = field(default=True)
    memory: Optional[str] = field(default=None)
    compaction: Optional[str] = field(default=None)
    guardrails: Optional[list[str]] = field(default=None)
    features: Optional[list[str]] = field(default=None)

    input_: Optional[str] = field(default=None, metadata={"alias": "input"})
    output_: Optional[str] = field(default=None, metadata={"alias": "output"})

    recursion_limit: int = field(default=DEFAULT_AGENT_RECURSION_LIMIT)

    base_dir: Optional[str] = field(default=None)

    resource_type: ResourceTypes = field(default=ResourceTypes.agent)

    def __post_init__(self):
        if self.base_dir and self.skills:
            self.skills = [(Path(self.base_dir) / Path(s)).as_posix() + "/" for s in self.skills]
        if self.agent_type is None:
            if self.subagents or self.filesystem_permissions:
                self.agent_type = AgentType.deepagent
            else:
                self.agent_type = AgentType.standard

    class Config(BaseResource.Config):
        aliases = {
            "input_": "input",  # write as "input"
            "output_": "output",  # write as "output"
        }
        allow_deserialization_not_by_alias = True

    @property
    def normalized_tools(self) -> list[ToolConfig]:
        return [
            tool if isinstance(tool, ToolConfig) else ToolConfig(name=tool)
            for tool in self.tools
        ]


@dataclass(kw_only=True)
class ToolMeta(BaseResource):
    prompt: str = field()
    resource_type: ResourceTypes = field(default=ResourceTypes.tool_meta)


@dataclass(kw_only=True)
class LLMTask(BaseResource):
    prompt: str = field()

    primary_model: str = field(metadata={"alias": "model"})
    fallback_model: list[str] = field(default_factory=list)

    temperature: Optional[float] = field(default=None)
    reasoning_effort: Optional[str] = field(default=None)

    input_: Optional[str] = field(default=None, metadata={"alias": "input"})
    output_: Optional[dict] = field(default=None, metadata={"alias": "output"})

    resource_type: ResourceTypes = field(default=ResourceTypes.llm_task)

    class Config(BaseResource.Config):
        aliases = {
            "input_": "input",  # write as "input"
            "output_": "output",  # write as "output"
        }
        allow_deserialization_not_by_alias = True


@dataclass
class AgentManifest(SchemaMixin):
    root_dir: Path
    base_dir: str
    agents: dict[str, Agent] = field(default_factory=dict)
    tools_meta: dict[str, ToolMeta] = field(default_factory=dict)
    llm_tasks: dict[str, LLMTask] = field(default_factory=dict)

    @classmethod
    def __from_project__(cls, project: Project):
        candidates = [
            (project.project_root_path / ".dlo", "/.dlo"),
            (project.project_root_path / ".opencode", "/.opencode"),
            (project.project_root_path / ".claude", "/.claude"),
        ]

        for path, base_dir in candidates:
            if path.exists():
                log.debug(f"Agent manifest loaded from directory: {path}")
                return cls(root_dir=path, base_dir=base_dir)

        raise errors.DloConfigError(
            "Agents configuration not found.\n"
            "Create .dlo or .opencode or .claude directory and add agents"
        )

    def save(self, project: Project):
        target_path = project.project_root_path / TARGET_DIR
        target_path.mkdir(parents=True, exist_ok=True)

        manifest_path = target_path / AGENT_FILE_NAME

        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# =========================
# Agent Resource Factory
# =========================

@dataclass
class AgentResourceConfig:
    """Configuration for an agent resource type."""
    resource_type: ResourceTypes
    model_class: type[BaseResource]
    manifest_key: str  # "agents" | "llm_tasks" | "tools_meta"
    directory_name: str  # "agents" | "llm_tasks" |"tools"


AGENT_RESOURCE_REGISTRY: list[AgentResourceConfig] = [
    AgentResourceConfig(
        resource_type=ResourceTypes.agent,
        model_class=Agent,
        manifest_key="agents",
        directory_name="agents",
    ),
    AgentResourceConfig(
        resource_type=ResourceTypes.llm_task,
        model_class=LLMTask,
        manifest_key="llm_tasks",
        directory_name="llm_tasks",
    ),
    AgentResourceConfig(
        resource_type=ResourceTypes.tool_meta,
        model_class=ToolMeta,
        manifest_key="tools_meta",
        directory_name="tools",
    ),
]


class AgentResource:
    """Helper for looking up agent resource configurations."""

    # Build lookup dict from registry
    _registry_map = {cfg.resource_type: cfg for cfg in AGENT_RESOURCE_REGISTRY}

    @classmethod
    def get_resource(cls, resource_type: str | ResourceTypes) -> dict | None:
        """Get resource config by type. Returns dict for backward compatibility."""
        cfg = cls._registry_map.get(resource_type)
        if cfg:
            return {"model": cfg.model_class, "key": cfg.manifest_key}
        return None

    @classmethod
    def get_config(cls, resource_type: str | ResourceTypes) -> AgentResourceConfig | None:
        """Get full resource configuration by type."""
        return cls._registry_map.get(resource_type)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of available resource type names for error messages."""
        return [rt.value for rt in cls._registry_map.keys()]
