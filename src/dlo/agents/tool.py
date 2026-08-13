import importlib
import importlib.util
import inspect
import logging

from abc import ABC
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel

from dlo.common.exception import errors
from dlo.core.models.agent import ToolMeta

log = logging.getLogger("__name__")


def _is_async(fn):
    if inspect.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)
    return call is not None and inspect.iscoroutinefunction(call)


class ToolArgsSchema(BaseModel):
    thinking: Annotated[
        str,
        "Reason for choosing the tool in concise form, Sacrifice grammar for the sake of concision",
    ]


class Tool(ABC):
    name: Annotated[str, "name of the tool"]
    description: Annotated[str, "description of the tool"]
    args_schema: Annotated[
        Optional[type[ToolArgsSchema]], "Input schema for the tool"
    ] = None

    @staticmethod
    async def tool(
        self,
        runtime: ToolRuntime,
        **arguments: dict,
    ) -> Command | ToolMessage | Any: ...


def register_tool(name: str):
    """Decorator to mark a callable as a registrable tool.

    Stamps `_tool_registry_name` on the object for later collection.
    No global side effects — purely metadata.

    Usage:
        @register_tool("execute_query")
        @tool(args_schema=ExecuteQueryArgs)
        def execute_query(query: str) -> QueryResult:
            ...
    """
    def wrapper(fn: Callable) -> Callable:
        fn._tool_registry_name = name
        return fn

    return wrapper


class ToolRegistry:
    """Instance-based tool registry.

    No shared global state. Tools are discovered by scanning modules
    for callables with `_tool_registry_name` attribute.

    Usage:
        registry = ToolRegistry()
        registry.discover_and_register("dlo.agents.tools")       # built-ins
        registry.discover_and_register_from_dir("/user/tools")   # user-specific
        tool = registry.get("execute_query")
    """

    def __init__(self, tools_meta: dict[str, ToolMeta]):
        self.tools: dict[str, Callable] = {}
        self.tools_meta: dict[str, ToolMeta] = tools_meta

    def register(self, name: str, tool_fn: Callable) -> None:
        """Manually register a tool."""
        self.tools[name] = tool_fn
        log.info(f"Tool registered: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self.tools.pop(name, None)

    def get(self, name: str) -> Callable:
        """Get a tool by name."""
        try:
            return self.tools[name]
        except KeyError:
            raise errors.MethodNotFoundError(f"unknown tools type {name!r}")

    def get_structured_tool(self, name: str) -> StructuredTool:
        tool = self.get(name)
        try:
            description = None
            args_schema = None

            # For the class based tool
            if isinstance(tool, type) and issubclass(tool, Tool):
                tool_instance: Tool = tool()
                name = tool_instance.name
                description = tool_instance.description
                args_schema = tool_instance.args_schema
                tool = tool_instance.tool

            # Override metadata from the tools_meta
            if tool_meta := self.tools_meta.get(name):
                name = tool_meta.name
                description = tool_meta.description

            # Detect sync vs async and pass to the right parameter
            if _is_async(tool):
                return StructuredTool.from_function(
                    coroutine=tool,
                    name=name,
                    description=description,
                    args_schema=args_schema,
                )
            else:
                return StructuredTool.from_function(
                    func=tool,
                    name=name,
                    description=description,
                    args_schema=args_schema,
                )
        except Exception as e:
            raise errors.DloRuntimeError(f"Error while creating tool `{name}`: {e}")

    def discover_and_register(self, package_path: str):
        """Auto-discover tool modules from an installed package.

        Imports all submodules, scans each for @register_tool decorated callables,
        registers them into this instance's tools dict.
        """
        import pkgutil

        try:
            package = importlib.import_module(package_path)
            package_walk = pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}.")
            for module_info in package_walk:
                module_path = module_info.name
                try:
                    module = importlib.import_module(module_path)
                except Exception:
                    log.exception(f"Failed to import tool module `{module_info.name}`")
                    continue

                self._collect_from_module(module)

                log.info(f"Tools scanned from package `{module_path}`")

        except ImportError:
            log.warning(f"Could not scan package '{package_path}'.")

    def discover_and_register_from_dir(self, package_path: str | Path):
        """Discover tools from a filesystem directory (tenant's project/tools/).

        Loads each .py file, scans for @register_tool decorated callables.
        """
        package_dir = Path(package_path)

        try:
            for file in package_dir.rglob("*.py"):
                if file.name == "__init__.py":
                    continue
                module_name = file.stem

                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec is None or spec.loader is None:
                    log.warning(f"Could not create module spec for `{file}`")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                self._collect_from_module(module)

                log.info(f"Tools scanned from file `{file}`")
        except Exception as e:
            log.warning(f"Could not scan directory '{package_dir}': {e}")

    def __collect_from_module(self, module):
        """Scan module for callables with `_tool_registry_name` attribute."""
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not callable(obj):
                continue
            name = getattr(obj, "_tool_registry_name", None)
            if name is not None:
                self.tools[name] = obj

    def _collect_from_module(self, module):
        """Scan module for callables with `_tool_registry_name` attribute."""
        for attr_name in dir(module):
            obj = getattr(module, attr_name)

            # Class-based
            # Collect any ToolBase subclasses defined in this module
            if (
                isinstance(obj, type)
                and issubclass(obj, Tool)
                and obj is not Tool
                and obj.name is not None
            ):
                self.tools[obj.name] = obj

            # Function-based
            # Support simple decorated functions
            elif callable(obj):
                name = getattr(obj, "_tool_registry_name", None)
                if name is not None:
                    self.tools[name] = obj
