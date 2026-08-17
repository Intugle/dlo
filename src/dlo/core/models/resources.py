import logging

from dataclasses import dataclass, field
from enum import auto
from pathlib import Path
from typing import Optional

import aiofiles.os

from quartz_cron_checker import QuartzCronChecker

from dlo.common.exception import errors
from dlo.common.schema import EnumBase, SchemaMixin

log = logging.getLogger(__name__)

# =========================
# Enums
# =========================


class NodeResourceTypes(EnumBase):
    source = auto()
    model = auto()


class ResourceTypes(EnumBase):
    relationship = auto()
    metric = auto()
    source = NodeResourceTypes.source
    model = NodeResourceTypes.model
    code = auto()
    chart = auto()
    dashboard = auto()
    agent = auto()
    tool_meta = auto()
    llm_task = auto()


class ColumnCategory(EnumBase):
    dimension = auto()
    measure = auto()


class StorageType(EnumBase):
    table = auto()
    csv = auto()


class ModelType(EnumBase):
    materialized = auto()
    view = auto()
    ephemeral = auto()
    # incremental = auto()


class CodeType(EnumBase):
    sql = auto()
    # python = auto()


class ChartDataSource(EnumBase):
    sql = auto()
    model = auto()
    # python = auto()


# =========================
# Base Models
# =========================


@dataclass(kw_only=True)
class MetaMixin(SchemaMixin):
    created_on: Optional[str] = field(default=None)
    modified_on: Optional[str] = field(default=None)
    created_by: Optional[str] = field(default=None)
    modified_by: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class BaseResource(MetaMixin, SchemaMixin):
    """Base class for all DLO resources (models, sources, charts, etc).

    Provides common fields like name, file path, description, and metadata
    tracking for resource lineage and documentation.
    """

    name: str
    file_path: Path
    resource_type: ResourceTypes
    description: str = field(default="")
    tags: Optional[list[str]] = field(default=None)
    unique_id: Optional[str] = field(default=None)

    # HACK: Having unique_id same as name. It may change for future requirements
    def __post_init__(self):
        if self.unique_id is None:
            self.unique_id = self.name

    def remove(self):
        self.file_path.unlink(missing_ok=True)

    async def aremove(self):
        try:
            # Asynchronously removes the file
            await aiofiles.os.remove(self.file_path)
            log.debug(f"Successfully deleted {self.file_path}")
        except FileNotFoundError:
            log.error(f"Error: {self.file_path} does not exist.")
        except PermissionError:
            log.error(f"Error: Insufficient permissions to delete {self.file_path}.")


# =========================
# Model Mixins
# =========================


@dataclass
class DependsOn(SchemaMixin):
    nodes: list[str] = field(default_factory=list)

    def add_node(self, value: str):
        """Add a dependency node if not already present.

        Args:
            value: Node name/ID to add as dependency.
        """
        if value not in self.nodes:
            self.nodes.append(value)


@dataclass
class InjectedCTE(SchemaMixin):
    id: str
    sql: str


@dataclass
class CompiledResourceMixin(SchemaMixin):
    sources: list[str] = field(default_factory=list)
    compiled_path: Optional[Path] = field(default=None)
    compiled_code: Optional[str] = field(default=None)
    compiled: bool = field(default=False)
    depends_on: DependsOn = field(default_factory=DependsOn)
    # contains ctes for all dependents
    extra_ctes: list[InjectedCTE] = field(default_factory=list)


@dataclass
class ScheduledResourceMixin(SchemaMixin):
    schedule_depends_on: DependsOn = field(default_factory=DependsOn)


# =========================
# Shared Models
# =========================


@dataclass
class ProfilingMetrics(SchemaMixin):
    count: Optional[int] = field(default=None)
    null_count: Optional[int] = field(default=None)
    distinct_count: Optional[int] = field(default=None)


@dataclass
class Column(SchemaMixin):
    name: str
    type: str
    category: Optional[ColumnCategory] = field(default=None)
    description: Optional[str] = field(default=None)
    tags: Optional[list[str]] = field(default=None)
    profiling_metrics: Optional[ProfilingMetrics] = field(default=None)
    sample_data: Optional[list[str | int | float]] = field(default=None)


# =========================
# Source Models
# =========================


@dataclass
class SourceDetails(SchemaMixin):
    full_name: str
    type: StorageType = field(default=StorageType.table)


@dataclass(kw_only=True)
class Source(BaseResource):
    """Represents a data source (table, CSV, etc) in the warehouse.

    Sources are the raw data inputs that models transform. They include
    schema information, connection details, and constraints.
    """

    name: str
    details: SourceDetails
    columns: list[Column] = field(default_factory=list)
    resource_type: ResourceTypes = field(default=ResourceTypes.source)
    description: Optional[str] = field(default=None)
    tags: Optional[list[str]] = field(default=None)
    connection: Optional[str] = field(default=None)
    primary_key: Optional[list[str]] = field(default=None)
    unique_keys: Optional[list[list[str]]] = field(default=None)

    @property
    def relation_name(self):
        return self.details.full_name


# =========================
# Model (Semantic / Transform)
# =========================


@dataclass
class ModelDetails(SchemaMixin):
    full_name: Optional[str] = field(default=None)
    type: StorageType = field(default=StorageType.table)


@dataclass(kw_only=True)
class Model(BaseResource, CompiledResourceMixin, ScheduledResourceMixin):
    """Represents a data transformation model in DLO.

    Models are SQL-based transformations that create materialized tables,
    views, or ephemeral CTEs. They track dependencies, support scheduling,
    and can be compiled for execution.
    """

    name: str
    type: ModelType
    columns: list[Column] = field(default_factory=list)
    resource_type: ResourceTypes = field(default=ResourceTypes.model)
    description: Optional[str] = field(default=None)
    tags: Optional[list[str]] = field(default=None)
    details: Optional[ModelDetails] = field(default=None)
    schedule: Optional[str] = field(default=None)
    primary_key: Optional[list[str]] = field(default=None)
    unique_keys: Optional[list[list[str]]] = field(default=None)
    raw_code: Optional[str] = field(default=None)
    code_path: Optional[Path] = field(default=None)

    def __post_init__(self):
        super().__post_init__()

        is_ephemeral = self.type == ModelType.ephemeral

        if is_ephemeral:
            self.details = None
        elif self.details is None:
            raise errors.DloCompilationError(
                f"Details must be added with full name. For model type: {self.type} "
                f"Invalid Model: `{self.name}` file: `{self.file_path}`"
            )

        # Validate schedule cron
        if self.schedule is not None:
            if self.type != ModelType.materialized:
                raise errors.DloCompilationError(
                    f"Only materiazlied model can be scheduled. Invalid Model: `{self.name}` file: `{self.file_path}`"  # noqa: E501
                )
            err_msg = f"Invalid cron expression for Model: `{self.name}` file: `{self.file_path}`"
            try:
                cron_checker = QuartzCronChecker.from_cron_string(self.schedule)
                if not cron_checker.validate():
                    raise errors.DloCompilationError(err_msg)
            except Exception as e:
                raise errors.DloCompilationError(f"{err_msg}\nMessage: {e}")


# =========================
# Relationships
# =========================


@dataclass(kw_only=True)
class Relationship(BaseResource):
    """Defines a relationship between two models/sources.

    Relationships establish connections between resources based on column
    mappings, enabling semantic understanding of data lineage.
    """

    name: str
    from_: str = field(metadata={"alias": "from"})
    to: str
    from_columns: list[str]
    to_columns: list[str]
    resource_type: ResourceTypes = field(default=ResourceTypes.relationship)
    description: Optional[str] = field(default=None)

    class Config(BaseResource.Config):
        aliases = {
            "from_": "from",  # write as "from"
        }
        allow_deserialization_not_by_alias = True


# =========================
# Metrics
# =========================


@dataclass(kw_only=True)
class Metric(BaseResource):
    name: str
    expression: str
    resource_type: ResourceTypes = field(default=ResourceTypes.metric)
    description: Optional[str] = field(default=None)


# =========================
# Sql
# =========================


@dataclass(kw_only=True)
class Code(BaseResource):
    name: str
    code: str
    resource_type: ResourceTypes = field(default=ResourceTypes.code)
    type: CodeType = field(default=CodeType.sql)


# =========================
# Charts
# =========================
class ChartEngine(EnumBase):
    echarts = auto()
    custom = auto()


@dataclass(kw_only=True)
class Chart(BaseResource):
    """Represents a data visualization chart.

    Charts can pull data from SQL queries or models and support various
    visualization engines (ECharts, custom). Includes caching via freshness.
    """

    sql: Optional[str] = field(default=None)
    model: Optional[str] = field(default=None)
    resource_type: ResourceTypes = field(default=ResourceTypes.chart)
    data_source: ChartDataSource = field(default=ChartDataSource.sql)
    freshness: Optional[int] = field(default=None)
    engine: ChartEngine = ChartEngine.echarts
    option: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        requirements = {
            ChartDataSource.sql: "sql",
            ChartDataSource.model: "model",
        }

        for typ, field_name in requirements.items():
            if self.data_source == typ:
                # Dynamically fetch the value of property using the string name
                field_value = getattr(self, field_name)
                if not field_value:
                    raise errors.DloCompilationError(
                        f"'{field_name}' must be provided when data_source='{typ.value}'"
                    )


# =========================
# Dashboards
# =========================


# Referenced from react-grid-layout
# https://github.com/react-grid-layout/react-grid-layout/blob/master/src/core/types.ts
@dataclass
class LayoutItem(SchemaMixin):
    i: str = field(metadata={"description": "Unique identifier for an item"})
    x: int = field(metadata={"description": "X position in grid units"})
    y: int = field(metadata={"description": "Y position in grid units"})
    w: int = field(metadata={"description": "Width in grid units"})
    h: int = field(metadata={"description": "Height in grid units"})
    minW: Optional[int] = field(
        default=None,
        metadata={"description": "Minimum width in grid units"},
    )
    minH: Optional[int] = field(
        default=None,
        metadata={"description": "Minimum height in grid units"},
    )
    maxW: Optional[int] = field(
        default=None,
        metadata={"description": "Maximum width in grid units"},
    )
    maxH: Optional[int] = field(
        default=None,
        metadata={"description": "Maximum height in grid units"},
    )
    static: Optional[bool] = field(
        default=None,
        metadata={"description": "If true, item cannot be dragged or resized"},
    )
    isDraggable: Optional[bool] = field(
        default=None,
        metadata={"description": "If false, item cannot be dragged but may be resizable"},
    )
    isResizable: Optional[bool] = field(
        default=None,
        metadata={"description": "If false, item cannot be resized but may be draggable"},
    )


# Referenced from react-grid-layout
# https://github.com/react-grid-layout/react-grid-layout/blob/master/src/core/types.ts
@dataclass
class GridConfig(SchemaMixin):
    cols: Optional[int] = field(
        default=None,
        metadata={"description": "Number of columns in the grid"},
    )
    rowHeight: Optional[int] = field(
        default=None,
        metadata={"description": "Height of a single row in pixels"},
    )
    margin: Optional[tuple[int, int]] = field(
        default=None,
        metadata={"description": "[horizontal, vertical] margin between items in pixels"},
    )
    containerPadding: Optional[tuple[int, int]] = field(
        default=None,
        metadata={"description": "[horizontal, vertical] padding inside the container"},
    )
    maxRows: Optional[int] = field(
        default=None,
        metadata={"description": "Maximum number of rows"},
    )


@dataclass(kw_only=True)
class Dashboard(BaseResource):
    """Represents a dashboard containing multiple charts.

    Dashboards organize charts in a grid layout with configurable positioning
    and sizing based on react-grid-layout.
    """

    resource_type: ResourceTypes = field(default=ResourceTypes.dashboard)
    charts: dict[str, str] = field(default_factory=dict)
    layout: list[LayoutItem] = field(default_factory=list)
    grid_config: Optional[GridConfig] = field(default=None)


# =========================
# Resource Factory
# =========================


class Resource:
    """Factory for creating resource instances from configuration data.

    Maps resource type strings to their corresponding model classes and
    provides validation during resource creation.
    """

    model_factory = {
        "models": Model,
        "relationships": Relationship,
        "sources": Source,
        "metrics": Metric,
        "charts": Chart,
        "dashboards": Dashboard,
    }

    def create_resource(self, resource_type: str, data: dict):
        """Create a resource instance from type and data dict.

        Args:
            resource_type: Resource type string (e.g., 'models', 'sources').
            data: Dictionary containing resource configuration.

        Returns:
            Resource instance of the appropriate type.

        Raises:
            DloCompilationError: If resource type is unknown.
        """
        model_cls = self.model_factory.get(resource_type.lower())
        if not model_cls:
            raise errors.DloCompilationError(
                f"Resource model not found: {resource_type}\n"
                f"Available resource types: {self.model_factory.keys()}"
            )
        return model_cls(**data)

    @classmethod
    def get_resource(self, resource_type: str) -> BaseResource:
        return self.model_factory.get(resource_type)
