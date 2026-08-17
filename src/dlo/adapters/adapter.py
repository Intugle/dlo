from abc import ABC, abstractmethod
from typing import Optional

from dlo.adapters.model import Node, NodeId, NodeMap, QueryResult
from dlo.common.exception import errors
from dlo.core.constants import DEFAULT_CURSOR_LIMIT


class Adapter(ABC):
    """Abstract base class for database adapters.

    Defines the interface for executing queries, creating tables, and managing
    scheduled jobs across different database platforms (Postgres, Databricks, etc).
    """

    def create_table(self, model: Node):
        """Create a table from a compiled model node.

        Args:
            model: The node representing the model/table to create.

        Returns:
            Result from the underlying database adapter.

        Raises:
            DloRuntimeError: If model hasn't been compiled.
        """
        if model.compiled is False:
            raise errors.DloRuntimeError(
                f"Model `{model.name}` was not complied but tried to create it"
            )

        return self._create_table(model)

    @abstractmethod
    def execute(self, query: str, cursor_limit: Optional[int] = DEFAULT_CURSOR_LIMIT):
        """Execute a SQL query without returning results.

        Args:
            query: SQL query string to execute.
            cursor_limit: Maximum number of rows to process.
        """
        ...

    @abstractmethod
    def query(self, query: str, cursor_limit: Optional[int] = DEFAULT_CURSOR_LIMIT) -> QueryResult:
        """Execute a SQL query and return results.

        Args:
            query: SQL query string to execute.
            cursor_limit: Maximum number of rows to return.

        Returns:
            QueryResult containing columns and rows.
        """
        ...

    @abstractmethod
    def _create_table(self, model: Node): ...

    @abstractmethod
    def create_job(
        self,
        node_map: NodeMap,
        nodes: list[NodeId],
        job_name: str,
        cron: Optional[str] = None,
        job_info: Optional[dict] = None,
    ) -> dict:
        """Create or update a scheduled job for running nodes.

        Args:
            node_map: Mapping of node IDs to node objects.
            nodes: List of node IDs to include in the job.
            job_name: Name for the scheduled job.
            cron: Optional cron expression for scheduling.
            job_info: Optional existing job information for updates.

        Returns:
            Job metadata dictionary.
        """
        ...

    @abstractmethod
    def pause_job(self, job_info: dict, cron: str) -> Optional[dict]:
        """Pause a scheduled job.

        Args:
            job_info: Job metadata dictionary.
            cron: Cron expression of the job to pause.

        Returns:
            Updated job metadata or None.
        """
        ...
