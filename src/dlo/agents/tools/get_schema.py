from typing import Annotated, Any, List, Tuple

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime
from langgraph.types import Command

from dlo.agents.tool import Tool, ToolArgsSchema


class ArgsSchema(ToolArgsSchema):
    tables: Annotated[List[str], "List of tables whose schema is to be fetched"]
    question: Annotated[str, "Question that needs to be answered"]


class GetSchema(Tool):
    name: str = "get_schema_new_test"
    description: str = "Tool to fetch database table(s) schema (i.e DDL) along with some sample values for each table(s)."
    args_schema: type[ToolArgsSchema] = ArgsSchema
    yoyo = "yoyo"

    async def tool(
        self, runtime: ToolRuntime, **arguments: dict
    ) -> Command | ToolMessage | str | Tuple[str, Any]:
        self.yoyo = "gogo"

        return "This is get schema"
