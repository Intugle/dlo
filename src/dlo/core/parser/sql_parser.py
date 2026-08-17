from functools import cached_property

import sqlglot


class SqlParser:
    """SQL parser for extracting dependencies and validating queries.

    Uses sqlglot to parse SQL and extract table references, validate
    SELECT statements, and identify CTEs.
    """

    def __init__(self, query: str):
        self.query = query

    @cached_property
    def expr(self):
        return sqlglot.parse_one(self.query)

    @property
    def is_only_select(self):
        """
        Checks if the given query string is exclusively a SELECT statement.

        Returns:
            bool: True if the query is a SELECT statement, False otherwise.
        """
        try:
            # Use parse_one to parse a single statement.
            # This will raise a ParseError if the SQL is invalid or contains multiple statements
            # unless configured otherwise.
            # Check if the main expression is of type Select
            return isinstance(self.expr, sqlglot.exp.Select)

        except sqlglot.errors.ParseError:
            # If parsing fails, it's not a valid single SQL statement,
            # or potentially an incomplete one.
            return False

    def extract_table(self) -> set:
        """
        Extracts all table names referenced in the SQL query, excluding CTEs.

        Returns:
            set: A set of fully qualified table names in the format 'catalog.db.table'
                 or partial names if catalog/db are not specified. CTE names are excluded.
        """
        # Get CTE names so we can exclude them
        ctes = self.expr.find_all(sqlglot.exp.CTE)
        cte_names = {cte.alias_or_name for cte in ctes}

        tables = self.expr.find_all(sqlglot.exp.Table)
        table_names = {
            ".".join(part for part in [t.catalog, t.db, t.name] if part)
            for t in tables
            if t.name not in cte_names
        }

        return table_names
