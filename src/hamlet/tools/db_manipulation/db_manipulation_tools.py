import json
import os
import re
import sqlite3
from typing import Any, Sequence

from src.hamlet.core.tools import Tool


class SQLiteQueryTool(Tool):
	name = "database_query"
	description = (
		"Execute a SELECT query against a known SQLite database and return the rows as a JSON-encoded list of dictionaries or tuples. "
		"Rejects non-SELECT statements and access to tables outside the allowlist."
	)
	inputs = {
		"sql_query": {
			"type": "string",
			"description": "The SQL statement to execute against the configured SQLite database.",
		}
	}
	output_type = "string"

	def __init__(
		self,
		database_path: str,
		allowed_tables: Sequence[str] | None = None,
		max_result_rows: int | None = None,
		result_format: str = "dict",
	) -> None:
		super().__init__()
		self.database_path = os.path.abspath(database_path)
		self.allowed_tables = {table.lower() for table in allowed_tables} if allowed_tables else None
		self.max_result_rows: int | None = max_result_rows
		format_normalized = result_format.lower()
		if format_normalized not in {"dict", "tuple"}:
			raise ValueError("result_format must be either 'dict' or 'tuple'.")
		self.result_format = format_normalized
		self._connection: sqlite3.Connection | None = None

	def setup(self) -> None:
		if not os.path.exists(self.database_path):
			raise FileNotFoundError(f"Database file not found: {self.database_path}")
		self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
		self._connection.row_factory = sqlite3.Row
		self.is_initialized = True

	def forward(self, sql_query: str) -> Any:
		if not sql_query or not sql_query.strip():
			return "Query rejected: SQL statement is empty."

		try:
			self._validate_query(sql_query)
		except ValueError as exc:
			return f"Query rejected: {exc}"

		assert self._connection is not None, "SQLite connection expected to be initialized."

		cursor: sqlite3.Cursor | None = None
		try:
			cursor = self._connection.cursor()
			cursor.execute(sql_query)
			if cursor.description:
				if self.max_result_rows is None:
					rows = cursor.fetchall()
				elif self.max_result_rows <= 0:
					rows = []
				else:
					rows = cursor.fetchmany(self.max_result_rows)
				if self.result_format == "dict":
					result = [dict(row) for row in rows]
				else:
					result = [tuple(row) for row in rows]
				return json.dumps(result, ensure_ascii=False)

			return "Query rejected: statement did not produce a result set."
		except sqlite3.Error as error:
			return f"SQLite error: {error}"
		finally:
			if cursor is not None:
				cursor.close()

	def _validate_query(self, sql_query: str) -> None:
		lowered = sql_query.lower()
		leading_token = re.match(r"\s*([a-zA-Z]+)", sql_query)
		opener = leading_token.group(1).lower() if leading_token else None
		if opener not in {"select", "with"}:
			raise ValueError("Only SELECT statements are supported.")
		cte_names: set[str] = set()
		if opener == "with":
			if not re.search(r"\)\s*select", lowered):
				raise ValueError("Common table expressions must end with a SELECT query.")
			cte_names = {
				match.lower()
				for match in re.findall(r"\b([a-zA-Z_][\w]*)\s+as\s*\(", sql_query, flags=re.IGNORECASE)
			}

		if self.allowed_tables is None:
			return

		referenced_tables = self._extract_table_names(lowered)
		unauthorized = referenced_tables - (self.allowed_tables | cte_names)
		if unauthorized:
			raise ValueError(
				"Unauthorized table access attempted: " + ", ".join(sorted(unauthorized))
			)

	@staticmethod
	def _extract_table_names(query: str) -> set[str]:
		keywords = ["from", "join", "update", "into", "table"]
		pattern = r"\b(?:" + "|".join(keywords) + r")\s+([a-zA-Z_][\w]*)"
		matches = re.findall(pattern, query)
		return {match.lower() for match in matches}

	def close(self) -> None:
		if self._connection is not None:
			self._connection.close()
			self._connection = None

	def __del__(self) -> None:
		self.close()


# test cases for SQLiteQueryTool
if __name__ == "__main__":
	import json
	import tempfile

	def initialize_demo_db(db_path: str) -> None:
		connection = sqlite3.connect(db_path)
		try:
			cursor = connection.cursor()
			cursor.executescript(
				"""
				DROP TABLE IF EXISTS users;
				DROP TABLE IF EXISTS logs;
				CREATE TABLE users (
					id INTEGER PRIMARY KEY,
					name TEXT NOT NULL,
					email TEXT NOT NULL
				);
				CREATE TABLE logs (
					id INTEGER PRIMARY KEY,
					message TEXT NOT NULL
				);
				INSERT INTO users (name, email) VALUES
					('Alice', 'alice@example.com'),
					('Bob', 'bob@example.com'),
					('Carol', 'carol@example.com');
				INSERT INTO logs (message) VALUES
					('system start'),
					('user login');
				"""
			)
			connection.commit()
		finally:
			connection.close()

	def print_assert(label: str, condition: bool) -> None:
		status = "PASS" if condition else "FAIL"
		print(f"[{status}] {label}")

	with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
		database_path = tmp.name

	tool_dict = None
	tuple_tool = None
	zero_tool = None

	try:
		initialize_demo_db(database_path)

		tool_dict = SQLiteQueryTool(
			database_path=database_path,
			allowed_tables=["users"],
		)
		response = tool_dict(sql_query="SELECT id, name FROM users ORDER BY id;")
		decoded = json.loads(response)
		print_assert(
			"basic SELECT returns all rows",
			decoded
			== [
				{"id": 1, "name": "Alice"},
				{"id": 2, "name": "Bob"},
				{"id": 3, "name": "Carol"},
			],
		)

		tuple_tool = SQLiteQueryTool(
			database_path=database_path,
			allowed_tables=["users"],
			max_result_rows=2,
			result_format="tuple",
		)
		tuple_response = tuple_tool(sql_query="SELECT id, name FROM users ORDER BY id;")
		print_assert(
			"tuple format respects max_result_rows",
			json.loads(tuple_response) == [[1, "Alice"], [2, "Bob"]],
		)

		zero_tool = SQLiteQueryTool(
			database_path=database_path,
			allowed_tables=["users"],
			max_result_rows=0,
		)
		zero_response = json.loads(zero_tool(sql_query="SELECT id FROM users;"))
		print_assert("max_result_rows=0 returns empty list", zero_response == [])

		empty_query = tool_dict(sql_query="   ")
		print_assert("empty query rejected", "empty" in empty_query.lower())

		non_select = tool_dict(sql_query="UPDATE users SET name='Zed' WHERE id=1;")
		print_assert("non-SELECT statements rejected", "Only SELECT" in non_select)

		unauthorized = tool_dict(sql_query="SELECT * FROM logs;")
		print_assert("unauthorized table rejected", "Unauthorized" in unauthorized)

		cte_response = json.loads(
			tool_dict(
				sql_query="WITH subset AS (SELECT id, name FROM users WHERE id <= 2) SELECT * FROM subset ORDER BY id;"
			)
		)
		print_assert(
			"WITH SELECT succeeds",
			cte_response == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
		)

		invalid_cte = tool_dict(sql_query="WITH subset AS (SELECT 1)")
		print_assert("WITH without SELECT rejected", "must end with a SELECT" in invalid_cte)

		no_rows = json.loads(tool_dict(sql_query="SELECT * FROM users WHERE id > 100;"))
		print_assert("empty result set returns []", no_rows == [])

	finally:
		if tool_dict is not None:
			tool_dict.close()
		if tuple_tool is not None:
			tuple_tool.close()
		if zero_tool is not None:
			zero_tool.close()
		if os.path.exists(database_path):
			os.remove(database_path)
