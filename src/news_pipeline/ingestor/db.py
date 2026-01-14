from __future__ import annotations

import psycopg
from typing import Any, Optional
from psycopg.types.json import Json


class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.Connection] = None

    def connect(self) -> None:
        if self._conn is None:
            self._conn = psycopg.connect(self._dsn, autocommit=False)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass  # Ignore errors when closing a broken connection
            finally:
                self._conn = None

    def is_connected(self) -> bool:
        """Check if connection exists and is still valid."""
        if self._conn is None:
            return False
        try:
            # Try to get connection status without executing a query
            # Check if connection is closed or broken
            if self._conn.closed:
                return False
            # Try a simple check - if socket is None, connection is lost
            try:
                socket = self._conn.pgconn.socket
                return socket is not None
            except Exception:
                return False
        except Exception:
            return False

    def reconnect(self) -> None:
        """Close existing connection and reconnect."""
        self.close()
        self.connect()

    def _ensure_connected(self) -> None:
        """Ensure connection is valid, reconnect if needed."""
        if not self.is_connected():
            self.reconnect()

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        assert self._conn is not None, "Not connected"
        with self._conn.cursor() as cur:
            adapted = self._adapt_params(params)
            cur.execute(sql, adapted)

    def execute_update(self, sql: str, params: tuple[Any, ...] | None = None) -> int:
        """Execute an UPDATE/INSERT/DELETE statement and return the number of rows affected."""
        assert self._conn is not None, "Not connected"
        with self._conn.cursor() as cur:
            adapted = self._adapt_params(params)
            cur.execute(sql, adapted)
            return cur.rowcount

    def query_one(self, sql: str, params: tuple[Any, ...] | None = None) -> Any:
        assert self._conn is not None, "Not connected"
        with self._conn.cursor() as cur:
            adapted = self._adapt_params(params)
            cur.execute(sql, adapted)
            return cur.fetchone()

    def query_all(self, sql: str, params: tuple[Any, ...] | None = None) -> list[Any]:
        """Fetch all rows for a query."""
        assert self._conn is not None, "Not connected"
        with self._conn.cursor() as cur:
            adapted = self._adapt_params(params)
            cur.execute(sql, adapted)
            return cur.fetchall()

    def cursor(self):
        assert self._conn is not None, "Not connected"
        return self._conn.cursor()

    def commit(self) -> None:
        """Commit transaction, handling connection errors."""
        if not self.is_connected():
            raise psycopg.OperationalError("Cannot commit: connection is lost")
        self._conn.commit()

    def rollback(self) -> None:
        """Rollback transaction, handling connection errors gracefully."""
        if self._conn is None:
            return
        try:
            if self.is_connected():
                self._conn.rollback()
            else:
                # Connection is lost, mark as None so it can be reconnected
                self._conn = None
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            # Connection lost or broken, mark as None
            self._conn = None
        except Exception:
            # Other errors - connection might be in bad state, mark as None
            self._conn = None

    def _adapt_params(self, params: tuple[Any, ...] | None) -> tuple[Any, ...]:
        if not params:
            return ()
        # Wrap dicts as JSON for psycopg adaptation; leave other types unchanged
        return tuple(Json(p) if isinstance(p, dict) else p for p in params)
