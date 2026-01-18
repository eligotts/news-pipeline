from __future__ import annotations

import time
import psycopg
import structlog
from typing import Any, Callable, Optional, TypeVar
from psycopg.types.json import Json

logger = structlog.get_logger()

T = TypeVar("T")

# Connection errors that indicate the connection is dead and should be retried
_RETRIABLE_ERRORS = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
)


class Database:
    def __init__(
        self,
        dsn: str,
        connect_timeout: int = 30,
        keepalives: bool = True,
        keepalives_idle: int = 60,
        keepalives_interval: int = 15,
        keepalives_count: int = 3,
    ):
        self._dsn = dsn
        self._connect_timeout = connect_timeout
        self._keepalives = keepalives
        self._keepalives_idle = keepalives_idle
        self._keepalives_interval = keepalives_interval
        self._keepalives_count = keepalives_count
        self._conn: Optional[psycopg.Connection] = None
        self._in_transaction = False

    def connect(self) -> None:
        if self._conn is None:
            conn_params = {
                "conninfo": self._dsn,
                "autocommit": False,
                "connect_timeout": self._connect_timeout,
            }

            # Add keepalive parameters to DSN if enabled
            if self._keepalives:
                # Append keepalive params to DSN
                separator = "&" if "?" in self._dsn else "?"
                keepalive_dsn = (
                    f"{self._dsn}{separator}"
                    f"keepalives=1&"
                    f"keepalives_idle={self._keepalives_idle}&"
                    f"keepalives_interval={self._keepalives_interval}&"
                    f"keepalives_count={self._keepalives_count}"
                )
                conn_params["conninfo"] = keepalive_dsn

            self._conn = psycopg.connect(**conn_params)
            self._in_transaction = False
            logger.debug("database_connected", keepalives=self._keepalives)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass  # Ignore errors when closing a broken connection
            finally:
                self._conn = None
                self._in_transaction = False

    def is_connected(self) -> bool:
        """Check if connection exists and is still valid."""
        if self._conn is None:
            return False
        try:
            # Check if connection is closed or broken
            if self._conn.closed:
                return False
            # Try a simple check - if socket is None, connection is lost
            try:
                socket = self._conn.pgconn.socket
                return socket is not None and socket >= 0
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

    def _can_auto_reconnect(self) -> bool:
        """Check if it's safe to auto-reconnect (not in the middle of a transaction)."""
        return not self._in_transaction

    def _execute_with_retry(
        self,
        operation: Callable[[], T],
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> T:
        """Execute an operation with automatic reconnection on transient errors.

        Only retries if we're not in the middle of a transaction (where reconnecting
        would lose transaction state).
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                self._ensure_connected()
                return operation()
            except _RETRIABLE_ERRORS as e:
                last_error = e

                # If we're in a transaction, we can't safely reconnect
                if not self._can_auto_reconnect():
                    logger.warning(
                        "db_connection_error_in_transaction",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    raise

                # If this was our last attempt, raise
                if attempt >= max_retries:
                    logger.error(
                        "db_connection_error_max_retries",
                        error=str(e),
                        attempts=attempt + 1,
                    )
                    raise

                # Log and retry
                logger.warning(
                    "db_connection_error_retrying",
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )

                # Mark connection as dead and sleep before retry
                self._conn = None
                self._in_transaction = False
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        # Should never reach here, but just in case
        raise last_error or RuntimeError("Unexpected retry loop exit")

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        def _do_execute():
            if self._conn is None:
                raise psycopg.InterfaceError("Not connected")
            with self._conn.cursor() as cur:
                adapted = self._adapt_params(params)
                cur.execute(sql, adapted)
            self._in_transaction = True

        self._execute_with_retry(_do_execute)

    def execute_update(self, sql: str, params: tuple[Any, ...] | None = None) -> int:
        """Execute an UPDATE/INSERT/DELETE statement and return the number of rows affected."""
        def _do_execute() -> int:
            if self._conn is None:
                raise psycopg.InterfaceError("Not connected")
            with self._conn.cursor() as cur:
                adapted = self._adapt_params(params)
                cur.execute(sql, adapted)
                self._in_transaction = True
                return cur.rowcount

        return self._execute_with_retry(_do_execute)

    def query_one(self, sql: str, params: tuple[Any, ...] | None = None) -> Any:
        def _do_query():
            if self._conn is None:
                raise psycopg.InterfaceError("Not connected")
            with self._conn.cursor() as cur:
                adapted = self._adapt_params(params)
                cur.execute(sql, adapted)
                self._in_transaction = True
                return cur.fetchone()

        return self._execute_with_retry(_do_query)

    def query_all(self, sql: str, params: tuple[Any, ...] | None = None) -> list[Any]:
        """Fetch all rows for a query."""
        def _do_query() -> list[Any]:
            if self._conn is None:
                raise psycopg.InterfaceError("Not connected")
            with self._conn.cursor() as cur:
                adapted = self._adapt_params(params)
                cur.execute(sql, adapted)
                self._in_transaction = True
                return cur.fetchall()

        return self._execute_with_retry(_do_query)

    def cursor(self):
        self._ensure_connected()
        if self._conn is None:
            raise psycopg.InterfaceError("Not connected")
        self._in_transaction = True
        return self._conn.cursor()

    def commit(self) -> None:
        """Commit transaction, handling connection errors."""
        if not self.is_connected():
            raise psycopg.OperationalError("Cannot commit: connection is lost")
        self._conn.commit()  # type: ignore[union-attr]
        self._in_transaction = False

    def rollback(self) -> None:
        """Rollback transaction, handling connection errors gracefully."""
        if self._conn is None:
            self._in_transaction = False
            return
        try:
            if self.is_connected():
                self._conn.rollback()
            else:
                # Connection is lost, mark as None so it can be reconnected
                self._conn = None
        except (psycopg.OperationalError, psycopg.InterfaceError):
            # Connection lost or broken, mark as None
            self._conn = None
        except Exception:
            # Other errors - connection might be in bad state, mark as None
            self._conn = None
        finally:
            self._in_transaction = False

    def _adapt_params(self, params: tuple[Any, ...] | None) -> tuple[Any, ...]:
        if not params:
            return ()
        # Wrap dicts as JSON for psycopg adaptation; leave other types unchanged
        return tuple(Json(p) if isinstance(p, dict) else p for p in params)
