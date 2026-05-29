"""Base protocol for data sources."""

from __future__ import annotations

from typing import Iterator, Protocol

from t20x.models.domain import ParsedMatch


class DataSource(Protocol):
    """Protocol for cricket data sources."""

    def parse(self, path: str | None = None) -> Iterator[ParsedMatch]:
        """Parse matches from this data source.

        Args:
            path: Path to local data file/directory. None = auto-download.

        Yields:
            ParsedMatch objects.
        """
        ...
