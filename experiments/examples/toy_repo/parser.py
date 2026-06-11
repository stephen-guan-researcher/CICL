from datetime import datetime, timezone


def parse_iso8601_timestamp(value: str) -> datetime:
    """Parse timestamps used by the ingestion pipeline.

    Local project rule: timestamps ending with Z must be interpreted as UTC.
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_user_id(value: str) -> str:
    return value.strip().lower().replace(" ", "-")

