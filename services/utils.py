from datetime import datetime

def normalize_datetime(value):
    if value is None:
        return datetime.now()

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        return datetime.fromisoformat(value)

    return datetime.now()