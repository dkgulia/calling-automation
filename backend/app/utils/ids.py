import uuid


def generate_session_id() -> str:
    return f"sim_{uuid.uuid4().hex[:12]}"
