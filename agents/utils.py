
# agents/utils.py
import json
from typing import Any, Dict, Optional

def _model_to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Try to convert pydantic/typed objects to plain dict."""
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if hasattr(obj, "dict"):
            return obj.dict()
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None

async def get_raw_request_json(context: Any) -> Dict[str, Any]:
    """Return raw JSON body if available (often {} because the framework consumes it)."""
    raw = None
    req = getattr(context, "request", None)
    if req is not None:
        if hasattr(req, "json"):
            try:
                raw = await req.json()
            except Exception:
                raw = None
        if raw is None and hasattr(req, "body"):
            try:
                b = await req.body()
                raw = json.loads(b.decode() if isinstance(b, (bytes, bytearray)) else b)
            except Exception:
                raw = None
    if not isinstance(raw, dict):
        raw = {}
    return raw

def context_attributes_snapshot(context: Any) -> str:
    """List available attributes on RequestContext (diagnostic)."""
    try:
        names = sorted([n for n in dir(context) if not n.startswith("_")])
        return ", ".join(names[:60])
    except Exception:
        return "<unavailable>"

def get_message_dict(context: Any) -> Dict[str, Any]:
    """Convert typed context.message to a plain dict."""
    msg_obj = getattr(context, "message", None)
    md = _model_to_dict(msg_obj)
    return md or {}

def extract_user_text_from_message(message: Dict[str, Any]) -> str:
    """
    Extract user query from message.parts where kind == 'text' and not a context marker.
    """
    parts = message.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if p.get("kind") == "text":
                txt = p.get("text", "")
                if isinstance(txt, str) and txt.strip() and not txt.strip().startswith("[[LOCAL_CONTEXT]]"):
                    return txt.strip()
    if isinstance(message.get("text"), str):
        return message["text"].strip()
    return ""

def extract_local_context_from_message(message: Dict[str, Any]) -> str:
    """
    Extract local context from part marked [[LOCAL_CONTEXT]].
    """
    parts = message.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if p.get("kind") == "text":
                txt = p.get("text", "")
                if isinstance(txt, str) and txt.strip().startswith("[[LOCAL_CONTEXT]]"):
                    return txt.strip().split("[[LOCAL_CONTEXT]]", 1)[-1].strip()
    return ""

async def safe_enqueue(event_queue, event_obj):
    """Avoid crashing when queue closes unexpectedly."""
    try:
        await event_queue.enqueue_event(event_obj)
    except Exception:
        pass

def mk_agent_skill(id_: str, name: str, description: str, tags=None, examples=None):
    from a2a.types import AgentSkill
    return AgentSkill(
        id=id_,
        name=name,
        description=description,
        tags=tags or [],
        examples=examples or [],
    )