from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import uuid
from typing import Any, Dict, List, Union

app = FastAPI(title="Mock Responses API", version="1.0.0")


def extract_echo_text(req_json: Dict[str, Any]) -> str:
    """
    Extract text to echo back from either:
    - responses-style 'input' (string or array of content parts)
    - chat-style 'messages' (array of {role, content})
    If multiple inputs exist, joins them with newlines.
    """
    # Prefer 'input' per the Responses API
    if "input" in req_json:
        inp = req_json["input"]
        if isinstance(inp, str):
            return inp
        elif isinstance(inp, list):
            parts: List[str] = []
            for item in inp:
                if isinstance(item, dict):
                    # Could be a message-like dict: {"role":"user","content":[...]}
                    if "content" in item and isinstance(item["content"], list):
                        for c in item["content"]:
                            if isinstance(c, dict) and c.get("type") == "text":
                                parts.append(c.get("text", ""))
                            elif isinstance(c, str):
                                parts.append(c)
                    # Could be a direct content part: {"type":"text","text":"..."}
                    elif item.get("type") == "text":
                        parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join([p for p in parts if p is not None])
        else:
            return str(inp)

    # Fallback to chat-style 'messages'
    elif "messages" in req_json:
        messages = req_json["messages"]
        if isinstance(messages, list):
            # Find last user message, else last message
            user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            last = user_msgs[-1] if user_msgs else (messages[-1] if messages else None)
            if last:
                content = last.get("content")
                if isinstance(content, list):
                    parts: List[str] = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c.get("text", ""))
                        elif isinstance(c, str):
                            parts.append(c)
                    return "\n".join(parts)
                elif isinstance(content, str):
                    return content
        return ""
    else:
        # No recognized fields; echo the whole JSON
        return str(req_json)


def build_responses_payload(echo_text: str, model: str = "mock-model") -> Dict[str, Any]:
    """
    Construct a JSON payload that matches the OpenAI Responses API shape.
    This includes an 'output' array containing a 'message' with text content.
    """
    now = int(time.time())
    response_id = f"resp_{uuid.uuid4().hex}"
    output_text = "Echo... " + echo_text

    payload: Dict[str, Any] = {
        "id": response_id,
        "object": "responses",
        "created": now,
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": output_text
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": len(output_text.split()),
            "total_tokens": len(output_text.split())
        }
    }
    return payload


@app.post("/v1/responses")
async def responses(request: Request):
    """
    Mimics the OpenAI Responses API endpoint.
    Echoes either 'input' or the last 'user' message from 'messages'.
    """
    try:
        req_json: Dict[str, Any] = (await request.json()) if request.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        return JSONResponse(content={"error": f"Invalid JSON: {str(e)}"}, status_code=400)
    model = req_json.get("model", "mock-model")
    # stream is ignored in this mock
    _stream = bool(req_json.get("stream", False))

    echo_text = extract_echo_text(req_json)
    payload = build_responses_payload(echo_text, model=model)
    return JSONResponse(content=payload, status_code=200)


if __name__ == "__main__":
    # Run the mock server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

