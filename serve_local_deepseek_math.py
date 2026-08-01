import argparse
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-math-7b-instruct"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = Field(default=1024, alias="max_tokens")
    top_p: float = 0.95
    stream: bool = False


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def clean_byte_level_text(text: str) -> str:
    try:
        raw = bytearray()
        for char in text:
            if char in BYTE_DECODER:
                raw.append(BYTE_DECODER[char])
            else:
                raw.extend(char.encode("utf-8"))
        return raw.decode("utf-8").strip()
    except UnicodeError:
        return text.replace("Ġ", " ").replace("Ċ", "\n").strip()


def build_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    plain_messages = [m.model_dump() for m in messages]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            plain_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_parts = []
    for message in messages:
        role = message.role.strip()
        content = message.content.strip()
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(f"User: {content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def create_app(model_path: str) -> FastAPI:
    app = FastAPI(title="Local DeepSeek Math OpenAI-compatible API")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print("Model loaded.")

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "deepseek-math-7b-instruct",
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise ValueError("Streaming is not implemented in this local server.")

        prompt = build_prompt(tokenizer, request.messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True)
        content = clean_byte_level_text(content)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].numel()),
                "completion_tokens": int(generated_ids.numel()),
                "total_tokens": int(inputs["input_ids"].numel() + generated_ids.numel()),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/deepseek-math-7b-instruct",
        help="Path to the local Hugging Face model directory.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    app = create_app(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
