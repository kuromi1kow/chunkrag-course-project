from __future__ import annotations

import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="chunkrag-demo-key")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", default="Explain in two sentences what chunking does in RAG.")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    models = client.models.list()
    print("Available models:")
    for model in models.data[:10]:
        print(f"- {model.id}")

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.1,
        max_tokens=160,
    )
    print("\nResponse:\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
