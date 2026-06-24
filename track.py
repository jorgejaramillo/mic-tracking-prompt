#!/usr/bin/env python3
"""
Prompt tracker via DataForSEO LLM Responses API.
Docs: https://docs.dataforseo.com/v3/ai_optimization/chat_gpt/llm_responses/live/

Results saved to output/<domain>_<timestamp>.csv and .pdf

Usage:
    python track.py domain/vajillascorona-com-co.md
    python track.py domain/vajillascorona-com-co.md --prompt 1
"""

import os
import sys
import time
import csv
import re
import base64
import argparse
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: instala requests con:  pip install requests")
    sys.exit(1)

import html

DATAFORSEO_BASE = "https://api.dataforseo.com/v3/ai_optimization/{se}/llm_responses/live"

# Load .env if present
_env_file = Path(".env")
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

DFS_LOGIN = os.getenv("DATAFORSEO_LOGIN", "")
DFS_PASSWORD = os.getenv("DATAFORSEO_PASSWORD", "")

# Modelos con web_search activo — confirmados desde DataForSEO /models
MODELS = [
    # OpenAI / ChatGPT
    {"name": "GPT-4o",              "id": "gpt-4o",               "se": "chat_gpt",  "reasoning": False},
    {"name": "o4-mini",             "id": "o4-mini",              "se": "chat_gpt",  "reasoning": True},
    # Claude
    {"name": "Claude Sonnet 4.6",   "id": "claude-sonnet-4-6",    "se": "claude",    "reasoning": True},
    # Gemini
    {"name": "Gemini 3.1 Pro",      "id": "gemini-3.1-pro-preview","se": "gemini",   "reasoning": True},
    # Perplexity
    {"name": "Sonar Reasoning Pro", "id": "sonar-reasoning-pro", "se": "perplexity", "reasoning": False},
]

MAX_SYSTEM_CHARS = 500
MAX_PROMPT_CHARS = 500


def build_system_prompt(geo: str) -> str:
    geo_note = f" Prioriza marcas y sitios de {geo}." if geo else ""
    base = (
        "Eres un asistente de investigación de mercado. "
        "Menciona siempre marcas, empresas y sitios web específicos. "
        "Incluye URLs cuando las conozcas. "
        "Lista marcas primero, luego explica brevemente cada una."
        + geo_note
    )
    return base[:MAX_SYSTEM_CHARS]


def parse_domain_file(path: Path) -> tuple[str, dict, list[str]]:
    """Returns (title, config, prompts)."""
    text = path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    config = {}
    config_section = re.search(r"##\s+config\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
    if config_section:
        for line in config_section.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                config[k.strip().lower()] = v.strip()

    prompts_section = re.search(r"##\s+prompts\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE)
    prompts = []
    if prompts_section:
        for line in prompts_section.group(1).splitlines():
            line = line.strip()
            if line.startswith("-"):
                prompts.append(line.lstrip("-").strip()[:MAX_PROMPT_CHARS])

    return title, config, prompts


def query_model(model: dict, prompt: str, system: str, geo_iso: str) -> dict:
    cred = base64.b64encode(f"{DFS_LOGIN}:{DFS_PASSWORD}".encode()).decode()
    headers = {"Authorization": f"Basic {cred}", "Content-Type": "application/json"}

    # Reasoning models don't support temperature
    payload = {
        "user_prompt": prompt,
        "model_name": model["id"],
        "system_message": system,
        "max_output_tokens": 2048,
        "web_search": True,
    }
    if geo_iso and model["se"] == "chat_gpt":
        payload["web_search_country_iso_code"] = geo_iso
    if not model["reasoning"]:
        payload["temperature"] = 0.7

    t0 = time.time()
    try:
        endpoint = DATAFORSEO_BASE.format(se=model["se"])
        r = requests.post(endpoint, headers=headers, json=[payload], timeout=120)
        r.raise_for_status()
        data = r.json()

        # Handle both response shapes (tasks[] or direct result[])
        if "tasks" in data:
            task = data["tasks"][0]
            if task.get("status_code", 20000) != 20000:
                raise ValueError(f"Task error {task.get('status_code')}: {task.get('status_message')}")
            result_data = (task.get("result") or [{}])[0]
            cost = task.get("cost", 0)
        else:
            result_data = (data.get("result") or [{}])[0]
            cost = data.get("cost", 0)

        if data.get("status_code", 20000) not in (20000,) and "tasks" not in data:
            raise ValueError(f"API error {data.get('status_code')}: {data.get('status_message')}")

        # Extract message text
        text = ""
        for item in result_data.get("items", []):
            if item.get("type") == "message":
                for section in item.get("sections", []):
                    if section.get("type") == "text":
                        text += section.get("text", "")

        # Extract cited URLs from annotations
        cited_urls = []
        for item in result_data.get("items", []):
            if item.get("type") == "message":
                for section in item.get("sections", []):
                    for ann in section.get("annotations") or []:
                        if ann.get("url"):
                            cited_urls.append(ann["url"])

        return {
            "text": text,
            "tokens_in": result_data.get("input_tokens", 0),
            "tokens_out": result_data.get("output_tokens", 0),
            "cost": result_data.get("money_spent", cost),
            "latency": round(time.time() - t0, 2),
            "cited_urls": cited_urls,
            "error": None,
        }

    except Exception as e:
        return {
            "text": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost": 0,
            "latency": round(time.time() - t0, 2),
            "cited_urls": [],
            "error": str(e),
        }


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\)\]\"'<]+", text)


CSV_COLUMNS = [
    "timestamp", "domain", "geo", "geo_iso", "prompt", "model",
    "response", "urls_mentioned", "cited_urls",
    "tokens_in", "tokens_out", "cost_usd", "latency_s", "error",
]


def save_csv(output_path: Path, title: str, geo: str, geo_iso: str, prompt: str, results: list):
    output_path.parent.mkdir(exist_ok=True)
    write_header = not output_path.exists()
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            urls_in_text = " | ".join(extract_urls(r["text"]))
            cited = " | ".join(r.get("cited_urls", []))
            writer.writerow({
                "timestamp": ts,
                "domain": title,
                "geo": geo,
                "geo_iso": geo_iso,
                "prompt": prompt,
                "model": r["model"]["name"],
                "response": r["text"],
                "urls_mentioned": urls_in_text,
                "cited_urls": cited,
                "tokens_in": r["tokens_in"],
                "tokens_out": r["tokens_out"],
                "cost_usd": f"{r['cost']:.6f}",
                "latency_s": r["latency"],
                "error": r["error"] or "",
            })


def save_html(output_path: Path, title: str, geo: str, all_results: list):
    h = html.escape

    rows = ""
    for entry in all_results:
        prompt = entry["prompt"]
        rows += f'<h2 class="prompt">{h(prompt)}</h2>\n'
        for r in entry["results"]:
            model_name = h(r["model"]["name"])
            if r["error"]:
                body = f'<p class="error">ERROR: {h(r["error"])}</p>'
            else:
                text = h(r["text"])
                all_urls = r.get("cited_urls", []) or extract_urls(r["text"])
                urls_html = ""
                if all_urls:
                    links = " ".join(f'<a href="{h(u)}">{h(u[:60])}</a>' for u in all_urls[:5])
                    urls_html = f'<p class="urls">🔗 {links}</p>'
                body = f'<div class="text"><pre>{text}</pre>{urls_html}</div>'
            rows += f'<div class="model"><div class="model-name">{model_name}</div>{body}</div>\n'

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Prompt Tracker: {h(title)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ font-size: 1.4rem; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .meta {{ color: #666; font-size: .85rem; margin-bottom: 24px; }}
  h2.prompt {{ background: #f0f0f0; padding: 10px 14px; border-left: 4px solid #555; font-size: 1rem; margin-top: 32px; }}
  .model {{ border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; overflow: hidden; }}
  .model-name {{ background: #333; color: #fff; padding: 6px 12px; font-weight: bold; font-size: .85rem; }}
  .text {{ padding: 12px; }}
  pre {{ white-space: pre-wrap; word-break: break-word; font-family: inherit; font-size: .85rem; margin: 0; }}
  .urls {{ font-size: .78rem; color: #555; margin-top: 8px; }}
  .urls a {{ color: #1a6ab1; text-decoration: none; margin-right: 8px; }}
  .error {{ color: #c00; padding: 8px 12px; font-size: .85rem; }}
</style>
</head>
<body>
<h1>Prompt Tracker: {h(title)}</h1>
<p class="meta">Geo: {h(geo or "N/A")} &nbsp;|&nbsp; Fecha: {fecha} &nbsp;|&nbsp; Provider: DataForSEO</p>
{rows}
</body>
</html>"""

    output_path.write_text(doc, encoding="utf-8")
    print(f"HTML guardado en: {output_path}")


def print_results(title: str, geo: str, prompt: str, prompt_num: int, total: int, results: list):
    divider = "=" * 115
    print(f"\n{divider}")
    print(f"  [{prompt_num}/{total}]  {title}  |  geo: {geo or 'N/A'}  |  DataForSEO")
    print(f"  PROMPT: {prompt}")
    print(divider)

    cw = [16, 52, 34, 9]
    print(f"{'Modelo':<{cw[0]}} {'Respuesta (preview)':<{cw[1]}} {'URLs / Marcas citadas':<{cw[2]}} {'Costo $':<{cw[3]}}")
    print("-" * 115)

    for r in results:
        if r["error"]:
            preview = f"ERROR: {r['error'][:50]}"
            urls_str = ""
        else:
            preview = r["text"].replace("\n", " ")[:cw[1]]
            all_urls = r.get("cited_urls", []) or extract_urls(r["text"])
            urls_str = all_urls[0][:cw[2]] if all_urls else "-"

        cost_str = f"${r['cost']:.5f}"
        print(f"{r['model']['name']:<{cw[0]}} {preview:<{cw[1]}} {urls_str:<{cw[2]}} {cost_str}")

    print()
    for r in results:
        if r["error"] or not r["text"]:
            continue
        print(f"--- {r['model']['name']} ---")
        print(r["text"].strip())
        print()


def main():
    parser = argparse.ArgumentParser(description="Prompt tracker via DataForSEO LLM Responses")
    parser.add_argument("domain_file", help="Path to domain markdown file")
    parser.add_argument("--prompt", type=int, default=None, help="Run only prompt #N (1-based)")
    args = parser.parse_args()

    if not DFS_LOGIN or not DFS_PASSWORD:
        print("Error: define DATAFORSEO_LOGIN y DATAFORSEO_PASSWORD en tu .env")
        sys.exit(1)

    domain_path = Path(args.domain_file)
    if not domain_path.exists():
        print(f"Error: no existe {domain_path}")
        sys.exit(1)

    title, config, prompts = parse_domain_file(domain_path)
    geo = config.get("geo", "")
    geo_iso = config.get("geo_iso", "")
    system = build_system_prompt(geo)

    if not prompts:
        print("Error: no se encontraron prompts. Revisa la sección '## prompts'.")
        sys.exit(1)

    if args.prompt:
        if args.prompt < 1 or args.prompt > len(prompts):
            print(f"Error: --prompt debe estar entre 1 y {len(prompts)}")
            sys.exit(1)
        prompts = [prompts[args.prompt - 1]]
        offset = args.prompt - 1
    else:
        offset = 0

    stem = re.sub(r"[^\w-]", "_", domain_path.stem)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output")
    csv_path = output_dir / f"{stem}_{ts_str}.csv"
    pdf_path = output_dir / f"{stem}_{ts_str}.pdf"

    print(f"\nDominio:  {title}")
    print(f"Geo:      {geo or 'N/A'}  |  ISO: {geo_iso or 'N/A'}")
    print(f"Prompts:  {len(prompts)}  |  Modelos: {', '.join(m['name'] for m in MODELS)}")
    print(f"Salida:   {csv_path}")

    all_results = []
    for i, prompt in enumerate(prompts, start=1):
        print(f"\nConsultando [{i+offset}/{len(prompts)+offset}]: {prompt[:70]}...")
        results = []
        for model in MODELS:
            sys.stdout.write(f"  → {model['name']}... ")
            sys.stdout.flush()
            r = query_model(model, prompt, system, geo_iso)
            r["model"] = model
            results.append(r)
            print(f"OK ({r['latency']}s  ${r['cost']:.5f})" if not r["error"] else f"ERROR: {r['error'][:80]}")

        print_results(title, geo, prompt, i + offset, len(prompts) + offset, results)
        save_csv(csv_path, title, geo, geo_iso, prompt, results)
        all_results.append({"prompt": prompt, "results": results})

    save_html(pdf_path.with_suffix(".html"), title, geo, all_results)
    total_cost = sum(r["cost"] for entry in all_results for r in entry["results"])
    print(f"CSV guardado en:  {csv_path}")
    print(f"Costo total:      ${total_cost:.5f} USD")


if __name__ == "__main__":
    main()
