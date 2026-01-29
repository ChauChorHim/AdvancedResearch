import json
import os
from typing import Any

import requests
from loguru import logger


def any_to_str(data: Any) -> str:
    return str(data)


def duckduckgo_search_tool(
    query: str, characters: int = 200, sources: int = 3
) -> str:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query (str): The search query.
        characters (int): Not used for DDG but kept for interface compatibility.
        sources (int): Number of results to return.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo-search package is not installed. Please install it with `pip install duckduckgo-search`."

    try:
        logger.info(f"[DDG SEARCH] Searching for: {query[:50]}...")
        with DDGS() as ddgs:
            # ddgs.text returns a generator of dicts: {'title':..., 'href':..., 'body':...}
            results = list(ddgs.text(query, max_results=sources))

        # Format results to look somewhat like Exa's output for consistency
        formatted_results = []
        for r in results:
            formatted_results.append(
                {
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "text": r.get("body", ""),
                    "score": 0.0,
                    "id": "",
                    "publishedDate": "",
                }
            )

        return json.dumps({"results": formatted_results}, indent=2)
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return f"Search failed: {str(e)}"


def gemini_search_tool(
    query: str, characters: int = 200, sources: int = 3
) -> str:
    """
    Perform a web search using Google Gemini Grounding.

    Args:
        query (str): The search query.
        characters (int): Not used but kept for interface compatibility.
        sources (int): Not directly controllable in grounding (usually ~5-10), kept for interface.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found in environment variables."

    # Using gemini-1.5-flash for speed and cost effectiveness
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": f"Search for: {query}"}]}],
        "tools": [
            {
                "google_search_retrieval": {
                    "dynamic_retrieval_config": {
                        "mode": "MODE_DYNAMIC",
                        "dynamic_threshold": 0.0,  # Force search
                    }
                }
            }
        ],
    }

    headers = {"Content-Type": "application/json"}

    try:
        logger.info(f"[GEMINI SEARCH] Searching for: {query[:50]}...")
        response = requests.post(
            url, headers=headers, json=payload, timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Gemini API error: {response.text}")
            return f"Search failed: Gemini API returned status {response.status_code}"

        data = response.json()

        # Extract grounding metadata
        candidates = data.get("candidates", [])
        if not candidates:
            return json.dumps({"results": []})

        candidate = candidates[0]
        grounding_metadata = candidate.get("groundingMetadata", {})
        chunks = grounding_metadata.get("groundingChunks", [])

        formatted_results = []

        # 1. Add the synthesized answer from Gemini as a primary result
        text_content = ""
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            text_content += part.get("text", "")

        if text_content:
            formatted_results.append(
                {
                    "title": "Gemini Search Summary",
                    "url": "google_search_grounding",
                    "text": text_content,
                    "score": 1.0,
                }
            )

        # 2. Add the sources as individual results (title/url)
        for chunk in chunks:
            web = chunk.get("web")
            if web:
                formatted_results.append(
                    {
                        "title": web.get("title", "Unknown Title"),
                        "url": web.get("uri"),
                        "text": "Source referenced in Gemini Grounding",
                        "score": 0.8,
                    }
                )

        return json.dumps({"results": formatted_results}, indent=2)

    except Exception as e:
        logger.error(f"Gemini search failed: {e}")
        return f"Search failed: {str(e)}"
