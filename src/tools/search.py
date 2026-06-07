import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from _qwen_stubs import BaseTool, register_tool
from source_quality import add_dashboard_to_response
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json

import os


SERPER_KEY=os.environ.get('SERPER_KEY_ID')


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
            "cumulative_scores": {
                "type": "object",
                "properties": {
                    "tier1_count": {"type": "integer"},
                    "tier2_count": {"type": "integer"},
                    "tier3_count": {"type": "integer"},
                    "tier4_count": {"type": "integer"},
                    "tier1_score": {"type": "integer"},
                    "tier2_score": {"type": "integer"},
                    "tier3_score": {"type": "integer"},
                    "tier4_score": {"type": "integer"},
                    "total_possible": {"type": "integer"}
                },
                "required": [],
                "description": "Cumulative quality scores from previous rounds. Pass the CUMULATIVE_SCORES value from the last search/scholar response verbatim."
            }
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    def google_search_with_serp(self, query: str, cumulative_scores: dict | None = None):
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        conn = http.client.HTTPSConnection("google.serper.dev")
        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
            
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })
        headers = {
                'X-API-KEY': SERPER_KEY,
                'Content-Type': 'application/json'
            }
        
        
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later.", cumulative_scores
                continue
    
        data = res.read()
        results = json.loads(data.decode("utf-8"))

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            # Append source quality dashboard
            if os.getenv("ENABLE_SOURCE_QUALITY", "true").lower() == "true":
                content, cumulative_scores = add_dashboard_to_response(
                    content, results["organic"], cumulative_scores
                )
            return content, cumulative_scores
        except:
            return f"No results found for '{query}'. Try with a more general query.", cumulative_scores


    
    def search_with_serp(self, query: str, cumulative_scores: dict | None = None):
        return self.google_search_with_serp(query, cumulative_scores)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response, _ = self.search_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.search_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)

        return response

