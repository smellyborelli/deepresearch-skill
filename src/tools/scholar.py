import os
import json
import requests
from typing import Union, List
from _qwen_stubs import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor
import http.client

from source_quality import add_dashboard_to_response

SERPER_KEY=os.environ.get('SERPER_KEY_ID')


@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar."
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

    def google_scholar_with_serp(self, query: str, cumulative_scores: dict | None = None):
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
        "q": query,
        })
        headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
        }
        for i in range(5):
            try:
                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google Scholar Timeout, return None, Please try again later.", cumulative_scores
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
                    if "year" in page:
                        date_published = "\nDate published: " + str(page["year"])

                    publicationInfo = ""
                    if "publicationInfo" in page:
                        publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    link_info = "no available link"
                    if "pdfUrl" in page:
                        link_info = "pdfUrl: " + page["pdfUrl"]

                    citedBy = ""
                    if "citedBy" in page:
                        citedBy = "\ncitedBy: " + str(page["citedBy"])

                    redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)

            # Scholar uses 'pdfUrl' instead of 'link' — normalise
            scholar_results = []
            for page in results["organic"]:
                page_copy = dict(page)
                if "pdfUrl" in page_copy and "link" not in page_copy:
                    page_copy["link"] = page_copy["pdfUrl"]
                scholar_results.append(page_copy)

            # Append source quality dashboard
            if os.getenv("ENABLE_SOURCE_QUALITY", "true").lower() == "true":
                content, cumulative_scores = add_dashboard_to_response(
                    content, scholar_results, cumulative_scores
                )

            return content, cumulative_scores
        except:
            return f"No results found for '{query}'. Try with a more general query.", cumulative_scores


    def call(self, params: Union[str, dict], **kwargs) -> str:
        # assert GOOGLE_SEARCH_KEY is not None, "Please set the IDEALAB_SEARCH_KEY environment variable."
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        cumulative_scores = params.get("cumulative_scores", {}) if isinstance(params, dict) else {}

        if isinstance(query, str):
            response, _ = self.google_scholar_with_serp(query, cumulative_scores)
        else:
            assert isinstance(query, List)
            responses = []
            cum = cumulative_scores
            for q in query:
                resp, cum = self.google_scholar_with_serp(q, cum)
                responses.append(resp)
            response = "\n=======\n".join(responses)
        return response
