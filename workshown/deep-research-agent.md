# Building a research agent that *does* the work

Samuel Borelli · June 2026

---

I was spending too much time on research. Not the reading — the *find the right sources, read them, figure out what they actually mean, put it together* loop. That sequence takes hours for any question with depth.

The tools exist. Search APIs, content scrapers, LLMs. But they're separate. You search, you get links, you open them, you read, you switch tabs, you synthesize. The friction of moving between tools is where the time goes. I wanted one thing that did the whole loop automatically.

---

## What I built

A ReAct agent — reasoning loop that searches the web, reads pages, and synthesizes findings into a structured report. It runs via any OpenAI-compatible API. Tools execute locally.

The loop:

- Send the question to the model
- Model decides: search, read a page, or answer
- Execute the tool locally (Serper.dev for search, Jina.ai for reading)
- Feed the result back into context
- Repeat until the model says it has enough

Max 50 rounds per session. Usually takes 4–8.

## How it decides what to search for

The model drives, not the code. It decides the search queries, picks which pages to read, and determines when it has enough information. The code just handles tool execution and passes results back.

This means the same agent handles a shallow question in 2 rounds and a deep one in 15 — no code changes, no configuration. The model adapts to the question.

## How pages get read

Jina.ai takes a URL and returns markdown. For long pages, I truncate and use a secondary LLM call to extract what's relevant. The extractor returns three things: why the page is relevant, the supporting quotes, and a condensed summary.

The two-step approach means the agent reads summaries with citations, not full pages. Keeps costs down and lets it cover more sources per session.

## What it produced

I ran it on a question about how regular people (not influencers, not course sellers) actually get jobs. The agent searched Reddit threads, read personal blogs, found forum discussions, and returned 10 sourced anecdotes with real conversion rates — 1.2% from 847 cold emails, 45% view rate on Upwork proposals, a summary table comparing tactics.

> ~4 minutes. ~$0.30. The same research manually would have been 2–3 hours, and I would have stopped at 2–3 sources instead of 10.

## What I'd change

The extractor prompt misses nuance in opinion-based content. It's built for extracting factual evidence, not reading between the lines. A secondary pass focused on tone would help for the kind of qualitative research I do most.

The report output is a flat markdown file. For longer sessions the structure works, but for quick questions I'd prefer summary-first with depth available on demand.

---

## The stack

- **Loop:** ReAct protocol from Alibaba-NLP/DeepResearch
- **Search:** Serper.dev API
- **Reading:** Jina.ai Reader API + LLM extractor
- **Model:** DeepSeek V4 Flash (any OpenAI-compatible API works)
- **Server:** MCP protocol wrapper for async dispatch
- **Total:** ~400 lines across 6 files
