# ai-blog-copilot

Two tools that share a database. A **trend radar** that goes and reads what
happened in AI engineering today and proposes what is worth writing about, and
a **writing studio** that turns one of those proposals into a Markdown file
ready for a blog.

They are built to different standards on purpose.

The radar is an agent written from scratch — a hand-rolled ReAct loop, no
LangChain, no CrewAI, no framework of any kind. That is the point of it: the
interesting part of an agent is not the happy path, it is what happens when the
model returns nonsense, a tool times out, a fetched page tries to give
instructions, or the model answers confidently without having looked anything
up. All of that is visible here rather than inherited.

The studio is an internal tool and uses whatever makes it quick. Streamlit,
libraries, no ceremony.

---

## What the radar does

```
$ uv run python -m radar.run --hours 24

LLM calls this run:
  gemini   9 calls, 2 failed, 40,167 in / 2,993 out, ~$0.0195
{
  "steps_used": 7,
  "stopped_because": "final_answer",
  "topics": [
    {
      "title": "Structured Output Request Significantly Reduces LLM Answer Diversity",
      "why_now": "Released within the last 24 hours on arXiv…",
      "sources": ["https://arxiv.org/abs/2607.18476"],
      "angle": "practical",
      "suggested_outline": ["The Hidden Cost of 'Reply with JSON'", "…"]
    }
  ]
}
```

It has three tools — Hacker News via the Algolia API, a curated list of RSS
feeds in [`feeds.yaml`](feeds.yaml), and full-text extraction of any single URL
— and it decides for itself which to call and in what order.

### The loop

`src/radar/agent.py`. Thought → Action → Observation, until the model produces
a final answer or runs out of steps. Every failure mode is handled explicitly:

| What goes wrong | What happens |
| --- | --- |
| Reply is not valid JSON | Parse error goes back to the model as a repair prompt |
| Reply is JSON but not a legal move | Told so; the loop continues |
| Tool does not exist, or raises | `ERROR:` observation; the loop continues |
| Steps run out | One final call that forbids tools and demands an answer from the evidence already gathered |
| Final answer fails the schema | Schema error back as a repair prompt, once |
| Final answer cites something no tool returned | **Rejected** |

That last row is there because of a real incident, described below.

---

## The three things worth reading the code for

### 1. Untrusted content is fenced structurally, not filtered

`src/radar/sanitize.py`. Everything the tools fetch — Hacker News titles, RSS
summaries, the body of an article — is attacker-controlled text that goes
straight into a prompt. The defence is a per-run 16-hex nonce:

```
<untrusted-data nonce="a3f9…" source="tool-result">
…fetched content…
</untrusted-data nonce="a3f9…">
```

Before wrapping, anything that could forge that structure is neutralised: the
nonce itself, tag fragments, role markers at the start of a line, control
characters, zero-width and bidi overrides.

What it deliberately does **not** do is filter on meaning. A page containing
the words "ignore previous instructions" passes through untouched, wrapped. A
blocklist of suspicious phrases would be security theatre in both directions:
trivially bypassed, and it would break the day the radar finds a genuinely
interesting article *about* prompt injection — exactly the kind of thing this
tool exists to find.

### 2. Evidence is verified mechanically, not requested politely

The first real scheduled run finished in **one step**, called **no tools at
all**, and produced three entirely convincing topics — correct format, sensible
outlines, sourced to Hacker News item IDs from May 2024. One `why_now` read
"within the last 24 hours (2024-05-28)". It had answered from training data.
Schema validation passed, CI went green, and the fabrication was committed to a
public repository.

That is the worst failure mode available to a trend radar. An empty result is
obviously broken. A plausible one that is two years stale is not.

The prompt had already asked for sources "you actually saw in an observation".
Asking is not a control. Two mechanical checks now enforce it:

- A final answer with no observations behind it is refused and the loop
  continues. Answering before reading anything means answering from memory, by
  definition.
- Every cited URL must have appeared in an observation. Invented ones are named
  back in the repair prompt. Comparison is normalised for trailing slashes and
  fragments — models reformat URLs, and failing over a slash would be pedantry
  rather than rigour.

If the step limit is reached with no evidence at all, the run raises rather
than producing anything. An alert about an empty run beats a committed
invention.

### 3. The model chooses the URLs, so the fetcher assumes hostility

`src/radar/net.py`. `fetch_article_text` takes a URL that came out of untrusted
content and was chosen by an LLM. So: scheme allowlist, DNS resolution with an
`ip.is_global` check before connecting, the check repeated **on every redirect
hop** (redirects are followed manually for exactly this reason), and a 2 MB
response cap. `http://169.254.169.254` and `http://127.0.0.1:11434` are
refused, not fetched.

---

## The provider chain

`NVIDIA NIM → Gemini → Ollama`, all over plain REST with `httpx`. No vendor
SDKs: three wire formats is little enough code to own outright, and owning it
keeps the whole chain mockable with one tool and free of dependency drift.

NIM leads, not Gemini. Gemini's free tier is 20 requests a day and a radar pass
makes 9 to 11 of them, so leading with it would spend the whole daily budget on
one run. Held as the second tier it is the reserve that catches NIM when NIM
fails — which is what a fallback is for — and the common path skips the 12s
pacing Gemini needs for its per-minute limit.

The error hierarchy is what everything else keys off:

- `RetryableError` — timeout, connection reset, 429, 5xx. Retry the same
  provider with exponential backoff.
- `FatalError` — bad credentials, malformed request. Skip to the next tier
  immediately; waiting will not mint a valid API key.

A provider with no credentials is dropped when the chain is built, not
discovered mid-run.

**Rate limits are respected rather than discovered.** Gemini's free tier allows
5 requests per minute and 20 per day; a radar pass makes 9 to 11. Calls to a
provider that declares a `min_interval` are paced (12s for Gemini, nothing for
NIM or local Ollama), and time already spent answering counts towards the
interval. When a 429 arrives anyway, `Retry-After` is honoured if the server
sent one — a rate limit knows its own window better than a backoff formula
does.

---

## The studio

```bash
uv run streamlit run src/studio/app.py
```

Three tabs.

**Radar** — read-only over the history. Filter by window and angle; each topic
shows why it matters now, its sources and a suggested outline. "Write this one"
hands it to the next tab.

**Studio** — outline, then section by section, then export. Nothing runs on its
own: every model call is behind a button and everything it returns lands in a
text area you can edit or throw away. Each section is drafted knowing what the
earlier ones said, so section three does not re-introduce the topic.

**Costs** — the headline number is not the interesting part; two of the three
tiers are free. The tab exists for the column next to it. A fallback chain can
degrade silently — the primary fails for a week, the second tier quietly
answers everything, every run stays green — and a per-tier success rate is the
only place that shows up. Latency percentiles cover successful calls only: a
failure records no time, and averaging those in would make a provider look
faster the more it broke.

### Export

`output/posts/<slug>.md`, with front-matter matching what the target blog's zod
schema validates at build time.

Two rules that are not configurable:

- **Always `draft: true`.** This tool proposes; publishing is a decision made
  deliberately, once per post, in the blog's own repository.
- **`relatedProject` is emitted only if the slug really exists.** It is a typed
  `reference()` in the blog's content config, so a slug naming no project does
  not warn — it breaks the build. Without `PORTFOLIO_PATH` set there is no way
  to check, so the field is left out.

Nothing here ever writes to, commits to, or pushes to the blog repository.
Copying the file across is a manual step.

---

## Setup

```bash
uv sync --extra radar --extra studio
cp .env.example .env      # then fill in at least one provider
```

Ollama needs no key and is the only tier that survives having no network, so a
local model is enough to run everything. `LLM_ONLY=ollama` pins the chain to
one tier, which is useful for trying things without spending credit.

```bash
uv run python -m radar.run --hours 24 --max-steps 8 --export data/topics.json
uv run streamlit run src/studio/app.py
```

History lives in `radar.db` (SQLite, gitignored). The committed
`data/topics.json` is the durable copy — the database is just the working one,
and the daily workflow keeps it in a cache that GitHub evicts after a week.

---

## Tests

```bash
uv run pytest          # 283 tests, no network
uv run ruff check .
```

The suite touches no sockets: HTTP is mocked with `respx`, providers with a
scripted fake, and `fetch` is injected into every tool. Two things are tested
in ways worth mentioning:

- The exporter is validated against the **real** zod schema from the target
  blog, transpiled and imported, rather than a reimplementation of it. A
  validator built from my own assumptions would agree with them and prove
  nothing.
- The Streamlit tabs are driven through Streamlit's own `AppTest` harness. That
  is not gold-plating — a section once generated correctly, logged its tokens,
  and displayed an empty box, because a keyed widget takes its content from
  session state and ignores `value=`. No test of the drafting code could have
  caught it.

---

## Known limits

- **Reddit is not a source.** It was, briefly. The public JSON endpoints now
  return 403 to unauthenticated clients, the current credential path (Devvit)
  is architecturally incompatible with a script like this, and Reddit's
  `robots.txt` disallows the paths that would otherwise work. The tool was
  removed rather than quietly degraded.
- **Costs are an estimate.** The price table is hand-maintained and only covers
  Gemini. NIM's free tier and local Ollama count as zero. It is an
  order-of-magnitude signal, not a bill.
- **The daily workflow is manual.** `workflow_dispatch` only. Not because of
  cost — a pass is about $0.02 — but because 20 Gemini requests per day is two
  passes, so a schedule plus one manual dispatch spends the entire budget.
- **The studio has no evidence guardrail, deliberately.** A small local model
  will invent plausible numbers in a draft. This half is a writing assistant
  with a human editing every line; the constraint belongs on the radar, which
  runs unattended, and not here.
