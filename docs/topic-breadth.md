# GitHub topic breadth collection

The topic collector is a separate, bounded source of discovery observations. It queries GitHub's GraphQL `topic(name:)` repository connection using the ordered 38-slug catalog in [`config/topics.toml`](../config/topics.toml). The eight field topics added for this expansion are `geospatial`, `remote-sensing`, `bioinformatics`, `cheminformatics`, `speech-recognition`, `text-to-speech`, `medical-imaging`, and `recommender-system`. Each request asks for up to 100 repositories ordered by most recently updated and follows the connection cursor. GitHub documents the [`Topic` GraphQL object](https://docs.github.com/en/graphql/reference/objects#topic) and [cursor pagination](https://docs.github.com/en/graphql/guides/using-pagination-in-the-graphql-api).

The configured daily workflow allows up to 68 GraphQL pages: up to 38 first-page refreshes, one for each configured topic, plus up to 30 deeper pages across topics. First-page refreshes query each topic from the head daily, so recently updated repositories can be rediscovered even while a deep traversal is paused or complete. Deeper pages follow each topic's saved cursor and are allocated fairly across eligible topics. The collector stores progress in `state/topic-breadth.json`; each topic tracks its deep cursor, page index, sweep number, and completion time. After a topic's deep traversal completes, it becomes eligible to restart that deep sweep after 30 days; its first page continues to refresh daily. The daily budget may end before all deep pages finish, and later Hub-backed runs continue from saved progress.

This process is bounded and best effort. GitHub's topic membership and repository metadata may change while pages are being traversed; the collector has no completeness or snapshot-isolation guarantee. Daily head refreshes, ordering by `UPDATED_AT`, cursor pagination, fair scheduling, and recurring 30-day deep sweeps improve coverage but cannot ensure a stable or exhaustive enumeration. The 68-page ceiling is a maximum, not a promise: GraphQL query costs consume the GraphQL point budget and can make the collector stop before using all page slots; the remaining point budget, throttling, and request errors can also stop a run early. Coverage and saved state record progress for Hub-backed runs. A missing topic returns an empty completed page. Forks are omitted. Per-page coverage records the topic slug, sweep and page numbers, observation count, cursor outcome, missing-topic flag, and available GraphQL rate-limit remainder.

Each collected repository observation is queryless: `query_ids` is empty, `queryless` is true, `discovery_source` is `topic`, and `topic_names` records the catalog slug. Repository metadata and tags are captured with the observation. Tags and topic slugs describe provenance or classification; they do not qualify a repository for the strict selector. Raw topic observations remain in history, and current-view selection applies `ml-contribution-v4` after Search-source precedence and latest-observation selection. For the same GitHub ID, Search observations take precedence over topic and census observations.

A published topic run is committed with its observation JSONL (when nonempty), aggregate coverage, updated `state/topic-breadth.json`, and a run marker at `runs/topic-breadth-<run-id>.manifest.json`. The marker hashes the run payloads. This is a collector run artifact; it is not the current-view Parquet snapshot. Snapshot generation is a separate operation.

## Local collection

With a GitHub token available in `GITHUB_TOKEN`, run a local, non-publishing collection with:

```sh
uv run gh-ml topic-breadth-daily --work-dir /tmp/gh-ml-topic-breadth --max-pages 68 --no-publish
```

The command writes run outputs and a checkpoint under a new run-ID subdirectory of the work directory. In `--no-publish` mode, a later invocation starts a fresh run; cross-run resume uses the Hub-backed `state/topic-breadth.json` state. `--no-publish` avoids downloading or publishing Hub state; it does not alter the collector's pagination or coverage limits. The CLI accepts a page budget up to 100, and the configured daily budget is 68. In local `--no-publish` mode, runs are isolated and do not carry progress to the next invocation.

The Search API has a separate per-query result cap of 1,000; see GitHub's [Search API documentation](https://docs.github.com/en/rest/search/search). Topic GraphQL collection does not consume that Search request budget, and its pagination is subject to GraphQL API limits and rate limits.
