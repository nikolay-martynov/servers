# Local Modifications

This is a personal fork of [mcp-server-git](https://github.com/modelcontextprotocol/servers/tree/main/src/git) with custom modifications.

See git history for details (`git diff upstream/main`).

## Changes

- Add auto-chunking for diff output (`git_diff_unstaged`, `git_diff_staged`, `git_diff`, `git_show`): splits large diffs at line boundaries (~5k tokens / 18KB per chunk) with continuation tokens for pagination, preventing LLM context overflow on large diffs
- Add `paths` parameter to diff tools (`git_diff_unstaged`, `git_diff_staged`, `git_diff`): filter diff to specific files instead of always getting the full working tree diff, avoiding unnecessary token usage
- Add `stat` parameter to diff tools (`git_diff_unstaged`, `git_diff_staged`, `git_diff`): include compact `--stat` summary (file names + insertion/deletion counts) alongside diff output; enabled by default since chunking handles large output, set `stat=false` for diff-only
- Add `diff` parameter to diff tools: set `diff=false` for stat-only output (no patch body), useful for quick change overviews