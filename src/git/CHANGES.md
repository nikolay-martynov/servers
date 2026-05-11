# Local Modifications

This is a personal fork of [mcp-server-git](https://github.com/modelcontextprotocol/servers/tree/main/src/git) with custom modifications.

See git history for details (`git diff upstream/main`).

## Changes

- Add auto-chunking for diff output (`git_diff_unstaged`, `git_diff_staged`, `git_diff`, `git_show`): splits large diffs at line boundaries (~5k tokens / 18KB per chunk) with continuation tokens for pagination, preventing LLM context overflow on large diffs