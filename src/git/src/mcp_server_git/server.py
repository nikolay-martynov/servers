import json
import logging
from pathlib import Path
from typing import Sequence, Optional
from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.types import (
    ClientCapabilities,
    TextContent,
    Tool,
    ListRootsResult,
    RootsCapability,
    ToolAnnotations,
)
from enum import Enum
import git
from git.exc import BadName
from pydantic import BaseModel, Field

# Default number of context lines to show in diff output
DEFAULT_CONTEXT_LINES = 3

# Maximum bytes per diff chunk (~5k tokens ≈ 18KB at ~3.6 chars/token)
DEFAULT_MAX_DIFF_BYTES = 18000


class GitStatus(BaseModel):
    repo_path: str

class GitDiffUnstaged(BaseModel):
    repo_path: str
    context_lines: int = DEFAULT_CONTEXT_LINES
    paths: Optional[list[str]] = Field(
        None,
        description="Optional list of file paths to filter the diff to. When omitted, shows diff for all changed files.",
    )
    stat: bool = Field(
        True,
        description="Include diff statistics (--stat) showing file names with insertion/deletion counts. Set false for diff-only output.",
    )
    diff: bool = Field(
        True,
        description="Include diff body (patch). Set false for stat-only output.",
    )
    auto_chunk: bool = Field(
        True,
        description="Enable automatic chunking of large diff output to fit within token limits",
    )
    continuation_token: Optional[str] = Field(
        None,
        description="Token from a previous chunked response to fetch the next chunk. Omit for first chunk.",
    )

class GitDiffStaged(BaseModel):
    repo_path: str
    context_lines: int = DEFAULT_CONTEXT_LINES
    paths: Optional[list[str]] = Field(
        None,
        description="Optional list of file paths to filter the diff to. When omitted, shows diff for all changed files.",
    )
    stat: bool = Field(
        True,
        description="Include diff statistics (--stat) showing file names with insertion/deletion counts. Set false for diff-only output.",
    )
    diff: bool = Field(
        True,
        description="Include diff body (patch). Set false for stat-only output.",
    )
    auto_chunk: bool = Field(
        True,
        description="Enable automatic chunking of large diff output to fit within token limits",
    )
    continuation_token: Optional[str] = Field(
        None,
        description="Token from a previous chunked response to fetch the next chunk. Omit for first chunk.",
    )

class GitDiff(BaseModel):
    repo_path: str
    target: str
    context_lines: int = DEFAULT_CONTEXT_LINES
    paths: Optional[list[str]] = Field(
        None,
        description="Optional list of file paths to filter the diff to. When omitted, shows diff for all changed files.",
    )
    stat: bool = Field(
        True,
        description="Include diff statistics (--stat) showing file names with insertion/deletion counts. Set false for diff-only output.",
    )
    diff: bool = Field(
        True,
        description="Include diff body (patch). Set false for stat-only output.",
    )
    auto_chunk: bool = Field(
        True,
        description="Enable automatic chunking of large diff output to fit within token limits",
    )
    continuation_token: Optional[str] = Field(
        None,
        description="Token from a previous chunked response to fetch the next chunk. Omit for first chunk.",
    )

class GitCommit(BaseModel):
    repo_path: str
    message: str

class GitAdd(BaseModel):
    repo_path: str
    files: list[str]

class GitReset(BaseModel):
    repo_path: str

class GitLog(BaseModel):
    repo_path: str
    max_count: int = 10
    start_timestamp: Optional[str] = Field(
        None,
        description="Start timestamp for filtering commits. Accepts: ISO 8601 format (e.g., '2024-01-15T14:30:25'), relative dates (e.g., '2 weeks ago', 'yesterday'), or absolute dates (e.g., '2024-01-15', 'Jan 15 2024')"
    )
    end_timestamp: Optional[str] = Field(
        None,
        description="End timestamp for filtering commits. Accepts: ISO 8601 format (e.g., '2024-01-15T14:30:25'), relative dates (e.g., '2 weeks ago', 'yesterday'), or absolute dates (e.g., '2024-01-15', 'Jan 15 2024')"
    )

class GitCreateBranch(BaseModel):
    repo_path: str
    branch_name: str
    base_branch: str | None = None

class GitCheckout(BaseModel):
    repo_path: str
    branch_name: str

class GitShow(BaseModel):
    repo_path: str
    revision: str
    auto_chunk: bool = Field(
        True,
        description="Enable automatic chunking of large diff output to fit within token limits",
    )
    continuation_token: Optional[str] = Field(
        None,
        description="Token from a previous chunked response to fetch the next chunk. Omit for first chunk.",
    )



class GitBranch(BaseModel):
    repo_path: str = Field(
        ...,
        description="The path to the Git repository.",
    )
    branch_type: str = Field(
        ...,
        description="Whether to list local branches ('local'), remote branches ('remote') or all branches('all').",
    )
    contains: Optional[str] = Field(
        None,
        description="The commit sha that branch should contain. Do not pass anything to this param if no commit sha is specified",
    )
    not_contains: Optional[str] = Field(
        None,
        description="The commit sha that branch should NOT contain. Do not pass anything to this param if no commit sha is specified",
    )


class GitTools(str, Enum):
    STATUS = "git_status"
    DIFF_UNSTAGED = "git_diff_unstaged"
    DIFF_STAGED = "git_diff_staged"
    DIFF = "git_diff"
    COMMIT = "git_commit"
    ADD = "git_add"
    RESET = "git_reset"
    LOG = "git_log"
    CREATE_BRANCH = "git_create_branch"
    CHECKOUT = "git_checkout"
    SHOW = "git_show"

    BRANCH = "git_branch"


def chunk_output(
    text: str,
    max_bytes: int = DEFAULT_MAX_DIFF_BYTES,
    continuation_token: Optional[str] = None,
) -> tuple[str, str]:
    """Chunk output to fit within token limits.

    Splits at line boundaries to avoid breaking mid-line.
    Returns (chunk_text, chunk_metadata_json).

    If the output fits within max_bytes, returns it as-is with empty metadata.
    Otherwise, returns the current chunk with metadata containing:
      - current_chunk: 1-based chunk number
      - total_chunks: estimated total chunks
      - has_more: whether there are more chunks
      - continuation_token: token to fetch next chunk (byte offset)
      - size_info: current_size and max_size in bytes
    """
    total_bytes = len(text.encode("utf-8"))

    if total_bytes <= max_bytes and continuation_token is None:
        return text, ""

    # Decode continuation_token as byte offset
    start_offset = 0
    if continuation_token is not None:
        try:
            start_offset = int(continuation_token)
        except ValueError:
            start_offset = 0

    if start_offset >= total_bytes:
        return "", json.dumps({
            "chunking": {
                "current_chunk": 0,
                "total_chunks": 0,
                "has_more": False,
                "continuation_token": None,
                "size_info": {
                    "current_size": 0,
                    "max_size": max_bytes,
                    "total_size": total_bytes,
                },
            }
        })

    # Find the end position for this chunk, splitting at line boundary
    # Work with the string but track byte positions
    text_from_offset = text[start_offset:] if start_offset > 0 else text
    chunk_bytes = 0
    chunk_end = 0

    for line in text_from_offset.splitlines(keepends=True):
        line_byte_len = len(line.encode("utf-8"))
        if chunk_bytes + line_byte_len > max_bytes and chunk_bytes > 0:
            break
        chunk_bytes += line_byte_len
        chunk_end += len(line)

    chunk_text = text_from_offset[:chunk_end]
    next_offset = start_offset + chunk_bytes
    has_more = next_offset < total_bytes

    current_chunk = (start_offset // max_bytes) + 1
    total_chunks = (total_bytes + max_bytes - 1) // max_bytes

    metadata = json.dumps({
        "chunking": {
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
            "has_more": has_more,
            "continuation_token": str(next_offset) if has_more else None,
            "size_info": {
                "current_size": chunk_bytes,
                "max_size": max_bytes,
                "total_size": total_bytes,
                "usage_percentage": round(chunk_bytes / max_bytes * 100, 1),
            },
        }
    })

    return chunk_text, metadata


def git_status(repo: git.Repo) -> str:
    return repo.git.status()

def git_diff_unstaged(repo: git.Repo, context_lines: int = DEFAULT_CONTEXT_LINES, paths: Optional[list[str]] = None, stat: bool = True, diff: bool = True) -> str:
    args = []
    if diff:
        args.append(f"--unified={context_lines}")
    if stat:
        args.append("--stat")
    if paths:
        args.append("--")
        args.extend(paths)
    return repo.git.diff(*args)

def git_diff_staged(repo: git.Repo, context_lines: int = DEFAULT_CONTEXT_LINES, paths: Optional[list[str]] = None, stat: bool = True, diff: bool = True) -> str:
    args = []
    if diff:
        args.append(f"--unified={context_lines}")
    args.append("--cached")
    if stat:
        args.append("--stat")
    if paths:
        args.append("--")
        args.extend(paths)
    return repo.git.diff(*args)

def git_diff(repo: git.Repo, target: str, context_lines: int = DEFAULT_CONTEXT_LINES, paths: Optional[list[str]] = None, stat: bool = True, diff: bool = True) -> str:
    # Defense in depth: reject targets starting with '-' to prevent flag injection,
    # even if a malicious ref with that name exists (e.g. via filesystem manipulation)
    if target.startswith("-"):
        raise BadName(f"Invalid target: '{target}' - cannot start with '-'")
    repo.rev_parse(target)  # Validates target is a real git ref, throws BadName if not
    args = []
    if diff:
        args.append(f"--unified={context_lines}")
    args.append(target)
    if stat:
        args.append("--stat")
    if paths:
        args.append("--")
        args.extend(paths)
    return repo.git.diff(*args)

def git_commit(repo: git.Repo, message: str) -> str:
    commit = repo.index.commit(message)
    return f"Changes committed successfully with hash {commit.hexsha}"

def git_add(repo: git.Repo, files: list[str]) -> str:
    if files == ["."]:
        repo.git.add(".")
    else:
        # Use '--' to prevent files starting with '-' from being interpreted as options
        repo.git.add("--", *files)
    return "Files staged successfully"

def git_reset(repo: git.Repo) -> str:
    repo.index.reset()
    return "All staged changes reset"

def git_log(repo: git.Repo, max_count: int = 10, start_timestamp: Optional[str] = None, end_timestamp: Optional[str] = None) -> list[str]:
    if start_timestamp or end_timestamp:
        # Defense in depth: reject timestamps starting with '-' to prevent flag injection
        if start_timestamp and start_timestamp.startswith("-"):
            raise ValueError(f"Invalid start_timestamp: '{start_timestamp}' - cannot start with '-'")
        if end_timestamp and end_timestamp.startswith("-"):
            raise ValueError(f"Invalid end_timestamp: '{end_timestamp}' - cannot start with '-'")
        # Use git log command with date filtering
        args = []
        if start_timestamp:
            args.extend(['--since', start_timestamp])
        if end_timestamp:
            args.extend(['--until', end_timestamp])
        args.extend(['--format=%H%n%an%n%ad%n%s%n'])

        log_output = repo.git.log(*args).split('\n')

        log = []
        # Process commits in groups of 4 (hash, author, date, message)
        for i in range(0, len(log_output), 4):
            if i + 3 < len(log_output) and len(log) < max_count:
                log.append(
                    f"Commit: {log_output[i]}\n"
                    f"Author: {log_output[i+1]}\n"
                    f"Date: {log_output[i+2]}\n"
                    f"Message: {log_output[i+3]}\n"
                )
        return log
    else:
        # Use existing logic for simple log without date filtering
        commits = list(repo.iter_commits(max_count=max_count))
        log = []
        for commit in commits:
            log.append(
                f"Commit: {commit.hexsha!r}\n"
                f"Author: {commit.author!r}\n"
                f"Date: {commit.authored_datetime}\n"
                f"Message: {commit.message!r}\n"
            )
        return log

def git_create_branch(repo: git.Repo, branch_name: str, base_branch: str | None = None) -> str:
    # Defense in depth: reject names starting with '-' to prevent flag injection
    if branch_name.startswith("-"):
        raise BadName(f"Invalid branch name: '{branch_name}' - cannot start with '-'")
    if base_branch and base_branch.startswith("-"):
        raise BadName(f"Invalid base branch: '{base_branch}' - cannot start with '-'")
    if base_branch:
        base = repo.references[base_branch]
    else:
        base = repo.active_branch

    repo.create_head(branch_name, base)
    return f"Created branch '{branch_name}' from '{base.name}'"

def git_checkout(repo: git.Repo, branch_name: str) -> str:
    # Defense in depth: reject branch names starting with '-' to prevent flag injection,
    # even if a malicious ref with that name exists (e.g. via filesystem manipulation)
    if branch_name.startswith("-"):
        raise BadName(f"Invalid branch name: '{branch_name}' - cannot start with '-'")
    repo.rev_parse(branch_name)  # Validates branch_name is a real git ref, throws BadName if not
    repo.git.checkout(branch_name)
    return f"Switched to branch '{branch_name}'"



def git_show(repo: git.Repo, revision: str) -> str:
    # Defense in depth: reject revisions starting with '-' to prevent flag injection,
    # even if a malicious ref with that name exists (e.g. via filesystem manipulation)
    if revision.startswith("-"):
        raise BadName(f"Invalid revision: '{revision}' - cannot start with '-'")
    commit = repo.commit(revision)
    output = [
        f"Commit: {commit.hexsha!r}\n"
        f"Author: {commit.author!r}\n"
        f"Date: {commit.authored_datetime!r}\n"
        f"Message: {commit.message!r}\n"
    ]
    if commit.parents:
        parent = commit.parents[0]
        diff = parent.diff(commit, create_patch=True)
    else:
        diff = commit.diff(git.NULL_TREE, create_patch=True)
    for d in diff:
        output.append(f"\n--- {d.a_path}\n+++ {d.b_path}\n")
        if d.diff is None:
            continue
        if isinstance(d.diff, bytes):
            output.append(d.diff.decode('utf-8'))
        else:
            output.append(d.diff)
    return "".join(output)

def validate_repo_path(repo_path: Path, allowed_repository: Path | None) -> None:
    """Validate that repo_path is within the allowed repository path."""
    if allowed_repository is None:
        return  # No restriction configured

    # Resolve both paths to handle symlinks and relative paths
    try:
        resolved_repo = repo_path.resolve()
        resolved_allowed = allowed_repository.resolve()
    except (OSError, RuntimeError):
        raise ValueError(f"Invalid path: {repo_path}")

    # Check if repo_path is the same as or a subdirectory of allowed_repository
    try:
        resolved_repo.relative_to(resolved_allowed)
    except ValueError:
        raise ValueError(
            f"Repository path '{repo_path}' is outside the allowed repository '{allowed_repository}'"
        )


def git_branch(repo: git.Repo, branch_type: str, contains: str | None = None, not_contains: str | None = None) -> str:
    # Defense in depth: reject values starting with '-' to prevent flag injection
    if contains and contains.startswith("-"):
        raise BadName(f"Invalid contains value: '{contains}' - cannot start with '-'")
    if not_contains and not_contains.startswith("-"):
        raise BadName(f"Invalid not_contains value: '{not_contains}' - cannot start with '-'")

    match contains:
        case None:
            contains_sha = (None,)
        case _:
            contains_sha = ("--contains", contains)

    match not_contains:
        case None:
            not_contains_sha = (None,)
        case _:
            not_contains_sha = ("--no-contains", not_contains)

    match branch_type:
        case 'local':
            b_type = None
        case 'remote':
            b_type = "-r"
        case 'all':
            b_type = "-a"
        case _:
            return f"Invalid branch type: {branch_type}"

    # None value will be auto deleted by GitPython
    branch_info = repo.git.branch(b_type, *contains_sha, *not_contains_sha)

    return branch_info


async def serve(repository: Path | None) -> None:
    logger = logging.getLogger(__name__)

    if repository is not None:
        try:
            git.Repo(repository)
            logger.info(f"Using repository at {repository}")
        except git.InvalidGitRepositoryError:
            logger.error(f"{repository} is not a valid Git repository")
            return

    server = Server("mcp-git")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=GitTools.STATUS,
                description="Shows the working tree status",
                inputSchema=GitStatus.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.DIFF_UNSTAGED,
                description="Shows changes in the working directory that are not yet staged",
                inputSchema=GitDiffUnstaged.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.DIFF_STAGED,
                description="Shows changes that are staged for commit",
                inputSchema=GitDiffStaged.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.DIFF,
                description="Shows differences between branches or commits",
                inputSchema=GitDiff.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.COMMIT,
                description="Records changes to the repository",
                inputSchema=GitCommit.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.ADD,
                description="Adds file contents to the staging area",
                inputSchema=GitAdd.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.RESET,
                description="Unstages all staged changes",
                inputSchema=GitReset.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.LOG,
                description="Shows the commit logs",
                inputSchema=GitLog.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.CREATE_BRANCH,
                description="Creates a new branch from an optional base branch",
                inputSchema=GitCreateBranch.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.CHECKOUT,
                description="Switches branches",
                inputSchema=GitCheckout.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.SHOW,
                description="Shows the contents of a commit",
                inputSchema=GitShow.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            ),
            Tool(
                name=GitTools.BRANCH,
                description="List Git branches",
                inputSchema=GitBranch.model_json_schema(),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]

    async def list_repos() -> Sequence[str]:
        async def by_roots() -> Sequence[str]:
            if not isinstance(server.request_context.session, ServerSession):
                raise TypeError("server.request_context.session must be a ServerSession")

            if not server.request_context.session.check_client_capability(
                ClientCapabilities(roots=RootsCapability())
            ):
                return []

            roots_result: ListRootsResult = await server.request_context.session.list_roots()
            logger.debug(f"Roots result: {roots_result}")
            repo_paths = []
            for root in roots_result.roots:
                path = root.uri.path
                try:
                    git.Repo(path)
                    repo_paths.append(str(path))
                except git.InvalidGitRepositoryError:
                    pass
            return repo_paths

        def by_commandline() -> Sequence[str]:
            return [str(repository)] if repository is not None else []

        cmd_repos = by_commandline()
        root_repos = await by_roots()
        return [*root_repos, *cmd_repos]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        repo_path = Path(arguments["repo_path"])

        # Validate repo_path is within allowed repository
        validate_repo_path(repo_path, repository)

        # For all commands, we need an existing repo
        repo = git.Repo(repo_path)

        match name:
            case GitTools.STATUS:
                status = git_status(repo)
                return [TextContent(
                    type="text",
                    text=f"Repository status:\n{status}"
                )]

            case GitTools.DIFF_UNSTAGED:
                diff = git_diff_unstaged(
                    repo,
                    arguments.get("context_lines", DEFAULT_CONTEXT_LINES),
                    arguments.get("paths"),
                    arguments.get("stat", True),
                    arguments.get("diff", True),
                )
                auto_chunk = arguments.get("auto_chunk", True)
                continuation_token = arguments.get("continuation_token")
                if auto_chunk:
                    chunk_text, metadata = chunk_output(diff, continuation_token=continuation_token)
                    result = f"Unstaged changes:\n{chunk_text}"
                    if metadata:
                        result += f"\n\n{metadata}"
                    return [TextContent(type="text", text=result)]
                return [TextContent(type="text", text=f"Unstaged changes:\n{diff}")]

            case GitTools.DIFF_STAGED:
                diff = git_diff_staged(
                    repo,
                    arguments.get("context_lines", DEFAULT_CONTEXT_LINES),
                    arguments.get("paths"),
                    arguments.get("stat", True),
                    arguments.get("diff", True),
                )
                auto_chunk = arguments.get("auto_chunk", True)
                continuation_token = arguments.get("continuation_token")
                if auto_chunk:
                    chunk_text, metadata = chunk_output(diff, continuation_token=continuation_token)
                    result = f"Staged changes:\n{chunk_text}"
                    if metadata:
                        result += f"\n\n{metadata}"
                    return [TextContent(type="text", text=result)]
                return [TextContent(type="text", text=f"Staged changes:\n{diff}")]

            case GitTools.DIFF:
                diff = git_diff(
                    repo,
                    arguments["target"],
                    arguments.get("context_lines", DEFAULT_CONTEXT_LINES),
                    arguments.get("paths"),
                    arguments.get("stat", True),
                    arguments.get("diff", True),
                )
                auto_chunk = arguments.get("auto_chunk", True)
                continuation_token = arguments.get("continuation_token")
                if auto_chunk:
                    chunk_text, metadata = chunk_output(diff, continuation_token=continuation_token)
                    result = f"Diff with {arguments['target']}:\n{chunk_text}"
                    if metadata:
                        result += f"\n\n{metadata}"
                    return [TextContent(type="text", text=result)]
                return [TextContent(type="text", text=f"Diff with {arguments['target']}:\n{diff}")]

            case GitTools.COMMIT:
                result = git_commit(repo, arguments["message"])
                return [TextContent(
                    type="text",
                    text=result
                )]

            case GitTools.ADD:
                result = git_add(repo, arguments["files"])
                return [TextContent(
                    type="text",
                    text=result
                )]

            case GitTools.RESET:
                result = git_reset(repo)
                return [TextContent(
                    type="text",
                    text=result
                )]

            # Update the LOG case:
            case GitTools.LOG:
                log = git_log(
                    repo,
                    arguments.get("max_count", 10),
                    arguments.get("start_timestamp"),
                    arguments.get("end_timestamp")
                )
                return [TextContent(
                    type="text",
                    text="Commit history:\n" + "\n".join(log)
                )]

            case GitTools.CREATE_BRANCH:
                result = git_create_branch(
                    repo,
                    arguments["branch_name"],
                    arguments.get("base_branch")
                )
                return [TextContent(
                    type="text",
                    text=result
                )]

            case GitTools.CHECKOUT:
                result = git_checkout(repo, arguments["branch_name"])
                return [TextContent(
                    type="text",
                    text=result
                )]

            case GitTools.SHOW:
                result = git_show(repo, arguments["revision"])
                auto_chunk = arguments.get("auto_chunk", True)
                continuation_token = arguments.get("continuation_token")
                if auto_chunk:
                    chunk_text, metadata = chunk_output(result, continuation_token=continuation_token)
                    output = chunk_text
                    if metadata:
                        output += f"\n\n{metadata}"
                    return [TextContent(type="text", text=output)]
                return [TextContent(type="text", text=result)]

            case GitTools.BRANCH:
                result = git_branch(
                    repo,
                    arguments.get("branch_type", 'local'),
                    arguments.get("contains", None),
                    arguments.get("not_contains", None),
                )
                return [TextContent(
                    type="text",
                    text=result
                )]

            case _:
                raise ValueError(f"Unknown tool: {name}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)