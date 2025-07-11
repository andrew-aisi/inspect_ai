import functools
from typing import Any, Callable, Literal, cast

import click
from typing_extensions import Unpack

from inspect_ai import Epochs, eval, eval_retry
from inspect_ai._eval.evalset import eval_set
from inspect_ai._util.config import resolve_args
from inspect_ai._util.constants import (
    ALL_LOG_LEVELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LOG_LEVEL_TRANSCRIPT,
    DEFAULT_LOG_SHARED,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_RETRY_ON_ERROR,
)
from inspect_ai._util.file import filesystem
from inspect_ai._util.samples import parse_sample_id, parse_samples_limit
from inspect_ai.log._file import log_file_info
from inspect_ai.model import GenerateConfigArgs
from inspect_ai.model._generate_config import BatchConfig, ResponseSchema
from inspect_ai.scorer._reducer import create_reducers
from inspect_ai.solver._solver import SolverSpec

from .common import (
    CommonOptions,
    common_options,
    process_common_options,
)
from .util import (
    int_bool_or_str_flag_callback,
    int_or_bool_flag_callback,
    parse_cli_args,
    parse_cli_config,
    parse_sandbox,
)

MAX_SAMPLES_HELP = "Maximum number of samples to run in parallel (default is running all samples in parallel)"
MAX_TASKS_HELP = "Maximum number of tasks to run in parallel (default is 1 for eval and 4 for eval-set)"
MAX_SUBPROCESSES_HELP = (
    "Maximum number of subprocesses to run in parallel (default is os.cpu_count())"
)
MAX_SANDBOXES_HELP = "Maximum number of sandboxes (per-provider) to run in parallel."
NO_SANDBOX_CLEANUP_HELP = "Do not cleanup sandbox environments after task completes"
FAIL_ON_ERROR_HELP = "Threshold of sample errors to tolerage (by default, evals fail when any error occurs). Value between 0 to 1 to set a proportion; value greater than 1 to set a count."
NO_LOG_SAMPLES_HELP = "Do not include samples in the log file."
NO_LOG_REALTIME_HELP = (
    "Do not log events in realtime (affects live viewing of samples in inspect view)"
)
NO_FAIL_ON_ERROR_HELP = "Do not fail the eval if errors occur within samples (instead, continue running other samples)"
RETRY_ON_ERROR_HELP = "Retry samples if they encounter errors (by default, no retries occur). Specify --retry-on-error to retry a single time, or specify e.g. `--retry-on-error=3` to retry multiple times."
LOG_IMAGES_HELP = (
    "Include base64 encoded versions of filename or URL based images in the log file."
)
LOG_BUFFER_HELP = "Number of samples to buffer before writing log file. If not specified, an appropriate default for the format and filesystem is chosen (10 for most all cases, 100 for JSON logs on remote filesystems)."
LOG_SHARED_HELP = "Sync sample events to log directory so that users on other systems can see log updates in realtime (defaults to no syncing). If enabled will sync every 10 seconds (or pass a value to sync every `n` seconds)."
NO_SCORE_HELP = (
    "Do not score model output (use the inspect score command to score output later)"
)
NO_SCORE_DISPLAY = "Do not display scoring metrics in realtime."
MAX_CONNECTIONS_HELP = f"Maximum number of concurrent connections to Model API (defaults to {DEFAULT_MAX_CONNECTIONS})"
MAX_RETRIES_HELP = (
    "Maximum number of times to retry model API requests (defaults to unlimited)"
)
TIMEOUT_HELP = "Model API request timeout in seconds (defaults to no timeout)"
BATCH_HELP = "Batch requests together to reduce API calls when using a model that supports batching (by default, no batching). Specify --batch to batch with default configuration,  specify a batch size e.g. `--batch=1000` to configure batches of 1000 requests, or pass the file path to a YAML or JSON config file with batch configuration."


def eval_options(func: Callable[..., Any]) -> Callable[..., click.Context]:
    @click.option(
        "--model",
        type=str,
        help="Model used to evaluate tasks.",
        envvar="INSPECT_EVAL_MODEL",
    )
    @click.option(
        "--model-base-url",
        type=str,
        help="Base URL for for model API",
    )
    @click.option(
        "-M",
        multiple=True,
        type=str,
        envvar="INSPECT_EVAL_MODEL_ARGS",
        help="One or more native model arguments (e.g. -M arg=value)",
    )
    @click.option(
        "--model-config",
        type=str,
        envvar="INSPECT_EVAL_MODEL_CONFIG",
        help="YAML or JSON config file with model arguments.",
    )
    @click.option(
        "--model-role",
        multiple=True,
        type=str,
        envvar="INSPECT_EVAL_MODEL_ROLE",
        help="Named model role, e.g. --model-role critic=openai/gpt-4o",
    )
    @click.option(
        "-T",
        multiple=True,
        type=str,
        envvar="INSPECT_EVAL_TASK_ARGS",
        help="One or more task arguments (e.g. -T arg=value)",
    )
    @click.option(
        "--task-config",
        type=str,
        envvar="INSPECT_EVAL_TASK_CONFIG",
        help="YAML or JSON config file with task arguments.",
    )
    @click.option(
        "--solver",
        type=str,
        envvar="INSPECT_EVAL_SOLVER",
        help="Solver to execute (overrides task default solver)",
    )
    @click.option(
        "-S",
        multiple=True,
        type=str,
        envvar="INSPECT_EVAL_SOLVER_ARGS",
        help="One or more solver arguments (e.g. -S arg=value)",
    )
    @click.option(
        "--solver-config",
        type=str,
        envvar="INSPECT_EVAL_SOLVER_CONFIG",
        help="YAML or JSON config file with solver arguments.",
    )
    @click.option(
        "--tags",
        type=str,
        help="Tags to associate with this evaluation run.",
        envvar="INSPECT_EVAL_TAGS",
    )
    @click.option(
        "--metadata",
        multiple=True,
        type=str,
        help="Metadata to associate with this evaluation run (more than one --metadata argument can be specified).",
        envvar="INSPECT_EVAL_METADATA",
    )
    @click.option(
        "--trace",
        type=bool,
        is_flag=True,
        hidden=True,
        envvar="INSPECT_EVAL_TRACE",
        help="Trace message interactions with evaluated model to terminal.",
    )
    @click.option(
        "--approval",
        type=str,
        envvar="INSPECT_EVAL_APPROVAL",
        help="Config file for tool call approval.",
    )
    @click.option(
        "--sandbox",
        type=str,
        help="Sandbox environment type (with optional config file). e.g. 'docker' or 'docker:compose.yml'",
        envvar="INSPECT_EVAL_SANDBOX",
    )
    @click.option(
        "--no-sandbox-cleanup",
        type=bool,
        is_flag=True,
        help=NO_SANDBOX_CLEANUP_HELP,
        envvar="INSPECT_EVAL_NO_SANDBOX_CLEANUP",
    )
    @click.option(
        "--limit",
        type=str,
        help="Limit samples to evaluate e.g. 10 or 10-20",
        envvar="INSPECT_EVAL_LIMIT",
    )
    @click.option(
        "--sample-id",
        type=str,
        help="Evaluate specific sample(s) (comma separated list of ids)",
        envvar="INSPECT_EVAL_SAMPLE_ID",
    )
    @click.option(
        "--epochs",
        type=int,
        help=f"Number of times to repeat dataset (defaults to {DEFAULT_EPOCHS}) ",
        envvar="INSPECT_EVAL_EPOCHS",
    )
    @click.option(
        "--epochs-reducer",
        type=str,
        help="Method for reducing per-epoch sample scores into a single score. Built in reducers include 'mean', 'median', 'mode', 'max', and 'at_least_{n}'.",
        envvar="INSPECT_EVAL_EPOCHS_REDUCER",
    )
    @click.option(
        "--max-connections",
        type=int,
        help=MAX_CONNECTIONS_HELP,
        envvar="INSPECT_EVAL_MAX_CONNECTIONS",
    )
    @click.option(
        "--max-retries",
        type=int,
        help=MAX_RETRIES_HELP,
        envvar="INSPECT_EVAL_MAX_RETRIES",
    )
    @click.option(
        "--timeout", type=int, help=TIMEOUT_HELP, envvar="INSPECT_EVAL_TIMEOUT"
    )
    @click.option(
        "--max-samples",
        type=int,
        help=MAX_SAMPLES_HELP,
        envvar="INSPECT_EVAL_MAX_SAMPLES",
    )
    @click.option(
        "--max-tasks", type=int, help=MAX_TASKS_HELP, envvar="INSPECT_EVAL_MAX_TASKS"
    )
    @click.option(
        "--max-subprocesses",
        type=int,
        help=MAX_SUBPROCESSES_HELP,
        envvar="INSPECT_EVAL_MAX_SUBPROCESSES",
    )
    @click.option(
        "--max-sandboxes",
        type=int,
        help=MAX_SANDBOXES_HELP,
        envvar="INSPECT_EVAL_MAX_SANDBOXES",
    )
    @click.option(
        "--message-limit",
        type=int,
        help="Limit on total messages used for each sample.",
        envvar="INSPECT_EVAL_MESSAGE_LIMIT",
    )
    @click.option(
        "--token-limit",
        type=int,
        help="Limit on total tokens used for each sample.",
        envvar="INSPECT_EVAL_TOKEN_LIMIT",
    )
    @click.option(
        "--time-limit",
        type=int,
        help="Limit on total running time for each sample.",
        envvar="INSPECT_EVAL_TIME_LIMIT",
    )
    @click.option(
        "--working-limit",
        type=int,
        help="Limit on total working time (e.g. model generation, tool calls, etc.) for each sample.",
        envvar="INSPECT_EVAL_WORKING_LIMIT",
    )
    @click.option(
        "--fail-on-error",
        type=float,
        is_flag=False,
        flag_value=0.0,
        help=FAIL_ON_ERROR_HELP,
        envvar="INSPECT_EVAL_FAIL_ON_ERROR",
    )
    @click.option(
        "--no-fail-on-error",
        type=bool,
        is_flag=True,
        default=False,
        help=NO_FAIL_ON_ERROR_HELP,
        envvar="INSPECT_EVAL_NO_FAIL_ON_ERROR",
    )
    @click.option(
        "--retry-on-error",
        is_flag=False,
        flag_value="true",
        default=None,
        callback=int_or_bool_flag_callback(DEFAULT_RETRY_ON_ERROR),
        help=RETRY_ON_ERROR_HELP,
        envvar="INSPECT_EVAL_RETRY_ON_ERROR",
    )
    @click.option(
        "--no-log-samples",
        type=bool,
        is_flag=True,
        help=NO_LOG_SAMPLES_HELP,
        envvar="INSPECT_EVAL_NO_LOG_SAMPLES",
    )
    @click.option(
        "--no-log-realtime",
        type=bool,
        is_flag=True,
        help=NO_LOG_REALTIME_HELP,
        envvar="INSPECT_EVAL_NO_LOG_REALTIME",
    )
    @click.option(
        "--log-images/--no-log-images",
        type=bool,
        default=True,
        is_flag=True,
        help=LOG_IMAGES_HELP,
    )
    @click.option(
        "--log-buffer", type=int, help=LOG_BUFFER_HELP, envvar="INSPECT_EVAL_LOG_BUFFER"
    )
    @click.option(
        "--log-shared",
        is_flag=False,
        flag_value="true",
        default=None,
        callback=int_or_bool_flag_callback(DEFAULT_LOG_SHARED),
        help=LOG_SHARED_HELP,
        envvar=["INSPECT_LOG_SHARED", "INSPECT_EVAL_LOG_SHARED"],
    )
    @click.option(
        "--no-score",
        type=bool,
        is_flag=True,
        help=NO_SCORE_HELP,
        envvar="INSPECT_EVAL_NO_SCORE",
    )
    @click.option(
        "--no-score-display",
        type=bool,
        is_flag=True,
        help=NO_SCORE_HELP,
        envvar="INSPECT_EVAL_SCORE_DISPLAY",
    )
    @click.option(
        "--max-tokens",
        type=int,
        help="The maximum number of tokens that can be generated in the completion (default is model specific)",
        envvar="INSPECT_EVAL_MAX_TOKENS",
    )
    @click.option(
        "--system-message",
        type=str,
        help="Override the default system message.",
        envvar="INSPECT_EVAL_SYSTEM_MESSAGE",
    )
    @click.option(
        "--best-of",
        type=int,
        help="Generates best_of completions server-side and returns the 'best' (the one with the highest log probability per token). OpenAI only.",
        envvar="INSPECT_EVAL_BEST_OF",
    )
    @click.option(
        "--frequency-penalty",
        type=float,
        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. OpenAI, Google, Grok, Groq, llama-cpp-python and vLLM only.",
        envvar="INSPECT_EVAL_FREQUENCY_PENALTY",
    )
    @click.option(
        "--presence-penalty",
        type=float,
        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. OpenAI, Google, Grok, Groq, llama-cpp-python and vLLM only.",
        envvar="INSPECT_EVAL_PRESENCE_PENALTY",
    )
    @click.option(
        "--logit-bias",
        type=str,
        help='Map token Ids to an associated bias value from -100 to 100 (e.g. "42=10,43=-10"). OpenAI, Grok, and Grok only.',
        envvar="INSPECT_EVAL_LOGIT_BIAS",
    )
    @click.option(
        "--seed",
        type=int,
        help="Random seed. OpenAI, Google, Groq, Mistral, HuggingFace, and vLLM only.",
        envvar="INSPECT_EVAL_SEED",
    )
    @click.option(
        "--stop-seqs",
        type=str,
        help="Sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.",
        envvar="INSPECT_EVAL_STOP_SEQS",
    )
    @click.option(
        "--temperature",
        type=float,
        help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
        envvar="INSPECT_EVAL_TEMPERATURE",
    )
    @click.option(
        "--top-p",
        type=float,
        help="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.",
        envvar="INSPECT_EVAL_TOP_P",
    )
    @click.option(
        "--top-k",
        type=int,
        help="Randomly sample the next word from the top_k most likely next words. Anthropic, Google, HuggingFace, and vLLM only.",
        envvar="INSPECT_EVAL_TOP_K",
    )
    @click.option(
        "--num-choices",
        type=int,
        help="How many chat completion choices to generate for each input message. OpenAI, Grok, Google, TogetherAI, and vLLM only.",
        envvar="INSPECT_EVAL_NUM_CHOICES",
    )
    @click.option(
        "--logprobs",
        type=bool,
        is_flag=True,
        help="Return log probabilities of the output tokens. OpenAI, Grok, TogetherAI, Huggingface, llama-cpp-python, and vLLM only.",
        envvar="INSPECT_EVAL_LOGPROBS",
    )
    @click.option(
        "--top-logprobs",
        type=int,
        help="Number of most likely tokens (0-20) to return at each token position, each with an associated log probability. OpenAI, Grok, TogetherAI, Huggingface, and vLLM only.",
        envvar="INSPECT_EVAL_TOP_LOGPROBS",
    )
    @click.option(
        "--parallel-tool-calls/--no-parallel-tool-calls",
        type=bool,
        is_flag=True,
        default=True,
        help="Whether to enable parallel function calling during tool use (defaults to True) OpenAI and Groq only.",
        envvar="INSPECT_EVAL_PARALLEL_TOOL_CALLS",
    )
    @click.option(
        "--internal-tools/--no-internal-tools",
        type=bool,
        is_flag=True,
        default=True,
        help="Whether to automatically map tools to model internal implementations (e.g. 'computer' for anthropic).",
        envvar="INSPECT_EVAL_INTERNAL_TOOLS",
    )
    @click.option(
        "--max-tool-output",
        type=int,
        help="Maximum size of tool output (in bytes). Defaults to 16 * 1024.",
        envvar="INSPECT_EVAL_MAX_TOOL_OUTPUT",
    )
    @click.option(
        "--cache-prompt",
        type=click.Choice(["auto", "true", "false"]),
        help='Cache prompt prefix (Anthropic only). Defaults to "auto", which will enable caching for requests with tools.',
        envvar="INSPECT_EVAL_CACHE_PROMPT",
    )
    @click.option(
        "--reasoning-effort",
        type=click.Choice(["low", "medium", "high"]),
        help="Constrains effort on reasoning for reasoning models (defaults to `medium`). Open AI o-series models only.",
        envvar="INSPECT_EVAL_REASONING_EFFORT",
    )
    @click.option(
        "--reasoning-tokens",
        type=int,
        help="Maximum number of tokens to use for reasoning. Anthropic Claude models only.",
        envvar="INSPECT_EVAL_REASONING_TOKENS",
    )
    @click.option(
        "--reasoning-summary",
        type=click.Choice(["concise", "detailed", "auto"]),
        help="Provide summary of reasoning steps (defaults to no summary). Use 'auto' to access the most detailed summarizer available for the current model. OpenAI reasoning models only.",
        envvar="INSPECT_EVAL_REASONING_SUMMARY",
    )
    @click.option(
        "--reasoning-history",
        type=click.Choice(["none", "all", "last", "auto"]),
        help='Include reasoning in chat message history sent to generate (defaults to "auto", which uses the recommended default for each provider)',
        envvar="INSPECT_EVAL_REASONING_HISTORY",
    )
    @click.option(
        "--response-schema",
        type=str,
        help="JSON schema for desired response format (output should still be validated). OpenAI, Google, and Mistral only.",
        envvar="INSPECT_EVAL_RESPONSE_SCHEMA",
    )
    @click.option(
        "--batch",
        is_flag=False,
        flag_value="true",
        default=None,
        callback=int_bool_or_str_flag_callback(DEFAULT_BATCH_SIZE, None),
        help=BATCH_HELP,
        envvar="INSPECT_EVAL_BATCH",
    )
    @click.option(
        "--log-format",
        type=click.Choice(["eval", "json"], case_sensitive=False),
        envvar=["INSPECT_LOG_FORMAT", "INSPECT_EVAL_LOG_FORMAT"],
        help="Format for writing log files.",
    )
    @click.option(
        "--log-level-transcript",
        type=click.Choice(
            [level.lower() for level in ALL_LOG_LEVELS],
            case_sensitive=False,
        ),
        default=DEFAULT_LOG_LEVEL_TRANSCRIPT,
        envvar="INSPECT_LOG_LEVEL_TRANSCRIPT",
        help=f"Set the log level of the transcript (defaults to '{DEFAULT_LOG_LEVEL_TRANSCRIPT}')",
    )
    @common_options
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> click.Context:
        return cast(click.Context, func(*args, **kwargs))

    return wrapper


@click.command("eval")
@click.argument("tasks", nargs=-1)
@eval_options
def eval_command(
    tasks: tuple[str] | None,
    solver: str | None,
    model: str | None,
    model_base_url: str | None,
    m: tuple[str] | None,
    model_config: str | None,
    model_role: tuple[str] | None,
    t: tuple[str] | None,
    task_config: str | None,
    s: tuple[str] | None,
    solver_config: str | None,
    tags: str | None,
    metadata: tuple[str] | None,
    trace: bool | None,
    approval: str | None,
    sandbox: str | None,
    no_sandbox_cleanup: bool | None,
    epochs: int | None,
    epochs_reducer: str | None,
    limit: str | None,
    sample_id: str | None,
    max_retries: int | None,
    timeout: int | None,
    max_connections: int | None,
    max_tokens: int | None,
    system_message: str | None,
    best_of: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    logit_bias: str | None,
    seed: int | None,
    stop_seqs: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    num_choices: int | None,
    logprobs: bool | None,
    top_logprobs: int | None,
    parallel_tool_calls: bool | None,
    internal_tools: bool | None,
    max_tool_output: int | None,
    cache_prompt: str | None,
    reasoning_effort: str | None,
    reasoning_tokens: int | None,
    reasoning_summary: Literal["concise", "detailed", "auto"] | None,
    reasoning_history: Literal["none", "all", "last", "auto"] | None,
    response_schema: ResponseSchema | None,
    batch: int | str | None,
    message_limit: int | None,
    token_limit: int | None,
    time_limit: int | None,
    working_limit: int | None,
    max_samples: int | None,
    max_tasks: int | None,
    max_subprocesses: int | None,
    max_sandboxes: int | None,
    fail_on_error: bool | float | None,
    no_fail_on_error: bool | None,
    retry_on_error: int | None,
    no_log_samples: bool | None,
    no_log_realtime: bool | None,
    log_images: bool | None,
    log_buffer: int | None,
    log_shared: int | None,
    no_score: bool | None,
    no_score_display: bool | None,
    log_format: Literal["eval", "json"] | None,
    log_level_transcript: str,
    **common: Unpack[CommonOptions],
) -> None:
    """Evaluate tasks."""
    # read config
    config = config_from_locals(dict(locals()))

    # resolve common options
    process_common_options(common)

    # exec eval
    eval_exec(
        tasks=tasks,
        solver=solver,
        log_level=common["log_level"],
        log_level_transcript=log_level_transcript,
        log_dir=common["log_dir"],
        log_format=log_format,
        model=model,
        model_base_url=model_base_url,
        m=m,
        model_config=model_config,
        model_role=model_role,
        t=t,
        task_config=task_config,
        s=s,
        solver_config=solver_config,
        tags=tags,
        metadata=metadata,
        trace=trace,
        approval=approval,
        sandbox=sandbox,
        no_sandbox_cleanup=no_sandbox_cleanup,
        epochs=epochs,
        epochs_reducer=epochs_reducer,
        limit=limit,
        sample_id=sample_id,
        message_limit=message_limit,
        token_limit=token_limit,
        time_limit=time_limit,
        working_limit=working_limit,
        max_samples=max_samples,
        max_tasks=max_tasks,
        max_subprocesses=max_subprocesses,
        max_sandboxes=max_sandboxes,
        fail_on_error=fail_on_error,
        no_fail_on_error=no_fail_on_error,
        retry_on_error=retry_on_error,
        debug_errors=common["debug_errors"],
        no_log_samples=no_log_samples,
        no_log_realtime=no_log_realtime,
        log_images=log_images,
        log_buffer=log_buffer,
        log_shared=log_shared,
        no_score=no_score,
        no_score_display=no_score_display,
        is_eval_set=False,
        **config,
    )


@click.command("eval-set")
@click.argument("tasks", nargs=-1)
@click.option(
    "--retry-attempts",
    type=int,
    help="Maximum number of retry attempts before giving up (defaults to 10).",
    envvar="INSPECT_EVAL_RETRY_ATTEMPS",
)
@click.option(
    "--retry-wait",
    type=int,
    help="Time in seconds wait between attempts, increased exponentially. "
    + "(defaults to 30, resulting in waits of 30, 60, 120, 240, etc.). Wait time "
    + "per-retry will in no case by longer than 1 hour.",
    envvar="INSPECT_EVAL_RETRY_WAIT",
)
@click.option(
    "--retry-connections",
    type=float,
    help="Reduce max_connections at this rate with each retry (defaults to 1.0, which results in no reduction).",
    envvar="INSPECT_EVAL_RETRY_CONNECTIONS",
)
@click.option(
    "--no-retry-cleanup",
    type=bool,
    is_flag=True,
    help="Do not cleanup failed log files after retries",
    envvar="INSPECT_EVAL_NO_RETRY_CLEANUP",
)
@click.option(
    "--bundle-dir",
    type=str,
    is_flag=False,
    help="Bundle viewer and logs into output directory",
)
@click.option(
    "--bundle-overwrite",
    type=str,
    is_flag=True,
    help="Overwrite existing bundle dir.",
)
@eval_options
@click.pass_context
def eval_set_command(
    ctx: click.Context,
    tasks: tuple[str] | None,
    retry_attempts: int | None,
    retry_wait: int | None,
    retry_connections: float | None,
    no_retry_cleanup: bool | None,
    solver: str | None,
    trace: bool | None,
    approval: str | None,
    model: str | None,
    model_base_url: str | None,
    m: tuple[str] | None,
    model_config: str | None,
    model_role: tuple[str] | None,
    t: tuple[str] | None,
    task_config: str | None,
    s: tuple[str] | None,
    solver_config: str | None,
    tags: str | None,
    metadata: tuple[str] | None,
    sandbox: str | None,
    no_sandbox_cleanup: bool | None,
    epochs: int | None,
    epochs_reducer: str | None,
    limit: str | None,
    sample_id: str | None,
    max_retries: int | None,
    timeout: int | None,
    max_connections: int | None,
    max_tokens: int | None,
    system_message: str | None,
    best_of: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    logit_bias: str | None,
    seed: int | None,
    stop_seqs: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    num_choices: int | None,
    logprobs: bool | None,
    top_logprobs: int | None,
    parallel_tool_calls: bool | None,
    internal_tools: bool | None,
    max_tool_output: int | None,
    cache_prompt: str | None,
    reasoning_effort: str | None,
    reasoning_tokens: int | None,
    reasoning_summary: Literal["concise", "detailed", "auto"] | None,
    reasoning_history: Literal["none", "all", "last", "auto"] | None,
    response_schema: ResponseSchema | None,
    batch: int | str | None,
    message_limit: int | None,
    token_limit: int | None,
    time_limit: int | None,
    working_limit: int | None,
    max_samples: int | None,
    max_tasks: int | None,
    max_subprocesses: int | None,
    max_sandboxes: int | None,
    fail_on_error: bool | float | None,
    no_fail_on_error: bool | None,
    retry_on_error: int | None,
    no_log_samples: bool | None,
    no_log_realtime: bool | None,
    log_images: bool | None,
    log_buffer: int | None,
    log_shared: int | None,
    no_score: bool | None,
    no_score_display: bool | None,
    bundle_dir: str | None,
    bundle_overwrite: bool | None,
    log_format: Literal["eval", "json"] | None,
    log_level_transcript: str,
    **common: Unpack[CommonOptions],
) -> int:
    """Evaluate a set of tasks with retries.

    Learn more about eval sets at https://inspect.aisi.org.uk/eval-sets.html.
    """
    # read config
    config = config_from_locals(dict(locals()))

    # resolve common options
    process_common_options(common)

    # exec eval
    success = eval_exec(
        tasks=tasks,
        solver=solver,
        log_level=common["log_level"],
        log_level_transcript=log_level_transcript,
        log_dir=common["log_dir"],
        log_format=log_format,
        model=model,
        model_base_url=model_base_url,
        m=m,
        model_config=model_config,
        model_role=model_role,
        t=t,
        task_config=task_config,
        s=s,
        solver_config=solver_config,
        tags=tags,
        metadata=metadata,
        trace=trace,
        approval=approval,
        sandbox=sandbox,
        no_sandbox_cleanup=no_sandbox_cleanup,
        epochs=epochs,
        epochs_reducer=epochs_reducer,
        limit=limit,
        sample_id=sample_id,
        message_limit=message_limit,
        token_limit=token_limit,
        time_limit=time_limit,
        working_limit=working_limit,
        max_samples=max_samples,
        max_tasks=max_tasks,
        max_subprocesses=max_subprocesses,
        max_sandboxes=max_sandboxes,
        fail_on_error=fail_on_error,
        no_fail_on_error=no_fail_on_error,
        retry_on_error=retry_on_error,
        debug_errors=common["debug_errors"],
        no_log_samples=no_log_samples,
        no_log_realtime=no_log_realtime,
        log_images=log_images,
        log_buffer=log_buffer,
        log_shared=log_shared,
        no_score=no_score,
        no_score_display=no_score_display,
        is_eval_set=True,
        retry_attempts=retry_attempts,
        retry_wait=retry_wait,
        retry_connections=retry_connections,
        retry_cleanup=not no_retry_cleanup,
        bundle_dir=bundle_dir,
        bundle_overwrite=True if bundle_overwrite else False,
        **config,
    )

    # exit code indicating whether the evals are all complete
    ctx.exit(0 if success else 1)


def eval_exec(
    tasks: tuple[str] | None,
    solver: str | None,
    log_level: str,
    log_level_transcript: str,
    log_dir: str,
    log_format: Literal["eval", "json"] | None,
    model: str | None,
    model_base_url: str | None,
    m: tuple[str] | None,
    model_config: str | None,
    model_role: tuple[str] | None,
    t: tuple[str] | None,
    task_config: str | None,
    s: tuple[str] | None,
    solver_config: str | None,
    tags: str | None,
    metadata: tuple[str] | None,
    trace: bool | None,
    approval: str | None,
    sandbox: str | None,
    no_sandbox_cleanup: bool | None,
    epochs: int | None,
    epochs_reducer: str | None,
    limit: str | None,
    sample_id: str | None,
    message_limit: int | None,
    token_limit: int | None,
    time_limit: int | None,
    working_limit: int | None,
    max_samples: int | None,
    max_tasks: int | None,
    max_subprocesses: int | None,
    max_sandboxes: int | None,
    fail_on_error: bool | float | None,
    no_fail_on_error: bool | None,
    retry_on_error: int | None,
    debug_errors: bool | None,
    no_log_samples: bool | None,
    no_log_realtime: bool | None,
    log_images: bool | None,
    log_buffer: int | None,
    log_shared: int | None,
    no_score: bool | None,
    no_score_display: bool | None,
    is_eval_set: bool = False,
    retry_attempts: int | None = None,
    retry_wait: int | None = None,
    retry_connections: float | None = None,
    retry_cleanup: bool | None = None,
    bundle_dir: str | None = None,
    bundle_overwrite: bool = False,
    **kwargs: Unpack[GenerateConfigArgs],
) -> bool:
    # parse task, solver, and model args
    task_args = parse_cli_config(t, task_config)
    solver_args = parse_cli_config(s, solver_config)
    model_args = parse_cli_config(m, model_config)

    # parse model roles
    eval_model_roles = parse_cli_args(model_role, force_str=True)

    # parse tags
    eval_tags = parse_comma_separated(tags)

    # parse metadata
    eval_metadata = parse_cli_args(metadata)

    # resolve epochs
    eval_epochs = (
        Epochs(epochs, create_reducers(parse_comma_separated(epochs_reducer)))
        if epochs
        else None
    )

    # resolve range and sample id
    eval_limit = parse_samples_limit(limit)
    eval_sample_id = parse_sample_id(sample_id)

    # resolve fail_on_error
    if no_fail_on_error is True:
        fail_on_error = False
    elif fail_on_error == 0.0:
        fail_on_error = True

    # resolve retry_on_error
    if retry_on_error == 0:
        retry_on_error = None

    # resolve negating options
    sandbox_cleanup = False if no_sandbox_cleanup else None
    log_samples = False if no_log_samples else None
    log_realtime = False if no_log_realtime else None
    log_images = False if log_images is False else None
    trace = True if trace else None
    score = False if no_score else True
    score_display = False if no_score_display else None

    # build params
    params: dict[str, Any] = (
        dict(
            tasks=list(tasks) if tasks else None,
            model=model,
            model_base_url=model_base_url,
            model_args=model_args,
            model_roles=eval_model_roles,
            task_args=task_args,
            solver=SolverSpec(solver, solver_args) if solver else None,
            tags=eval_tags,
            metadata=eval_metadata,
            trace=trace,
            approval=approval,
            sandbox=parse_sandbox(sandbox),
            sandbox_cleanup=sandbox_cleanup,
            log_level=log_level,
            log_level_transcript=log_level_transcript,
            log_dir=log_dir,
            log_format=log_format,
            limit=eval_limit,
            sample_id=eval_sample_id,
            epochs=eval_epochs,
            fail_on_error=fail_on_error,
            retry_on_error=retry_on_error,
            debug_errors=debug_errors,
            message_limit=message_limit,
            token_limit=token_limit,
            time_limit=time_limit,
            working_limit=working_limit,
            max_samples=max_samples,
            max_tasks=max_tasks,
            max_subprocesses=max_subprocesses,
            max_sandboxes=max_sandboxes,
            log_samples=log_samples,
            log_realtime=log_realtime,
            log_images=log_images,
            log_buffer=log_buffer,
            log_shared=log_shared,
            score=score,
            score_display=score_display,
        )
        | kwargs
    )

    # evaluate
    if is_eval_set:
        params["retry_attempts"] = retry_attempts
        params["retry_wait"] = retry_wait
        params["retry_connections"] = retry_connections
        params["retry_cleanup"] = retry_cleanup
        params["bundle_dir"] = bundle_dir
        params["bundle_overwrite"] = bundle_overwrite
        success, _ = eval_set(**params)
        return success
    else:
        params["log_header_only"] = True  # cli invocation doesn't need full log
        eval(**params)
        return True


def config_from_locals(locals: dict[str, Any]) -> GenerateConfigArgs:
    # build generate config
    config_keys = list(GenerateConfigArgs.__mutable_keys__)  # type: ignore
    config = GenerateConfigArgs()
    for key, value in locals.items():
        if key in config_keys and value is not None:
            if key == "stop_seqs":
                value = value.split(",")
            if key == "logprobs" and value is False:
                value = None
            if key == "logit_bias" and value is not None:
                value = parse_logit_bias(value)
            if key == "cache_prompt":
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
            if key == "parallel_tool_calls":
                if value is not False:
                    value = None
            if key == "internal_tools":
                if value is not False:
                    value = None
            if key == "reasoning_history":
                if value is not False:
                    value = None
            if key == "response_schema":
                if value is not None:
                    value = ResponseSchema.model_validate_json(value)
            if key == "batch":
                match value:
                    case str():
                        value = BatchConfig.model_validate(resolve_args(value))

            config[key] = value  # type: ignore
    return config


def parse_logit_bias(logit_bias: str | None) -> dict[int, float] | None:
    logit_biases = parse_cli_args(logit_bias.split(",")) if logit_bias else None
    if logit_biases:
        return dict(
            zip([int(key) for key in logit_biases.keys()], logit_biases.values())
        )
    else:
        return None


def parse_comma_separated(value: str | None) -> list[str] | None:
    if value is not None:
        return value.split(",")
    else:
        return None


@click.command("eval-retry")
@click.argument("log_files", nargs=-1, required=True)
@click.option(
    "--max-samples", type=int, help=MAX_SAMPLES_HELP, envvar="INSPECT_EVAL_MAX_SAMPLES"
)
@click.option(
    "--max-tasks", type=int, help=MAX_TASKS_HELP, envvar="INSPECT_EVAL_MAX_TASKS"
)
@click.option(
    "--max-subprocesses",
    type=int,
    help=MAX_SUBPROCESSES_HELP,
    envvar="INSPECT_EVAL_MAX_SUBPROCESSES",
)
@click.option(
    "--max-sandboxes",
    type=int,
    help=MAX_SANDBOXES_HELP,
    envvar="INSPECT_EVAL_MAX_SANDBOXES",
)
@click.option(
    "--no-sandbox-cleanup",
    type=bool,
    is_flag=True,
    help=NO_SANDBOX_CLEANUP_HELP,
)
@click.option(
    "--trace",
    type=bool,
    is_flag=True,
    hidden=True,
    help="Trace message interactions with evaluated model to terminal.",
    envvar="INSPECT_EVAL_TRACE",
)
@click.option(
    "--fail-on-error",
    type=float,
    is_flag=False,
    flag_value=0.0,
    help=FAIL_ON_ERROR_HELP,
    envvar="INSPECT_EVAL_FAIL_ON_ERROR",
)
@click.option(
    "--no-fail-on-error",
    type=bool,
    is_flag=True,
    default=False,
    help=NO_FAIL_ON_ERROR_HELP,
    envvar="INSPECT_EVAL_NO_FAIL_ON_ERROR",
)
@click.option(
    "--retry-on-error",
    is_flag=False,
    flag_value="true",
    default=None,
    callback=int_or_bool_flag_callback(DEFAULT_RETRY_ON_ERROR),
    help=RETRY_ON_ERROR_HELP,
    envvar="INSPECT_EVAL_RETRY_ON_ERROR",
)
@click.option(
    "--no-log-samples",
    type=bool,
    is_flag=True,
    help=NO_LOG_SAMPLES_HELP,
    envvar="INSPECT_EVAL_LOG_SAMPLES",
)
@click.option(
    "--no-log-realtime",
    type=bool,
    is_flag=True,
    help=NO_LOG_REALTIME_HELP,
    envvar="INSPECT_EVAL_LOG_REALTIME",
)
@click.option(
    "--log-images/--no-log-images",
    type=bool,
    default=True,
    is_flag=True,
    help=LOG_IMAGES_HELP,
    envvar="INSPECT_EVAL_LOG_IMAGES",
)
@click.option(
    "--log-buffer", type=int, help=LOG_BUFFER_HELP, envvar="INSPECT_EVAL_LOG_BUFFER"
)
@click.option(
    "--log-shared",
    is_flag=False,
    flag_value="true",
    default=None,
    callback=int_or_bool_flag_callback(DEFAULT_LOG_SHARED),
    help=LOG_SHARED_HELP,
    envvar=["INSPECT_LOG_SHARED", "INSPECT_EVAL_LOG_SHARED"],
)
@click.option(
    "--no-score",
    type=bool,
    is_flag=True,
    help=NO_SCORE_HELP,
    envvar="INSPECT_EVAL_SCORE",
)
@click.option(
    "--no-score-display",
    type=bool,
    is_flag=True,
    help=NO_SCORE_HELP,
    envvar="INSPECT_EVAL_SCORE_DISPLAY",
)
@click.option(
    "--max-connections",
    type=int,
    help=MAX_CONNECTIONS_HELP,
    envvar="INSPECT_EVAL_MAX_CONNECTIONS",
)
@click.option(
    "--max-retries", type=int, help=MAX_RETRIES_HELP, envvar="INSPECT_EVAL_MAX_RETRIES"
)
@click.option("--timeout", type=int, help=TIMEOUT_HELP, envvar="INSPECT_EVAL_TIMEOUT")
@click.option(
    "--log-level-transcript",
    type=click.Choice(
        [level.lower() for level in ALL_LOG_LEVELS],
        case_sensitive=False,
    ),
    default=DEFAULT_LOG_LEVEL_TRANSCRIPT,
    envvar="INSPECT_LOG_LEVEL_TRANSCRIPT",
    help=f"Set the log level of the transcript (defaults to '{DEFAULT_LOG_LEVEL_TRANSCRIPT}')",
)
@common_options
def eval_retry_command(
    log_files: tuple[str],
    max_samples: int | None,
    max_tasks: int | None,
    max_subprocesses: int | None,
    max_sandboxes: int | None,
    no_sandbox_cleanup: bool | None,
    trace: bool | None,
    fail_on_error: bool | float | None,
    no_fail_on_error: bool | None,
    retry_on_error: int | None,
    no_log_samples: bool | None,
    no_log_realtime: bool | None,
    log_images: bool | None,
    log_buffer: int | None,
    log_shared: int | None,
    no_score: bool | None,
    no_score_display: bool | None,
    max_connections: int | None,
    max_retries: int | None,
    timeout: int | None,
    log_level_transcript: str,
    **common: Unpack[CommonOptions],
) -> None:
    """Retry failed evaluation(s)"""
    # resolve common options
    process_common_options(common)

    # resolve negating options
    sandbox_cleanup = False if no_sandbox_cleanup else None
    log_samples = False if no_log_samples else None
    log_realtime = False if no_log_realtime else None
    log_images = False if log_images is False else None
    score = False if no_score else True
    score_display = False if no_score_display else None

    # resolve fail_on_error
    if no_fail_on_error is True:
        fail_on_error = False
    elif fail_on_error == 0.0:
        fail_on_error = True

    # resolve retry on error
    if retry_on_error == 0:
        retry_on_error = None

    # resolve log file
    retry_log_files = [
        log_file_info(filesystem(log_file).info(log_file)) for log_file in log_files
    ]

    # retry
    eval_retry(
        retry_log_files,
        log_level=common["log_level"],
        log_level_transcript=log_level_transcript,
        log_dir=common["log_dir"],
        max_samples=max_samples,
        max_tasks=max_tasks,
        max_subprocesses=max_subprocesses,
        max_sandboxes=max_sandboxes,
        sandbox_cleanup=sandbox_cleanup,
        trace=trace,
        fail_on_error=fail_on_error,
        retry_on_error=retry_on_error,
        debug_errors=common["debug_errors"],
        log_samples=log_samples,
        log_realtime=log_realtime,
        log_images=log_images,
        log_buffer=log_buffer,
        log_shared=log_shared,
        score=score,
        score_display=score_display,
        max_retries=max_retries,
        timeout=timeout,
        max_connections=max_connections,
    )
