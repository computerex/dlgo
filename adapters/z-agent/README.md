# Z-Agent for Harbor Terminal-Bench

This is a custom agent adapter for Harbor Terminal-Bench that wraps your "z" streaming harness tool.

## Understanding Your Z Tool

Your "z" command appears to be a "Streaming Harness" tool with the following features:
- Session management
- Workspace support
- API integration
- Model configuration
- Policy evaluation
- Debug and safe modes

## Usage with Harbor

To use your z-agent with Harbor Terminal-Bench:

```bash
# Basic usage (uses default session)
harbor run -d terminal-bench/terminal-bench-2 \
  --agent-import-path adapters.z-agent.agent:ZAgent

# With custom workspace
harbor run -d terminal-bench/terminal-bench-2 \
  --agent-import-path adapters.z-agent.agent:ZAgent \
  --ak workspace=./z-workspace

# With model and API configuration
harbor run -d terminal-bench/terminal-bench-2 \
  --agent-import-path adapters.z-agent.agent:ZAgent \
  --ak model=gpt-4 \
  --ak api_url=https://api.example.com \
  --ak api_key=your-api-key
```

## Configuration Options

You can configure the agent using `--ak` (agent kwargs) flags:

- `z_command`: Path to z executable (default: "z")
- `workspace`: Working directory for z
- `session`: Session name (default: "default")
- `model`: Model to use with z
- `api_url`: API base URL
- `api_key`: API key
- `timeout`: Command timeout in seconds (default: 600)

## Testing Your Setup

Before running the full benchmark, test your z-agent:

```bash
# Test basic z command
z --list
z --help

# Test with Harbor (single task)
harbor run -d terminal-bench/terminal-bench-2 \
  --agent-import-path adapters.z-agent.agent:ZAgent \
  --n-tasks 1
```

## Important Notes

1. **Non-interactive Mode**: Harbor expects agents to run non-interactively. You may need to configure z to work without user input.

2. **Task Processing**: Harbor will pass task descriptions to your agent. Ensure z can process these and return results via stdout.

3. **Error Handling**: The agent wrapper captures both stdout and stderr for debugging.

4. **Workspace Management**: Consider using a dedicated workspace for Terminal-Bench tasks to avoid conflicts.

## Troubleshooting

If Harbor can't find your z command:
1. Ensure "z" is in your system PATH
2. Try using the full path: `--ak z_command=/path/to/z`
3. Test z independently: `z --help`

If z requires interactive input:
1. Check if z has a headless or batch mode
2. You may need to modify the agent wrapper to handle z's specific interface
3. Consider piping input to z instead of using it directly

## Example Full Run

```bash
# Run a few tasks to test
harbor run -d terminal-bench/terminal-bench-2 \
  --agent-import-path adapters.z-agent.agent:ZAgent \
  --ak workspace=./terminal-bench-workspace \
  --n-tasks 3 \
  --n-concurrent 1
```