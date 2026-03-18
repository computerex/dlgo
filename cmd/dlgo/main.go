package main

import (
	"fmt"
	"os"
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "server", "serve":
		cmdServer(os.Args[2:])
	case "info":
		cmdInfo(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Printf("dlgo v%s\n", version)
	case "help", "--help", "-h":
		printUsage()
	default:
		// If the first arg looks like a path, treat it as `dlgo run <path>`
		if looksLikeModelPath(os.Args[1]) {
			cmdRun(os.Args[1:])
		} else {
			fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", os.Args[1])
			printUsage()
			os.Exit(1)
		}
	}
}

func looksLikeModelPath(s string) bool {
	if len(s) > 0 && s[0] == '-' {
		return false
	}
	for _, ext := range []string{".gguf", ".ggml", ".bin"} {
		if len(s) > len(ext) && s[len(s)-len(ext):] == ext {
			return true
		}
	}
	return false
}

func printUsage() {
	fmt.Printf(`dlgo v%s — fast LLM inference in pure Go

Usage:
  dlgo run <model.gguf> [flags]    Start interactive chat (like ollama)
  dlgo server [flags]              Start API server with web UI
  dlgo info <model.gguf>           Show model metadata
  dlgo version                     Print version
  dlgo help                        Show this help

Run flags:
  --gpu                Use Vulkan GPU backend
  --ctx N              Context length in tokens (default: 8192)
  --max-tokens N       Max tokens per response (default: 512)
  --temp T             Sampling temperature (default: 0.7)
  --top-k K            Top-K sampling (default: 40)
  --top-p P            Nucleus sampling (default: 0.9)
  --system "..."       System prompt
  --threads N          Worker threads (0 = auto)

Server flags:
  --model <path>       GGUF model to pre-load
  --gpu                Use Vulkan GPU backend
  --host ADDR          Bind address (default: 0.0.0.0)
  --port PORT          Listen port (default: 8080)
  --ctx N              Context length (0 = native model context, default: 0)

Examples:
  dlgo run llama-3.2-1b-q4_k_m.gguf
  dlgo run qwen3.5-0.8b-q8_0.gguf --gpu
  dlgo server --model model.gguf --gpu --port 8080
  dlgo info model.gguf
`, version)
}
