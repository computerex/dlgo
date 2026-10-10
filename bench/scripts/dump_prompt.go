//go:build ignore

package main

import (
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
)

func main() {
	modelPath := os.Args[1]
	m, err := llm.LoadModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	cfg := m.Config

	noThink := false
	msgs := []llm.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Write a comprehensive tutorial on building a REST API with Python Flask. Cover project setup, routing, request handling, database integration with SQLAlchemy, authentication with JWT tokens, error handling, testing, deployment, and best practices. Include complete code examples for each section."},
	}
	prompt := llm.FormatMessages(cfg, msgs, llm.FormatOptions{EnableThinking: &noThink})
	fmt.Printf("=== FORMATTED PROMPT (%d chars) ===\n%s\n=== END ===\n", len(prompt), prompt)
}
