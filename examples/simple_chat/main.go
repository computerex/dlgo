// Simple example: chat with an LLM using dlgo's high-level API.
//
// Usage:
//
//	go run . [model.gguf]
package main

import (
	"fmt"
	"os"

	dlgo "github.com/computerex/dlgo"
)

func main() {
	modelPath := `C:\projects\evoke\models\smollm2-360m-instruct-q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	model, err := dlgo.LoadLLM(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Model:", model.ModelInfo())
	fmt.Println()

	response, err := model.Chat(
		"You are a helpful assistant.",
		"What is the Go programming language?",
		dlgo.WithMaxTokens(128),
		dlgo.WithTemperature(0.7),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Response:", response)
}
