// Example: JSON grammar-constrained generation with Qwen 3.5 0.6B
//
// Usage: go run examples/json_grammar/main.go [model.gguf]
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/grammar"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

func main() {
	modelPath := `C:\models\Qwen3.5-0.8B-Q8_0.gguf`
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	fmt.Printf("Loading model: %s\n", modelPath)
	pipe, err := llm.NewPipeline(modelPath, 2048)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model: %s (%d layers, %d dim, vocab %d)\n",
		pipe.Model.Config.Architecture,
		pipe.Model.Config.NumLayers,
		pipe.Model.Config.EmbeddingDim,
		pipe.Model.Config.VocabSize)

	// Parse the JSON grammar
	gram, err := grammar.Parse(grammar.JSONGrammar)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse grammar: %v\n", err)
		os.Exit(1)
	}

	// Test 1: Basic JSON generation
	fmt.Println("\n=== Test 1: Basic JSON object ===")
	prompt1 := llm.FormatChat(pipe.Model.Config, "",
		"Return a JSON object with fields: name (string), age (number), city (string). "+
			"Example: {\"name\": \"Alice\", \"age\": 30, \"city\": \"NYC\"}. "+
			"Return ONLY the JSON, no other text.")

	result1 := generateWithGrammar(pipe, prompt1, gram.Clone(), 256)
	fmt.Printf("\nGenerated: %s\n", result1)
	validateJSON("Test 1", result1)

	// Test 2: JSON with nested objects
	fmt.Println("\n=== Test 2: Nested JSON ===")
	gram2, _ := grammar.Parse(grammar.JSONGrammar)
	prompt2 := llm.FormatChat(pipe.Model.Config, "",
		"Return a JSON object representing a person with nested address. "+
			"Include: name, age, address (with street, city, zip). "+
			"Return ONLY the JSON.")

	result2 := generateWithGrammar(pipe, prompt2, gram2, 512)
	fmt.Printf("\nGenerated: %s\n", result2)
	validateJSON("Test 2", result2)

	// Test 3: JSON array
	fmt.Println("\n=== Test 3: JSON array (using array grammar) ===")
	arrayGrammar := `root   ::= array
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= | " " | "\n" [ \t]*
`
	gram3, err := grammar.Parse(arrayGrammar)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse array grammar: %v\n", err)
		os.Exit(1)
	}

	prompt3 := llm.FormatChat(pipe.Model.Config, "",
		"Return a JSON array of 3 fruits with their colors. "+
			"Example: [{\"fruit\": \"apple\", \"color\": \"red\"}]. "+
			"Return ONLY the JSON array.")

	result3 := generateWithGrammar(pipe, prompt3, gram3, 512)
	fmt.Printf("\nGenerated: %s\n", result3)
	validateJSON("Test 3", result3)

	// Test 4: Compare with unconstrained generation
	fmt.Println("\n=== Test 4: Unconstrained (no grammar) for comparison ===")
	prompt4 := llm.FormatChat(pipe.Model.Config, "",
		"Return a JSON object with fields: name (string), age (number). "+
			"Return ONLY the JSON, no other text.")

	result4 := generateWithGrammar(pipe, prompt4, nil, 256)
	fmt.Printf("\nGenerated (no grammar): %s\n", result4)
	validateJSON("Test 4 (unconstrained)", result4)
}

func generateWithGrammar(pipe *llm.Pipeline, prompt string, gram *grammar.Grammar, maxTokens int) string {
	tokens := pipe.Tokenizer.Encode(prompt)
	fmt.Printf("Prompt tokens: %d\n", len(tokens))

	cfg := llm.GenerateConfig{
		MaxTokens: maxTokens,
		Sampler: ops.SamplerConfig{
			Temperature:       0.3,
			TopK:              40,
			TopP:              0.9,
			RepetitionPenalty: 1.1,
		},
		Seed:    42,
		Grammar: gram,
		Stream: func(token string) {
			fmt.Print(token)
		},
	}

	start := time.Now()
	generated, err := pipe.Generate(tokens, cfg)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Fprintf(os.Stderr, "\nGeneration error: %v\n", err)
		return ""
	}

	tokPerSec := float64(len(generated)) / elapsed.Seconds()
	fmt.Printf("\n[%d tokens in %.2fs = %.1f tok/s]\n", len(generated), elapsed.Seconds(), tokPerSec)

	return pipe.Tokenizer.Decode(generated)
}

func validateJSON(testName, s string) {
	var v interface{}
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		fmt.Printf("  ❌ %s: INVALID JSON: %v\n", testName, err)
	} else {
		// Pretty print
		pretty, _ := json.MarshalIndent(v, "  ", "  ")
		fmt.Printf("  ✅ %s: Valid JSON!\n  %s\n", testName, string(pretty))
	}
}