//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

type diagResult struct {
	Text   string `json:"text"`
	Tokens string `json:"tokens"`
	Err    string `json:"err,omitempty"`
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "usage: diag_cpu_worker <model.gguf> <output.json>\n")
		os.Exit(1)
	}
	modelPath := os.Args[1]
	outPath := os.Args[2]

	res := diagResult{}
	defer func() {
		data, _ := json.MarshalIndent(res, "", "  ")
		os.WriteFile(outPath, data, 0644)
	}()

	pipe, err := llm.NewPipeline(modelPath, 512)
	if err != nil {
		res.Err = fmt.Sprintf("load: %v", err)
		return
	}

	cfg := pipe.Model.Config
	prompt := llm.FormatChat(cfg, "You are a helpful assistant.", "Explain what a computer is in exactly two sentences.")

	genCfg := llm.DefaultGenerateConfig()
	genCfg.MaxTokens = 100
	genCfg.Seed = 42
	genCfg.Sampler = ops.SamplerConfig{
		Temperature:       0,
		RepetitionPenalty: 1.1,
	}

	result, err := pipe.GenerateDetailed(prompt, genCfg)
	if err != nil {
		res.Err = fmt.Sprintf("gen: %v", err)
		return
	}

	res.Text = result.Text
	tokenStrs := make([]string, len(result.Tokens))
	for i, t := range result.Tokens {
		tokenStrs[i] = fmt.Sprintf("%d", t)
	}
	res.Tokens = fmt.Sprintf("[%s]", join(tokenStrs, ","))

	fmt.Fprintf(os.Stderr, "  CPU: %d tokens generated\n", result.TotalTokens)
}

func join(strs []string, sep string) string {
	result := ""
	for i, s := range strs {
		if i > 0 {
			result += sep
		}
		result += s
	}
	return result
}
