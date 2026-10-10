package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/computerex/dlgo/models/llm"
)

type systemOneInput struct {
	State        string            `json:"state"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
	Type         string            `json:"type,omitempty"`
	Min          int               `json:"min,omitempty"`
	Max          int               `json:"max,omitempty"`
}

func cmdSystemOne(args []string) {
	fs := flag.NewFlagSet("systemone", flag.ExitOnError)
	modelPath := fs.String("model", "", "Q4 GGUF backbone path")
	readoutPath := fs.String("readout", "", "trained System One readout artifact path")
	inputPath := fs.String("input", "", "JSON input path, or - for stdin")
	ctx := fs.Int("ctx", 512, "context length")
	fs.Parse(args)
	if *modelPath == "" || *readoutPath == "" || *inputPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: dlgo systemone --model MODEL.gguf --readout READOUT.json --input REQUEST.json")
		os.Exit(2)
	}
	var raw []byte
	var err error
	if *inputPath == "-" {
		raw, err = os.ReadFile("CONIN$")
	} else {
		raw, err = os.ReadFile(*inputPath)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "read input:", err)
		os.Exit(1)
	}
	var input systemOneInput
	if err := json.Unmarshal(raw, &input); err != nil {
		fmt.Fprintln(os.Stderr, "parse input:", err)
		os.Exit(1)
	}
	if input.State == "" || input.Instructions == "" {
		fmt.Fprintln(os.Stderr, "input requires state and instructions")
		os.Exit(1)
	}

	pipe, err := llm.NewPipeline(*modelPath, *ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load model:", err)
		os.Exit(1)
	}
	defer pipe.Model.Close()
	readout, err := llm.LoadSystemOneReadout(*readoutPath, pipe.Model.Config.EmbeddingDim)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load readout:", err)
		os.Exit(1)
	}

	var result *llm.SystemOneResult
	switch input.Type {
	case "", "choice":
		if len(input.Criteria) < 2 {
			fmt.Fprintln(os.Stderr, "choice requires at least two criteria")
			os.Exit(1)
		}
		keys := make([]string, 0, len(input.Criteria))
		for key := range input.Criteria {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		choices := make([]llm.Choice, len(keys))
		for i, key := range keys {
			choices[i] = llm.Choice{Value: key, Label: input.Criteria[key]}
		}
		result, err = pipe.SystemOneWithReadout(input.State, llm.SystemOneQuestion{ID: "decision", Prompt: input.Instructions, Choices: choices}, llm.SystemOneConfig{}, readout)
	case "noul":
		result, err = pipe.SystemOneNoul(input.State, input.Instructions, llm.SystemOneConfig{}, readout)
	case "score":
		result, err = pipe.SystemOneScore(input.State, input.Instructions, input.Min, input.Max, llm.SystemOneConfig{}, readout)
	default:
		fmt.Fprintln(os.Stderr, "unknown type:", input.Type, "(expected choice, noul, or score)")
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "evaluate:", err)
		os.Exit(1)
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, "write result:", err)
		os.Exit(1)
	}
}
