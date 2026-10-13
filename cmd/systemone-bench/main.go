// systemone-bench measures end-to-end latency of the two System One engines:
// generic continuation scoring and the trained native readout. It reports
// prefill, scoring, and total time per request so the readout speedup can be
// quantified on this hardware.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

type example struct {
	ID           string            `json:"id"`
	Family       string            `json:"family"`
	State        string            `json:"state"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
	Target       string            `json:"target"`
}

func main() {
	model := flag.String("model", "", "Q4 GGUF model path")
	artifact := flag.String("artifact", "", "System One readout JSON (enables readout benchmark)")
	data := flag.String("data", "", "JSONL benchmark cases")
	limit := flag.Int("limit", 20, "maximum number of cases to benchmark")
	ctx := flag.Int("ctx", 512, "context length")
	flag.Parse()
	if *model == "" || *data == "" {
		fmt.Fprintln(os.Stderr, "usage: systemone-bench -model MODEL.gguf -data CASES.jsonl [-artifact READOUT.json] [-limit N]")
		os.Exit(2)
	}
	pipe, err := llm.NewPipeline(*model, *ctx)
	if err != nil {
		panic(err)
	}
	defer pipe.Model.Close()

	var readout *llm.SystemOneReadout
	if *artifact != "" {
		readout, err = llm.LoadSystemOneReadout(*artifact, pipe.Model.Config.EmbeddingDim)
		if err != nil {
			panic(err)
		}
	}

	file, err := os.Open(*data)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	var cases []example
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024), 4*1024*1024)
	for scanner.Scan() {
		var ex example
		if err := json.Unmarshal(scanner.Bytes(), &ex); err != nil {
			panic(err)
		}
		cases = append(cases, ex)
		if len(cases) >= *limit {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	if len(cases) == 0 {
		fmt.Println("no benchmark cases")
		return
	}

	// Warm up both paths once so first-touch allocations do not skew results.
	warm := cases[0]
	q := question(warm)
	if _, err := pipe.SystemOne(warm.State, q, llm.SystemOneConfig{}); err != nil {
		panic(err)
	}
	if readout != nil {
		if _, err := pipe.SystemOneWithReadout(warm.State, q, llm.SystemOneConfig{}, readout); err != nil {
			panic(err)
		}
	}

	run := func(name string, useReadout bool) {
		var totalPrefill, totalScore time.Duration
		correct := 0
		start := time.Now()
		for _, ex := range cases {
			q := question(ex)
			var r *llm.SystemOneResult
			var err error
			if useReadout {
				r, err = pipe.SystemOneWithReadout(ex.State, q, llm.SystemOneConfig{}, readout)
			} else {
				r, err = pipe.SystemOne(ex.State, q, llm.SystemOneConfig{})
			}
			if err != nil {
				panic(err)
			}
			totalPrefill += time.Duration(r.PrefillTimeMs * float64(time.Millisecond))
			totalScore += time.Duration(r.ScoreTimeMs * float64(time.Millisecond))
			if r.Value == ex.Target {
				correct++
			}
		}
		wall := time.Since(start)
		perCase := wall / time.Duration(len(cases))
		fmt.Printf("%-10s cases=%d wall=%s per-case=%s prefill/case=%s score/case=%s accuracy=%.1f%%\n",
			name, len(cases), wall.Round(time.Millisecond), perCase.Round(time.Millisecond),
			(totalPrefill / time.Duration(len(cases))).Round(time.Microsecond),
			(totalScore / time.Duration(len(cases))).Round(time.Microsecond),
			100*float64(correct)/float64(len(cases)))
	}

	fmt.Printf("benchmarking %d cases (%s model)\n", len(cases), *model)
	run("fallback", false)
	if readout != nil {
		run("readout", true)
	}
}

func question(ex example) llm.SystemOneQuestion {
	keys := make([]string, 0, len(ex.Criteria))
	for k := range ex.Criteria {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	choices := make([]llm.Choice, len(keys))
	for i, k := range keys {
		choices[i] = llm.Choice{Value: k, Label: ex.Criteria[k]}
	}
	return llm.SystemOneQuestion{ID: ex.ID, Prompt: ex.Instructions, Choices: choices}
}