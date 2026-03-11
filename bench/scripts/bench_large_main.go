//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/computerex/dlgo/gpu"
	"github.com/computerex/dlgo/memory"
	"github.com/computerex/dlgo/models/llm"
	"github.com/computerex/dlgo/ops"
)

type modelSpec struct {
	name, ggufPath, ollamaName string
}

var models = []modelSpec{
	{"Qwen3.5 9B Q3_K_M", `C:\projects\gollm\Qwen3.5-9B-Q3_K_M.gguf`, "qwen35-9b"},
	{"Qwen3.5 27B Q3_K_M", `C:\projects\gollm\Qwen3.5-27B-Q3_K_M.gguf`, "qwen35-27b"},
	{"Qwen3.5 35B-A3B MoE Q3_K_M", `C:\projects\gollm\Qwen3.5-35B-A3B-Q3_K_M.gguf`, "qwen35-35b-moe"},
}

func main() {
	prompt := "Explain the theory of general relativity in simple terms"
	genTokens := 20

	for _, ms := range models {
		fmt.Printf("\n========== %s ==========\n", ms.name)

		// Unload any previously cached Ollama model to free RAM/VRAM
		ollamaUnload(ms.ollamaName)
		// Also unload any other models from previous iterations
		for _, other := range models {
			if other.ollamaName != ms.ollamaName {
				ollamaUnload(other.ollamaName)
			}
		}
		time.Sleep(2 * time.Second)

		fmt.Printf("Loading %s...\n", ms.name)
		pipe, err := llm.NewPipeline(ms.ggufPath, 512)
		if err != nil {
			fmt.Printf("  Load FAIL: %v\n", err)
			continue
		}
		cfg := pipe.Model.Config
		m := pipe.Model

		tokens := pipe.Tokenizer.Encode(prompt)
		if len(tokens) > 0 && tokens[0] == int32(cfg.BOS) {
			tokens = tokens[1:]
		}
		fmt.Printf("  Config: layers=%d dim=%d heads=%d headDim=%d\n",
			cfg.NumLayers, cfg.EmbeddingDim, cfg.NumHeads, cfg.HeadDim)
		fmt.Printf("  Prompt tokens: %d\n", len(tokens))

		// --- dlgo CPU (uses batch prefill) ---
		{
			rs := llm.NewRunState(cfg, 512)
			bs := llm.NewBatchState(cfg, 512)
			kv := memory.NewMultiLayerKVCache(cfg.NumLayers, cfg.NumKVHeads*cfg.HeadDim, 512)

			t0 := time.Now()
			int32Tokens := make([]int32, len(tokens))
			copy(int32Tokens, tokens)
			llm.ForwardBatch(m, int32Tokens, 0, kv, rs, bs)
			prefillDur := time.Since(t0)

			t1 := time.Now()
			pos := len(tokens)
			for g := 0; g < genTokens; g++ {
				best := argmax(rs.Logits)
				llm.Forward(m, int32(best), pos, kv, rs)
				pos++
			}
			genDur := time.Since(t1)
			fmt.Printf("  dlgo CPU: prefill=%dms  gen=%.1f tok/s\n",
				prefillDur.Milliseconds(), float64(genTokens)/genDur.Seconds())
		}
		runtime.GC()

		// --- dlgo GPU ---
		{
			gpuPipe, err := gpu.NewGpuPipeline(pipe)
			if err != nil {
				fmt.Printf("  dlgo GPU: FAIL to create pipeline: %v\n", err)
			} else {
				// Warm up
				res, err := gpuPipe.GenerateDetailed(prompt, llm.GenerateConfig{
					MaxTokens: genTokens,
					Sampler:   ops.SamplerConfig{Temperature: 0},
					Seed:      42,
				})
				if err != nil {
					fmt.Printf("  dlgo GPU: FAIL: %v\n", err)
				} else {
					fmt.Printf("  dlgo GPU: prefill=%.0fms  gen=%.1f tok/s  text=%q\n",
						res.PrefillTimeMs, res.TokensPerSec, truncate(res.Text, 60))
				}
				_ = res
				gpuPipe.FreeAll()
			}
		}
		// Free dlgo model before running Ollama to avoid competing for RAM
		pipe = nil
		m = nil
		runtime.GC()
		debug.FreeOSMemory()
		time.Sleep(1 * time.Second)

		// --- Ollama CPU ---
		{
			olPrefill, olGen, olPrefTok, olGenTok, olText, err := ollamaGenerate(ms.ollamaName, prompt, genTokens, 0)
			if err != nil {
				fmt.Printf("  Ollama CPU: FAIL: %v\n", err)
			} else {
				olGenTokS := float64(olGenTok) / (olGen / 1000.0)
				fmt.Printf("  Ollama CPU: prefill=%.0fms (%d tok)  gen=%.1f tok/s (%d tok)  text=%q\n",
					olPrefill, olPrefTok, olGenTokS, olGenTok, truncate(olText, 60))
			}
		}

		// --- Ollama GPU ---
		{
			olPrefill, olGen, olPrefTok, olGenTok, olText, err := ollamaGenerate(ms.ollamaName, prompt, genTokens, 999)
			if err != nil {
				fmt.Printf("  Ollama GPU: FAIL: %v\n", err)
			} else {
				olGenTokS := float64(olGenTok) / (olGen / 1000.0)
				fmt.Printf("  Ollama GPU: prefill=%.0fms (%d tok)  gen=%.1f tok/s (%d tok)  text=%q\n",
					olPrefill, olPrefTok, olGenTokS, olGenTok, truncate(olText, 60))
			}
		}

		// Unload Ollama model after benchmarking to free RAM/VRAM
		ollamaUnload(ms.ollamaName)
		time.Sleep(1 * time.Second)
		runtime.GC()
	}
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

type ollamaReq struct {
	Model     string      `json:"model"`
	Msgs      []ollamaMsg `json:"messages"`
	Stream    bool        `json:"stream"`
	Options   ollamaOpts  `json:"options"`
	KeepAlive string      `json:"keep_alive"`
}
type ollamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type ollamaOpts struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
	Seed        int     `json:"seed"`
	NumGPU      int     `json:"num_gpu"`
}
type ollamaResp struct {
	Message            ollamaMsg     `json:"message"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func ollamaGenerate(model, prompt string, maxTok, numGPU int) (prefillMs, genMs float64, prefillTok, genTok int, text string, err error) {
	req := ollamaReq{
		Model:     model,
		Msgs:      []ollamaMsg{{Role: "user", Content: prompt}},
		Stream:    false,
		Options:   ollamaOpts{Temperature: 0, NumPredict: maxTok, Seed: 42, NumGPU: numGPU},
		KeepAlive: "0s",
	}
	b, _ := json.Marshal(req)
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(b))
	if err != nil {
		return 0, 0, 0, 0, "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var r ollamaResp
	if err := json.Unmarshal(raw, &r); err != nil {
		return 0, 0, 0, 0, "", fmt.Errorf("json: %v body: %s", err, string(raw[:min(len(raw), 200)]))
	}
	pms := float64(r.PromptEvalDuration) / 1e6
	gms := float64(r.EvalDuration) / 1e6
	return pms, gms, r.PromptEvalCount, r.EvalCount, r.Message.Content, nil
}

func ollamaUnload(model string) {
	body, _ := json.Marshal(map[string]interface{}{
		"model":      model,
		"keep_alive": 0,
	})
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post("http://localhost:11434/api/generate", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
}

func argmax(v []float32) int {
	b := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[b] {
			b = i
		}
	}
	return b
}
