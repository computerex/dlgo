// systemone-eval evaluates an independently trained Q4 System One readout.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/computerex/dlgo/models/llm"
)

type example struct {
	ID           string            `json:"id"`
	Family       string            `json:"family"`
	State        string            `json:"state"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
	Target       string            `json:"target"`
	Codes        map[string]string `json:"codes,omitempty"`
	SplitRole    string            `json:"split_role,omitempty"`
}

type familyMetrics struct {
	Rows, Correct int
	NLL, Brier    float64
}

func main() {
	model := flag.String("model", "", "Q4 GGUF model path")
	artifact := flag.String("artifact", "", "System One readout JSON")
	data := flag.String("data", "", "evaluation JSONL")
	ctx := flag.Int("ctx", 512, "context length")
	bins := flag.Int("bins", 10, "number of ECE bins")
	device := flag.String("device", "", "evaluation device: cpu, gpu, or empty to follow the artifact's training device (gpu requires -tags \"cgo vulkan\")")
	flag.Parse()
	if *model == "" || *artifact == "" || *data == "" {
		fmt.Fprintln(os.Stderr, "usage: systemone-eval -model MODEL.gguf -artifact READOUT.json -data EVAL.jsonl")
		os.Exit(2)
	}
	pipe, err := llm.NewPipeline(*model, *ctx)
	if err != nil {
		panic(err)
	}
	defer pipe.Model.Close()
	readout, err := llm.LoadSystemOneReadout(*artifact, pipe.Model.Config.EmbeddingDim)
	if err != nil {
		panic(err)
	}
	// Feature extraction must happen on a device the readout supports: GPU and
	// CPU extract structurally different hidden states, except for "any"
	// artifacts which were trained on the union of both devices' features.
	evalDevice := *device
	if evalDevice == "" {
		evalDevice = readout.NormalizeDevice()
		if evalDevice == "any" {
			evalDevice = "cpu"
		}
	}
	if artifactDevice := readout.NormalizeDevice(); artifactDevice != "any" && evalDevice != artifactDevice {
		panic(fmt.Sprintf("artifact was trained on %s features but -device %s was requested", artifactDevice, evalDevice))
	}
	var evaluate = func(ex example, q llm.SystemOneQuestion) (*llm.SystemOneResult, error) {
		return pipe.SystemOneWithReadout(ex.State, q, llm.SystemOneConfig{}, readout)
	}
	var gpuCleanup func()
	if evalDevice == "gpu" {
		var extractor hiddenExtractor
		extractor, gpuCleanup, err = newGPUExtractor(pipe)
		if err != nil {
			panic(err)
		}
		defer gpuCleanup()
		evaluate = func(ex example, q llm.SystemOneQuestion) (*llm.SystemOneResult, error) {
			return extractor.(interface {
				SystemOneWithReadout(state string, q llm.SystemOneQuestion, cfg llm.SystemOneConfig, readout *llm.SystemOneReadout) (*llm.SystemOneResult, error)
			}).SystemOneWithReadout(ex.State, q, llm.SystemOneConfig{}, readout)
		}
		fmt.Printf("evaluating on %s\n", evalDevice)
	}

	file, err := os.Open(*data)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024), 4*1024*1024)
	byFamily := map[string]*familyMetrics{}
	total, correct := 0, 0
	var nll, brier float64
	var confs, accs []float64
	var examples []example
	for scanner.Scan() {
		var ex example
		if err := json.Unmarshal(scanner.Bytes(), &ex); err != nil {
			panic(err)
		}
		examples = append(examples, ex)
		q := question(ex)
		r, err := evaluate(ex, q)
		if err != nil {
			panic(err)
		}
		total++
		ok := r.Value == ex.Target
		if ok {
			correct++
		}
		confs = append(confs, r.Confidence)
		if ok {
			accs = append(accs, 1)
		} else {
			accs = append(accs, 0)
		}
		var rowNLL, rowBrier float64
		for _, score := range r.Choices {
			if score.Value == ex.Target {
				rowNLL -= math.Log(math.Max(score.Probability, 1e-12))
				d := score.Probability - 1
				rowBrier += d * d
			} else {
				rowBrier += score.Probability * score.Probability
			}
		}
		nll += rowNLL
		brier += rowBrier
		fm := byFamily[ex.Family]
		if fm == nil {
			fm = &familyMetrics{}
			byFamily[ex.Family] = fm
		}
		fm.Rows++
		if ok {
			fm.Correct++
		}
		fm.NLL += rowNLL
		fm.Brier += rowBrier
		fmt.Printf("%s family=%s target=%s got=%s confidence=%.3f\n", ex.ID, ex.Family, ex.Target, r.Value, r.Confidence)
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	ece := expectedCalibrationError(confs, accs, *bins)
	fmt.Printf("\nOVERALL accuracy=%.2f%% rows=%d nll=%.4f brier=%.4f ece=%.4f\n", 100*float64(correct)/float64(total), total, nll/float64(total), brier/float64(total), ece)
	printReliabilityDiagram(confs, accs, *bins)
	names := make([]string, 0, len(byFamily))
	for name := range byFamily {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fm := byFamily[name]
		fmt.Printf("FAMILY %-18s accuracy=%.2f%% rows=%d nll=%.4f brier=%.4f\n", name, 100*float64(fm.Correct)/float64(fm.Rows), fm.Rows, fm.NLL/float64(fm.Rows), fm.Brier/float64(fm.Rows))
	}
	// Zero-shot gate: compare held-out accuracy on domains that never appeared
	// in training against domains that did. A domain-agnostic readout needs the
	// two groups to be close.
	seenRoles := map[string]bool{}
	for _, ex := range examples {
		if ex.SplitRole != "zeroshot" {
			seenRoles[ex.Family] = true
		}
	}
	var seenRows, zsRows, seenCorrect, zsCorrect int
	var seenNLL, zsNLL float64
	for _, name := range names {
		fm := byFamily[name]
		if seenRoles[name] {
			seenRows += fm.Rows
			seenCorrect += fm.Correct
			seenNLL += fm.NLL
		} else {
			zsRows += fm.Rows
			zsCorrect += fm.Correct
			zsNLL += fm.NLL
		}
	}
	if zsRows > 0 {
		fmt.Printf("ZERO-SHOT domains=%d rows=%d accuracy=%.2f%% nll=%.4f\n", len(names)-len(seenRoles), zsRows, 100*float64(zsCorrect)/float64(zsRows), zsNLL/float64(zsRows))
	}
	if seenRows > 0 {
		fmt.Printf("SEEN-DOMAIN rows=%d accuracy=%.2f%% nll=%.4f\n", seenRows, 100*float64(seenCorrect)/float64(seenRows), seenNLL/float64(seenRows))
	}
}

// expectedCalibrationError computes the Expected Calibration Error over the
// given confidence/accuracy pairs using fixed-width bins. A perfectly
// calibrated model has ECE == 0.
func expectedCalibrationError(confs, accs []float64, bins int) float64 {
	if len(confs) == 0 || bins <= 0 {
		return 0
	}
	type bin struct {
		count, confSum, accSum float64
	}
	bs := make([]bin, bins)
	for i := range confs {
		c := confs[i]
		if c < 0 {
			c = 0
		}
		if c > 1 {
			c = 1
		}
		idx := int(c * float64(bins))
		if idx >= bins {
			idx = bins - 1
		}
		bs[idx].count++
		bs[idx].confSum += c
		bs[idx].accSum += accs[i]
	}
	var ece float64
	for _, b := range bs {
		if b.count == 0 {
			continue
		}
		conf := b.confSum / b.count
		acc := b.accSum / b.count
		ece += (b.count / float64(len(confs))) * math.Abs(conf-acc)
	}
	return ece
}

// printReliabilityDiagram renders a text reliability diagram: for each
// confidence bin, the mean predicted confidence versus observed accuracy.
// Perfect calibration places every bar on the diagonal.
func printReliabilityDiagram(confs, accs []float64, bins int) {
	if len(confs) == 0 || bins <= 0 {
		return
	}
	type bin struct {
		count, confSum, accSum float64
	}
	bs := make([]bin, bins)
	for i := range confs {
		c := confs[i]
		if c < 0 {
			c = 0
		}
		if c > 1 {
			c = 1
		}
		idx := int(c * float64(bins))
		if idx >= bins {
			idx = bins - 1
		}
		bs[idx].count++
		bs[idx].confSum += c
		bs[idx].accSum += accs[i]
	}
	fmt.Println("RELIABILITY (bin: mean_conf / observed_acc / count / gap)")
	width := 1.0 / float64(bins)
	for i, b := range bs {
		if b.count == 0 {
			continue
		}
		conf := b.confSum / b.count
		acc := b.accSum / b.count
		gap := conf - acc
		fmt.Printf("  [%.1f-%.1f) conf=%.3f acc=%.3f n=%-4d gap=%+.3f\n",
			float64(i)*width, float64(i+1)*width, conf, acc, int(b.count), gap)
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
		choice := llm.Choice{Value: k, Label: ex.Criteria[k]}
		if letter, ok := ex.Codes[k]; ok {
			choice.Code = fmt.Sprintf(" %c", letter[0])
		}
		choices[i] = choice
	}
	return llm.SystemOneQuestion{ID: ex.ID, Prompt: ex.Instructions, Choices: choices}
}
