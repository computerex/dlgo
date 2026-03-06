package main

import (
	"fmt"
	"os"
	"time"

	"github.com/computerex/dlgo/models/whisper"
)

func main() {
	modelPath := `C:\projects\evoke\models\whisper-base-q8_0.gguf`
	audioPath := `C:\projects\evoke\testdata\jfk.wav`

	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		audioPath = os.Args[2]
	}

	fmt.Printf("Loading Whisper model from %s...\n", modelPath)
	start := time.Now()
	m, err := whisper.LoadWhisperModel(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model loaded in %v\n", time.Since(start))
	fmt.Printf("  Encoder: %d layers, %d dim, %d heads\n",
		m.Config.NEncLayers, m.Config.DModel, m.Config.NHeads)
	fmt.Printf("  Decoder: %d layers, vocab %d\n",
		m.Config.NDecLayers, m.Config.NVocab)

	fmt.Printf("\nTranscribing %s...\n", audioPath)
	text, err := m.TranscribeFile(audioPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error transcribing: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nTranscription:\n  %s\n", text)
}
