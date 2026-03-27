package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/computerex/dlgo/server"
)

type stringSlice []string

func (s *stringSlice) String() string {
	return strings.Join(*s, ", ")
}

func (s *stringSlice) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func main() {
	host := flag.String("host", "0.0.0.0", "bind address")
	port := flag.String("port", "8080", "listen port")
	modelPath := flag.String("model", "", "initial GGUF model to load")
	modelID := flag.String("id", "", "model ID (default: filename without .gguf)")
	useGPU := flag.Bool("gpu", false, "use GPU (Vulkan) for inference")
	ctx := flag.Int("ctx", 0, "max context length (0 = native model context, auto-reduced by memory budget)")
	maxVRAM := flag.Int("max-vram", 0, "max VRAM to use in MB (0 = auto, reserves 1.5 GB for system)")
	frontendDir := flag.String("frontend", "", "path to frontend dist/ directory to serve")

	var modelsDirs stringSlice
	flag.Var(&modelsDirs, "models-dir", "directory to scan for .gguf models (can be specified multiple times)")

	flag.Parse()

	if *maxVRAM > 0 {
		os.Setenv("DLGO_MAX_VRAM_MB", fmt.Sprintf("%d", *maxVRAM))
	}

	manager := server.NewModelManager()
	chatManager := server.NewChatManager()
	registerGPU(manager)

	// Scan and register available models from specified directories
	for _, dir := range modelsDirs {
		if err := scanForModels(dir, manager); err != nil {
			log.Printf("Warning: failed to scan directory %s: %v", dir, err)
		}
	}

	// Load single model if specified
	if *modelPath != "" {
		id := *modelID
		if id == "" {
			base := filepath.Base(*modelPath)
			id = base[:len(base)-len(filepath.Ext(base))]
		}
		if err := manager.LoadModel(id, *modelPath, *useGPU, *ctx); err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}
	}

	addr := *host + ":" + *port
	srv := server.NewServer(addr, manager, chatManager)

	if *frontendDir != "" {
		if info, err := os.Stat(*frontendDir); err == nil && info.IsDir() {
			log.Printf("Serving frontend from %s", *frontendDir)
			srv.SetFrontendHandler(http.FileServer(http.Dir(*frontendDir)))
		}
	}

	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

func scanForModels(dir string, manager *server.ModelManager) error {
	info, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return nil
	}

	log.Printf("Scanning for models in: %s", dir)

	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	foundCount := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") {
			path := filepath.Join(dir, name)
			id := name[:len(name)-len(filepath.Ext(name))]
			manager.RegisterAvailableModel(id, path)
			foundCount++
		}
	}

	log.Printf("Found %d models in %s", foundCount, dir)
	return nil
}
