package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/computerex/dlgo/server"
)

func main() {
	host := flag.String("host", "0.0.0.0", "bind address")
	port := flag.String("port", "8080", "listen port")
	modelPath := flag.String("model", "", "initial GGUF model to load")
	modelID := flag.String("id", "", "model ID (default: filename without .gguf)")
	useGPU := flag.Bool("gpu", false, "use GPU (Vulkan) for inference")
	ctx := flag.Int("ctx", 2048, "max context length")
	frontendDir := flag.String("frontend", "", "path to frontend dist/ directory to serve")
	flag.Parse()

	manager := server.NewModelManager()
	registerGPU(manager)

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
	srv := server.NewServer(addr, manager)

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
