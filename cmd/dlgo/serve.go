package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"github.com/computerex/dlgo/server"
)

func cmdServer(args []string) {
	fs := flag.NewFlagSet("server", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: dlgo server [flags]")
		fs.PrintDefaults()
	}

	host := fs.String("host", "0.0.0.0", "bind address")
	port := fs.String("port", "8080", "listen port")
	modelPath := fs.String("model", "", "GGUF model to pre-load")
	modelID := fs.String("id", "", "model ID (default: filename without .gguf)")
	useGPU := fs.Bool("gpu", false, "use Vulkan GPU backend")
	ctx := fs.Int("ctx", 0, "max context length (0 = native model context, auto-reduced by memory budget)")
	frontendDir := fs.String("frontend", "", "path to frontend dist/ directory")

	fs.Parse(args)

	manager := server.NewModelManager()
	chatManager := server.NewChatManager()
	registerGPU(manager)

	if *modelPath != "" {
		id := *modelID
		if id == "" {
			base := filepath.Base(*modelPath)
			id = base[:len(base)-len(filepath.Ext(base))]
		}
		log.Printf("Loading model %q from %s ...", id, *modelPath)
		if err := manager.LoadModel(id, *modelPath, *useGPU, *ctx); err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}
		log.Printf("Model %q ready", id)
	}

	addr := *host + ":" + *port
	srv := server.NewServer(addr, manager, chatManager)

	// Auto-detect frontend directory
	feDir := *frontendDir
	if feDir == "" {
		candidates := []string{
			"frontend/dist",
			filepath.Join(execDir(), "frontend", "dist"),
		}
		for _, c := range candidates {
			if info, err := os.Stat(c); err == nil && info.IsDir() {
				feDir = c
				break
			}
		}
	}

	if feDir != "" {
		if info, err := os.Stat(feDir); err == nil && info.IsDir() {
			log.Printf("Serving web UI from %s", feDir)
			srv.SetFrontendHandler(http.FileServer(http.Dir(feDir)))
		}
	}

	fmt.Println()
	fmt.Printf("  dlgo server v%s\n", version)
	fmt.Printf("  %s/%s, %d cores\n", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Println()
	fmt.Printf("  Web UI:  http://localhost:%s\n", *port)
	fmt.Printf("  API:     http://localhost:%s/v1/chat/completions\n", *port)
	fmt.Printf("  Health:  http://localhost:%s/health\n", *port)
	fmt.Println()

	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

func execDir() string {
	ex, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(ex)
}
