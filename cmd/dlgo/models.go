package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Catalog entries intentionally name known-good GGUFs and quantizations for
// architectures dlgo supports. The URLs use Hugging Face's resolve endpoint;
// users can also pass any compatible GGUF URL directly to `dlgo pull`.
type modelSpec struct {
	Name    string
	Repo    string
	File    string
	Summary string
	SHA256  string // Optional published checksum. Empty means no checksum is asserted.
}

var modelCatalog = []modelSpec{
	{Name: "qwen3.5:0.8b-q4_k_m", Repo: "unsloth/Qwen3.5-0.8B-GGUF", File: "Qwen3.5-0.8B-Q4_K_M.gguf", Summary: "Qwen3.5 0.8B, Q4_K_M; compact general chat model"},
	{Name: "qwen3.5:0.8b-q8_0", Repo: "unsloth/Qwen3.5-0.8B-GGUF", File: "Qwen3.5-0.8B-Q8_0.gguf", Summary: "Qwen3.5 0.8B, Q8_0; higher-quality compact model"},
	{Name: "qwen3.5:9b-q4_k_m", Repo: "unsloth/Qwen3.5-9B-GGUF", File: "Qwen3.5-9B-Q4_K_M.gguf", Summary: "Qwen3.5 9B, Q4_K_M; general chat model"},
	{Name: "qwen3.5:9b-q3_k_m", Repo: "unsloth/Qwen3.5-9B-GGUF", File: "Qwen3.5-9B-Q3_K_M.gguf", Summary: "Qwen3.5 9B, Q3_K_M; lower-memory model used by the published 9B System One head"},
}

type installedModel struct {
	Name      string `json:"name"`
	Repo      string `json:"repo"`
	File      string `json:"file"`
	Path      string `json:"path"`
	Bytes     int64  `json:"bytes"`
	SHA256    string `json:"sha256,omitempty"`
	Installed string `json:"installed"`
}

type modelStore struct {
	Models []installedModel `json:"models"`
}

func modelDir() (string, error) {
	if configured := os.Getenv("DLGO_MODELS"); configured != "" {
		return configured, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("locate user cache: %w (set DLGO_MODELS to choose a model directory)", err)
	}
	return filepath.Join(base, "dlgo", "models"), nil
}

func storePath(dir string) string { return filepath.Join(dir, "models.json") }

func loadStore(dir string) (modelStore, error) {
	var store modelStore
	data, err := os.ReadFile(storePath(dir))
	if errors.Is(err, os.ErrNotExist) {
		return store, nil
	}
	if err != nil {
		return store, err
	}
	if err := json.Unmarshal(data, &store); err != nil {
		return store, fmt.Errorf("read model manifest: %w", err)
	}
	return store, nil
}

func saveStore(dir string, store modelStore) error {
	data, err := json.MarshalIndent(store, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(storePath(dir), append(data, '\n'), 0o644)
}

func catalogSpec(name string) (modelSpec, bool) {
	for _, spec := range modelCatalog {
		if spec.Name == name {
			return spec, true
		}
	}
	return modelSpec{}, false
}

func huggingFaceURL(repo, file string) string {
	return "https://huggingface.co/" + repo + "/resolve/main/" + file + "?download=true"
}

func cmdModels() {
	fmt.Println("Available dlgo model aliases (served from Hugging Face / Unsloth):")
	for _, spec := range modelCatalog {
		fmt.Printf("  %-24s %s\n    %s\n", spec.Name, spec.Summary, huggingFaceURL(spec.Repo, spec.File))
	}
	fmt.Println("\nUse: dlgo pull <alias>  (or: dlgo pull <https://...model.gguf>)")
}

func cmdList() {
	dir, err := modelDir()
	if err != nil {
		fmt.Fprintln(os.Stderr, "models:", err)
		return
	}
	store, err := loadStore(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "models:", err)
		return
	}
	if len(store.Models) == 0 {
		fmt.Printf("No downloaded models. Use `dlgo models` then `dlgo pull <name>`.\nDirectory: %s\n", dir)
		return
	}
	sort.Slice(store.Models, func(i, j int) bool { return store.Models[i].Name < store.Models[j].Name })
	fmt.Printf("NAME\tSIZE\tPATH\n")
	for _, m := range store.Models {
		fmt.Printf("%s\t%s\t%s\n", m.Name, humanBytes(m.Bytes), m.Path)
	}
}

func cmdPull(args []string) {
	fs := flag.NewFlagSet("pull", flag.ExitOnError)
	dirFlag := fs.String("dir", "", "model directory (default: DLGO_MODELS or user cache)")
	fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "Usage: dlgo pull [-dir DIRECTORY] <alias|https://...gguf>")
		os.Exit(2)
	}
	requested := fs.Arg(0)
	dir := *dirFlag
	if dir == "" {
		var err error
		dir, err = modelDir()
		if err != nil {
			fmt.Fprintln(os.Stderr, "pull:", err)
			os.Exit(1)
		}
	}
	var spec modelSpec
	url := requested
	if found, ok := catalogSpec(requested); ok {
		spec = found
		url = huggingFaceURL(spec.Repo, spec.File)
	} else if !strings.HasPrefix(url, "https://") && !strings.HasPrefix(url, "http://") {
		fmt.Fprintf(os.Stderr, "Unknown alias %q. Run `dlgo models` to list supported aliases, or provide a direct HTTPS GGUF URL.\n", requested)
		os.Exit(2)
	} else {
		spec = modelSpec{Name: filepath.Base(strings.Split(url, "?")[0]), File: filepath.Base(strings.Split(url, "?")[0]), Summary: "direct URL"}
	}
	if !strings.HasSuffix(strings.ToLower(spec.File), ".gguf") {
		fmt.Fprintln(os.Stderr, "pull: only .gguf downloads are supported")
		os.Exit(2)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "pull: create model directory:", err)
		os.Exit(1)
	}
	dest := filepath.Join(dir, spec.File)
	if info, err := os.Stat(dest); err == nil && info.Size() > 0 {
		fmt.Printf("Already downloaded: %s (%s)\n", dest, humanBytes(info.Size()))
		return
	}
	fmt.Printf("Downloading %s\nFrom: %s\nTo:   %s\n", spec.Name, url, dest)
	bytes, digest, err := downloadGGUF(url, dest)
	if err != nil {
		_ = os.Remove(dest)
		fmt.Fprintln(os.Stderr, "pull:", err)
		os.Exit(1)
	}
	if spec.SHA256 != "" && !strings.EqualFold(spec.SHA256, digest) {
		_ = os.Remove(dest)
		fmt.Fprintln(os.Stderr, "pull: downloaded file SHA-256 does not match catalog checksum")
		os.Exit(1)
	}
	store, err := loadStore(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "pull: load manifest:", err)
		os.Exit(1)
	}
	updated := installedModel{Name: spec.Name, Repo: spec.Repo, File: spec.File, Path: dest, Bytes: bytes, SHA256: digest, Installed: time.Now().UTC().Format(time.RFC3339)}
	found := false
	for i := range store.Models {
		if store.Models[i].Name == updated.Name {
			store.Models[i] = updated
			found = true
		}
	}
	if !found {
		store.Models = append(store.Models, updated)
	}
	if err := saveStore(dir, store); err != nil {
		fmt.Fprintln(os.Stderr, "pull: save manifest:", err)
		os.Exit(1)
	}
	fmt.Printf("Downloaded %s (%s)\nSHA-256: %s\n", dest, humanBytes(bytes), digest)
}

func cmdRM(args []string) {
	if len(args) != 1 {
		fmt.Fprintln(os.Stderr, "Usage: dlgo rm <model name listed by `dlgo list`>")
		os.Exit(2)
	}
	dir, err := modelDir()
	if err != nil {
		fmt.Fprintln(os.Stderr, "rm:", err)
		os.Exit(1)
	}
	store, err := loadStore(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "rm:", err)
		os.Exit(1)
	}
	kept := make([]installedModel, 0, len(store.Models))
	removed := false
	for _, m := range store.Models {
		if m.Name != args[0] {
			kept = append(kept, m)
			continue
		}
		if err := os.Remove(m.Path); err != nil && !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintln(os.Stderr, "rm:", err)
			os.Exit(1)
		}
		fmt.Println("Removed", m.Path)
		removed = true
	}
	if !removed {
		fmt.Fprintf(os.Stderr, "rm: model %q is not installed\n", args[0])
		os.Exit(1)
	}
	store.Models = kept
	if err := saveStore(dir, store); err != nil {
		fmt.Fprintln(os.Stderr, "rm:", err)
		os.Exit(1)
	}
}

func downloadGGUF(url, dest string) (int64, string, error) {
	client := &http.Client{Timeout: 0}
	resp, err := client.Get(url)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, "", fmt.Errorf("download failed: %s", resp.Status)
	}
	file, err := os.Create(dest)
	if err != nil {
		return 0, "", err
	}
	ok := false
	defer func() {
		file.Close()
		if !ok {
			_ = os.Remove(dest)
		}
	}()
	hash := sha256.New()
	reader := io.TeeReader(resp.Body, hash)
	var written int64
	if resp.ContentLength > 0 {
		written, err = copyProgress(file, reader, resp.ContentLength)
	} else {
		written, err = io.Copy(file, reader)
	}
	if err != nil {
		return 0, "", err
	}
	if err := file.Close(); err != nil {
		return 0, "", err
	}
	ok = true
	return written, fmt.Sprintf("%x", hash.Sum(nil)), nil
}

func copyProgress(dst io.Writer, src io.Reader, total int64) (int64, error) {
	buf := make([]byte, 1024*1024)
	var written int64
	last := time.Now()
	for {
		n, readErr := src.Read(buf)
		if n > 0 {
			wn, writeErr := dst.Write(buf[:n])
			written += int64(wn)
			if writeErr != nil {
				return written, writeErr
			}
			if wn != n {
				return written, io.ErrShortWrite
			}
			if time.Since(last) >= time.Second || written == total {
				fmt.Printf("\r%s / %s (%.1f%%)", humanBytes(written), humanBytes(total), 100*float64(written)/float64(total))
				last = time.Now()
			}
		}
		if readErr == io.EOF {
			fmt.Println()
			return written, nil
		}
		if readErr != nil {
			return written, readErr
		}
	}
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for n/div >= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGTPE"[exp])
}
