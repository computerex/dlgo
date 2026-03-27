package server

import (
	"fmt"
	"log"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"

	"github.com/computerex/dlgo/mmap"
	"github.com/computerex/dlgo/models/llm"
)

// LoadedModel holds all state for a single loaded model.
type LoadedModel struct {
	ID          string
	Path        string
	CPUPipeline *llm.Pipeline
	GpuPipeline GpuPipelineInterface
	Scheduler   *Scheduler
	UseGPU      bool
	Info        ModelObject
}

// GpuPipelineInterface abstracts the GPU pipeline to avoid build tag issues.
// The actual implementation is in model_manager_gpu.go (vulkan build).
type GpuPipelineInterface interface {
	GenerateDetailed(prompt string, cfg llm.GenerateConfig) (*llm.GenerateResult, error)
	Free()
}

// AvailableModel represents a model that can be loaded but isn't yet.
type AvailableModel struct {
	ID   string `json:"id"`
	Path string `json:"path"`
}

// VRAMStatus holds GPU memory information for health reporting.
type VRAMStatus struct {
	TotalMB float64 `json:"total_mb"`
	FreeMB  float64 `json:"free_mb"`
	UsedMB  float64 `json:"used_mb"`
}

// ModelManager manages multiple loaded models.
type ModelManager struct {
	mu             sync.RWMutex
	models         map[string]*LoadedModel
	available      map[string]*AvailableModel
	gpuInit        func() error
	gpuNewPipeline func(pipe *llm.Pipeline) (GpuPipelineInterface, error)
	vramStatus     func() *VRAMStatus
}

// SetVRAMStatusFunc registers a callback to query GPU VRAM usage.
func (mm *ModelManager) SetVRAMStatusFunc(f func() *VRAMStatus) {
	mm.vramStatus = f
}

// GetVRAMStatus returns current VRAM usage, or nil if GPU is not active.
func (mm *ModelManager) GetVRAMStatus() *VRAMStatus {
	if mm.vramStatus == nil {
		return nil
	}
	return mm.vramStatus()
}

// NewModelManager creates a new model manager.
func NewModelManager() *ModelManager {
	return &ModelManager{
		models:    make(map[string]*LoadedModel),
		available: make(map[string]*AvailableModel),
	}
}

// RegisterAvailableModel registers a model as available to load.
func (mm *ModelManager) RegisterAvailableModel(id, path string) {
	mm.mu.Lock()
	defer mm.mu.Unlock()
	mm.available[id] = &AvailableModel{ID: id, Path: path}
}

// ListAvailableModels returns all models that can be loaded.
func (mm *ModelManager) ListAvailableModels() []AvailableModel {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	result := make([]AvailableModel, 0, len(mm.available))
	for _, m := range mm.available {
		// Skip if already loaded
		if _, loaded := mm.models[m.ID]; !loaded {
			result = append(result, *m)
		}
	}
	return result
}

// SetGPUFunctions allows the GPU layer to register its init and pipeline creation functions.
func (mm *ModelManager) SetGPUFunctions(
	initFn func() error,
	newPipelineFn func(pipe *llm.Pipeline) (GpuPipelineInterface, error),
) {
	mm.gpuInit = initFn
	mm.gpuNewPipeline = newPipelineFn
}

// LoadModel loads a GGUF model and optionally sets up GPU acceleration.
func (mm *ModelManager) LoadModel(id, path string, useGPU bool, contextLen int) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	if _, exists := mm.models[id]; exists {
		return fmt.Errorf("model %q already loaded", id)
	}

	log.Printf("Loading model %q from %s (gpu=%v, ctx=%d)", id, path, useGPU, contextLen)

	pipe, err := llm.NewPipeline(path, contextLen)
	if err != nil {
		return fmt.Errorf("load pipeline: %w", err)
	}

	loaded := &LoadedModel{
		ID:          id,
		Path:        path,
		CPUPipeline: pipe,
		UseGPU:      useGPU,
		Info: ModelObject{
			ID:      id,
			Object:  "model",
			Created: nowUnix(),
			OwnedBy: "dlgo",
			Arch:    pipe.Model.Config.Architecture,
			GPU:     useGPU,
			Path:    path,
		},
	}

	if useGPU && mm.gpuNewPipeline != nil {
		if mm.gpuInit != nil {
			if err := mm.gpuInit(); err != nil {
				log.Printf("GPU init failed, falling back to CPU: %v", err)
				useGPU = false
			}
		}

		if useGPU {
			// Free CPU-side buffers BEFORE GPU pipeline creation.
			// The GPU pipeline allocates its own KV cache, run state, and batch
			// state in VRAM. Keeping CPU buffers alive during GPU allocation
			// doubles peak memory, which can crash the system with large contexts
			// (e.g., 256K native context = ~8.5 GB CPU + GPU VRAM).
			// We save the needed info and rebuild CPU buffers only if GPU fails.
			savedMaxSeqLen := pipe.MaxSeqLen
			pipe.FreeForGPU()
			runtime.GC()
			debug.FreeOSMemory()
			mmap.TrimWorkingSet()

			gpuPipe, err := mm.gpuNewPipeline(pipe)
			if err != nil {
				log.Printf("GPU pipeline creation failed, falling back to CPU: %v", err)
				// Rebuild CPU buffers for CPU-only inference.
				// Note: pipe.MaxSeqLen may have been reduced by the GPU pipeline
				// estimator; use the (possibly reduced) value.
				if pipe.MaxSeqLen <= 0 {
					pipe.MaxSeqLen = savedMaxSeqLen
				}
				pipe.RebuildBuffers()
			} else {
				loaded.GpuPipeline = gpuPipe
			}
		}
	}

	loaded.Info.GPU = loaded.GpuPipeline != nil
	loaded.Scheduler = NewScheduler(loaded)
	mm.models[id] = loaded

	log.Printf("Model %q loaded successfully (arch=%s, layers=%d, gpu=%v, ctx=%d)",
		id, pipe.Model.Config.Architecture, pipe.Model.Config.NumLayers, loaded.GpuPipeline != nil, pipe.MaxSeqLen)
	return nil
}

// UnloadModel removes and cleans up a loaded model.
func (mm *ModelManager) UnloadModel(id string) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	m, exists := mm.models[id]
	if !exists {
		return fmt.Errorf("model %q not found", id)
	}

	m.Scheduler.Stop()
	if m.GpuPipeline != nil {
		m.GpuPipeline.Free()
	}
	if m.CPUPipeline.Model.MmapFile != nil {
		m.CPUPipeline.Model.Close()
	}

	delete(mm.models, id)

	// Aggressively reclaim memory after unload. The GC frees heap objects
	// (pinned layers, KV cache, run state) and TrimWorkingSet evicts the
	// mmap page cache so the OS sees the RAM as free before the next load.
	runtime.GC()
	debug.FreeOSMemory()
	mmap.TrimWorkingSet()

	log.Printf("Model %q unloaded", id)
	return nil
}

// GetModel returns a loaded model by ID, or nil if not found.
func (mm *ModelManager) GetModel(id string) *LoadedModel {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	// Exact match first
	if m, ok := mm.models[id]; ok {
		return m
	}

	// Fuzzy match: check if any model ID contains the requested ID
	for k, m := range mm.models {
		if strings.Contains(strings.ToLower(k), strings.ToLower(id)) {
			return m
		}
	}
	return nil
}

// ListModels returns metadata for all loaded models.
func (mm *ModelManager) ListModels() []ModelObject {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	result := make([]ModelObject, 0, len(mm.models))
	for _, m := range mm.models {
		result = append(result, m.Info)
	}
	return result
}
