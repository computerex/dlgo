//go:build cgo && vulkan

package gpu

import (
	"fmt"
)

// GpuStateSnapshot is a device-side copy of every recurrent inference state —
// the used KV-cache region, per-layer SSM recurrent state, and SSM conv
// history — captured after a shared-prefix prefill. Restoring it lets several
// System One suffix passes branch from one prefill without re-reading the
// prefix, mirroring the CPU pipeline's KV/SSM Clone-branch behavior.
type GpuStateSnapshot struct {
	kvLen         int   // positions of KV captured
	kvKeys        []Buf // indexed by layer; 0 when the layer has no KV cache
	kvVals        []Buf
	kvBytes       uint64 // per-buffer byte size of the captured region
	ssmState      []Buf  // indexed by layer; 0 when the layer has no SSM state
	ssmConv       []Buf
	ssmStateBytes uint64
	ssmConvBytes  uint64
}

// SnapshotState copies the currently used KV region and all SSM state into
// fresh device buffers. Call it immediately after prefilling the shared
// prefix, before any suffix pass mutates the live state.
func (p *GpuPipeline) SnapshotState(npos int) (*GpuStateSnapshot, error) {
	p.invalidateChatPrefix()
	if npos <= 0 || npos > p.KVCache.MaxSeq {
		return nil, fmt.Errorf("gpu: cannot snapshot state at %d positions (max %d)", npos, p.KVCache.MaxSeq)
	}
	if p.KVCache.BytesPerElem == 0 {
		p.KVCache.BytesPerElem = 2
	}
	// Snapshot/restore copies each layer's first npos rows and assumes they are
	// laid out by position. A ring cache breaks that layout, and the restore
	// would also have to rewind rows that later positions have since overwritten
	// (a ring only guarantees the last `cap` positions). Ring layers are only
	// allocated for models with no SSM and no MoE, which are also the models that
	// never reach this path, so refuse rather than silently corrupt.
	if p.KVCache.RingLayers() > 0 {
		return nil, fmt.Errorf("gpu: state snapshot is not supported with a ring KV cache (%d layers)", p.KVCache.RingLayers())
	}
	// The snapshot covers KV + SSM state only. LFM2/LFM2MoE also carries a
	// per-layer short-convolution rolling history, so restoring one of these
	// snapshots would silently resume with a stale conv window. Refuse instead:
	// no path needs it for these models (cross-request prefix reuse is already
	// gated off when HasMoE, and short-conv models are MoE models).
	if p.CPUModel != nil && p.CPUModel.Config.HasShortConv() {
		return nil, fmt.Errorf("gpu: state snapshot is not supported for short-convolution models")
	}
	Sync()
	snap := &GpuStateSnapshot{kvLen: npos}
	layers := len(p.KVCache.KeyBufs)
	snap.kvKeys = make([]Buf, layers)
	snap.kvVals = make([]Buf, layers)
	snap.kvBytes = uint64(npos) * uint64(p.KVCache.KVDim) * uint64(p.KVCache.BytesPerElem)

	BeginBatch()
	for l := 0; l < layers; l++ {
		if p.KVCache.KeyBufs[l] == 0 || p.KVCache.ValBufs[l] == 0 {
			continue
		}
		k := Alloc(snap.kvBytes)
		v := Alloc(snap.kvBytes)
		if k == 0 || v == 0 {
			EndBatch()
			freeBuf(k)
			freeBuf(v)
			snap.Free()
			return nil, fmt.Errorf("gpu: KV snapshot alloc (%d bytes/buffer) failed at layer %d", snap.kvBytes, l)
		}
		if err := CopyRegion(k, 0, p.KVCache.KeyBufs[l], 0, snap.kvBytes); err != nil {
			EndBatch()
			freeBuf(k)
			freeBuf(v)
			snap.Free()
			return nil, fmt.Errorf("gpu: KV snapshot copy layer %d: %w", l, err)
		}
		if err := CopyRegion(v, 0, p.KVCache.ValBufs[l], 0, snap.kvBytes); err != nil {
			EndBatch()
			freeBuf(k)
			freeBuf(v)
			snap.Free()
			return nil, fmt.Errorf("gpu: KV snapshot copy layer %d: %w", l, err)
		}
		snap.kvKeys[l], snap.kvVals[l] = k, v
	}

	if p.HasSSM {
		cfg := p.CPUModel.Config
		numHeads := cfg.SSMTimeStepRank
		numKVGroups := cfg.SSMGroupCount
		if numKVGroups <= 0 {
			numKVGroups = numHeads
		}
		headVDim := cfg.SSMInnerSize / numHeads
		headKDim := cfg.SSMStateSize
		keyDim := numKVGroups * headKDim
		qkvDim := keyDim*2 + numHeads*headVDim
		convK := cfg.SSMConvKernel
		snap.ssmStateBytes = uint64(numHeads*headKDim*headVDim) * 4
		snap.ssmConvBytes = uint64(convK*qkvDim) * 4
		snap.ssmState = make([]Buf, cfg.NumLayers)
		snap.ssmConv = make([]Buf, cfg.NumLayers)
		for l := 0; l < cfg.NumLayers; l++ {
			gl := &p.GpuModel.Layers[l]
			if gl.SSMState == 0 || gl.SSMConvBuf == 0 {
				continue
			}
			s := Alloc(snap.ssmStateBytes)
			c := Alloc(snap.ssmConvBytes)
			if s == 0 || c == 0 {
				EndBatch()
				freeBuf(s)
				freeBuf(c)
				snap.Free()
				return nil, fmt.Errorf("gpu: SSM snapshot alloc failed at layer %d", l)
			}
			if err := CopyRegion(s, 0, gl.SSMState, 0, snap.ssmStateBytes); err != nil {
				EndBatch()
				freeBuf(s)
				freeBuf(c)
				snap.Free()
				return nil, fmt.Errorf("gpu: SSM snapshot copy layer %d: %w", l, err)
			}
			if err := CopyRegion(c, 0, gl.SSMConvBuf, 0, snap.ssmConvBytes); err != nil {
				EndBatch()
				freeBuf(s)
				freeBuf(c)
				snap.Free()
				return nil, fmt.Errorf("gpu: SSM conv snapshot copy layer %d: %w", l, err)
			}
			snap.ssmState[l], snap.ssmConv[l] = s, c
		}
	}
	EndBatch()
	Sync()
	return snap, nil
}

// snapshotRecurrentState copies only the GPU SSM recurrent state and conv
// history into fresh device buffers, recording kvLen = npos. KV rows are NOT
// copied; the caller must keep KV[0, npos) resident. Used by cross-request chat
// prefix reuse, where the resident KV is reused as-is and only the recurrent
// state needs to be rewound to the prefix boundary.
func (p *GpuPipeline) snapshotRecurrentState(npos int) (*GpuStateSnapshot, error) {
	if npos <= 0 || npos > p.KVCache.MaxSeq {
		return nil, fmt.Errorf("gpu: cannot snapshot recurrent state at %d positions (max %d)", npos, p.KVCache.MaxSeq)
	}
	if !p.HasSSM {
		return nil, fmt.Errorf("gpu: recurrent snapshot requires SSM state")
	}
	if p.CPUModel.Config.HasShortConv() {
		return nil, fmt.Errorf("gpu: recurrent snapshot is not supported for short-convolution models")
	}
	Sync()
	snap := &GpuStateSnapshot{kvLen: npos}
	cfg := p.CPUModel.Config
	numHeads := cfg.SSMTimeStepRank
	numKVGroups := cfg.SSMGroupCount
	if numKVGroups <= 0 {
		numKVGroups = numHeads
	}
	headVDim := cfg.SSMInnerSize / numHeads
	headKDim := cfg.SSMStateSize
	keyDim := numKVGroups * headKDim
	qkvDim := keyDim*2 + numHeads*headVDim
	convK := cfg.SSMConvKernel
	snap.ssmStateBytes = uint64(numHeads*headKDim*headVDim) * 4
	snap.ssmConvBytes = uint64(convK*qkvDim) * 4
	snap.ssmState = make([]Buf, cfg.NumLayers)
	snap.ssmConv = make([]Buf, cfg.NumLayers)

	BeginBatch()
	for l := 0; l < cfg.NumLayers; l++ {
		gl := &p.GpuModel.Layers[l]
		if gl.SSMState == 0 || gl.SSMConvBuf == 0 {
			continue
		}
		s := Alloc(snap.ssmStateBytes)
		c := Alloc(snap.ssmConvBytes)
		if s == 0 || c == 0 {
			EndBatch()
			freeBuf(s)
			freeBuf(c)
			snap.Free()
			return nil, fmt.Errorf("gpu: recurrent snapshot alloc failed at layer %d", l)
		}
		if err := CopyRegion(s, 0, gl.SSMState, 0, snap.ssmStateBytes); err != nil {
			EndBatch()
			freeBuf(s)
			freeBuf(c)
			snap.Free()
			return nil, fmt.Errorf("gpu: recurrent state snapshot copy layer %d: %w", l, err)
		}
		if err := CopyRegion(c, 0, gl.SSMConvBuf, 0, snap.ssmConvBytes); err != nil {
			EndBatch()
			freeBuf(s)
			freeBuf(c)
			snap.Free()
			return nil, fmt.Errorf("gpu: recurrent conv snapshot copy layer %d: %w", l, err)
		}
		snap.ssmState[l], snap.ssmConv[l] = s, c
	}
	EndBatch()
	Sync()
	return snap, nil
}

// RestoreState copies all captured KV, SSM, and conv state back into the live
// pipeline and resets KVCache.Len to the shared prefix boundary. Use it when
// switching cached prefixes or after any operation that can overwrite prefix KV.
func (p *GpuPipeline) RestoreState(s *GpuStateSnapshot) error {
	p.invalidateChatPrefix()
	return p.restoreState(s, true)
}

// RestoreRecurrentState restores only SSM state and conv history, and resets
// KVCache.Len to the snapshot's prefix boundary without copying any KV bytes.
// The caller must guarantee that this snapshot's KV prefix is still resident:
// either it was just prefilled or RestoreState installed it, and intervening
// branches wrote only positions at or beyond the prefix boundary. Suffix KV is
// left untouched and must be overwritten before it is read by the next branch.
// A cache lookup alone does NOT establish residency. Standalone GPU inference,
// ResetState, or switching prefixes invalidates this guarantee. Pipeline use
// (including restore and subsequent inference) must be externally serialized.
func (p *GpuPipeline) RestoreRecurrentState(s *GpuStateSnapshot) error {
	p.invalidateChatPrefix()
	return p.restoreState(s, false)
}

func (p *GpuPipeline) restoreState(s *GpuStateSnapshot, restoreKV bool) error {
	if s == nil {
		return fmt.Errorf("gpu: nil state snapshot")
	}
	if p == nil || p.KVCache == nil || s.kvLen <= 0 || s.kvLen > p.KVCache.MaxSeq {
		return fmt.Errorf("gpu: invalid state snapshot prefix length %d", s.kvLen)
	}
	Sync()
	BeginBatch()
	if restoreKV {
		for l := range s.kvKeys {
			if s.kvKeys[l] == 0 {
				continue
			}
			if err := CopyRegion(p.KVCache.KeyBufs[l], 0, s.kvKeys[l], 0, s.kvBytes); err != nil {
				EndBatch()
				return fmt.Errorf("gpu: KV restore layer %d: %w", l, err)
			}
			if err := CopyRegion(p.KVCache.ValBufs[l], 0, s.kvVals[l], 0, s.kvBytes); err != nil {
				EndBatch()
				return fmt.Errorf("gpu: KV restore layer %d: %w", l, err)
			}
		}
	}
	for l := range s.ssmState {
		if s.ssmState[l] == 0 {
			continue
		}
		if err := CopyRegion(p.GpuModel.Layers[l].SSMState, 0, s.ssmState[l], 0, s.ssmStateBytes); err != nil {
			EndBatch()
			return fmt.Errorf("gpu: SSM restore layer %d: %w", l, err)
		}
		if err := CopyRegion(p.GpuModel.Layers[l].SSMConvBuf, 0, s.ssmConv[l], 0, s.ssmConvBytes); err != nil {
			EndBatch()
			return fmt.Errorf("gpu: SSM conv restore layer %d: %w", l, err)
		}
	}
	EndBatch()
	p.KVCache.Len = s.kvLen
	Sync()
	return nil
}

// Free releases the snapshot's device buffers.
func (s *GpuStateSnapshot) Free() {
	if s == nil {
		return
	}
	for _, b := range s.kvKeys {
		freeBuf(b)
	}
	for _, b := range s.kvVals {
		freeBuf(b)
	}
	for _, b := range s.ssmState {
		freeBuf(b)
	}
	for _, b := range s.ssmConv {
		freeBuf(b)
	}
	s.kvKeys, s.kvVals, s.ssmState, s.ssmConv = nil, nil, nil, nil
}
