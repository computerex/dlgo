package diffusion

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// SafetensorsFile represents a parsed safetensors file.
type SafetensorsFile struct {
	Tensors    map[string]STensorInfo
	Data       []byte // mmap'd or loaded raw data
	DataOffset int64
}

// STensorInfo describes one tensor in the safetensors file.
type STensorInfo struct {
	DType   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

// OpenSafetensors parses a safetensors file and loads all tensor data into memory.
func OpenSafetensors(path string) (*SafetensorsFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer f.Close()

	// Read header length (8 bytes, little-endian uint64)
	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("read header length: %w", err)
	}
	if headerLen > 100*1024*1024 { // sanity: 100MB max header
		return nil, fmt.Errorf("header too large: %d bytes", headerLen)
	}

	// Read JSON header
	headerBytes := make([]byte, headerLen)
	if _, err := f.Read(headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Parse JSON into map (includes __metadata__ key we skip)
	raw := make(map[string]json.RawMessage)
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}

	tensors := make(map[string]STensorInfo)
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var info STensorInfo
		if err := json.Unmarshal(msg, &info); err != nil {
			return nil, fmt.Errorf("parse tensor %q: %w", name, err)
		}
		tensors[name] = info
	}

	// Read all tensor data
	dataOffset := int64(8 + headerLen)
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	dataSize := stat.Size() - dataOffset
	data := make([]byte, dataSize)
	if _, err := f.ReadAt(data, dataOffset); err != nil {
		return nil, fmt.Errorf("read tensor data: %w", err)
	}

	return &SafetensorsFile{
		Tensors:    tensors,
		Data:       data,
		DataOffset: dataOffset,
	}, nil
}

// GetFloat32 extracts a tensor as []float32. Supports F32, F16, BF16.
func (sf *SafetensorsFile) GetFloat32(name string) ([]float32, []int64, error) {
	info, ok := sf.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}

	raw := sf.Data[info.Offsets[0]:info.Offsets[1]]

	var numel int64 = 1
	for _, d := range info.Shape {
		numel *= d
	}

	out := make([]float32, numel)

	switch info.DType {
	case "F32":
		for i := int64(0); i < numel; i++ {
			out[i] = float32FromLE(raw[i*4:])
		}
	case "F16":
		for i := int64(0); i < numel; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		for i := int64(0); i < numel; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = float32FromBits(bits)
		}
	default:
		return nil, nil, fmt.Errorf("unsupported dtype %q for tensor %q", info.DType, name)
	}

	return out, info.Shape, nil
}

func float32FromLE(b []byte) float32 {
	return float32FromBits(binary.LittleEndian.Uint32(b))
}

func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	if exp == 0 {
		if mant == 0 {
			return float32FromBits(sign << 31)
		}
		// Subnormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		exp += 127 - 15
		return float32FromBits((sign << 31) | (exp << 23) | (mant << 13))
	} else if exp == 31 {
		// Inf/NaN
		return float32FromBits((sign << 31) | 0x7F800000 | (mant << 13))
	}
	exp += 127 - 15
	return float32FromBits((sign << 31) | (exp << 23) | (mant << 13))
}
