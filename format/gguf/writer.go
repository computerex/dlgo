package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// WriteTensor describes one tensor to serialize into a GGUF file.
type WriteTensor struct {
	Name       string
	Dimensions []int64
	Type       GGMLType
	Data       []byte // already in the on-disk layout for Type
}

// countingWriter tracks the number of bytes written to an io.Writer.
type countingWriter struct {
	w     io.Writer
	count int64
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.count += int64(n)
	return n, err
}

// WriteGGUF writes a complete GGUF v3 file to path. Metadata values must be one
// of the supported scalar Go types (string, bool, uint8/16/32/64, int8/16/32/64,
// float32, float64) or a []T of a supported scalar type (serialized as an array).
// Tensor data is written after the header, aligned to the given alignment.
//
// This is used to export independently trained System One decision readouts as
// standalone GGUF artifacts without touching the frozen backbone model file.
func WriteGGUF(path string, metadata map[string]interface{}, tensors []WriteTensor, alignment uint32) error {
	if alignment == 0 {
		alignment = 32
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create GGUF: %w", err)
	}
	defer file.Close()
	bw := bufio.NewWriter(file)
	cw := &countingWriter{w: bw}

	// Header: magic, version, tensor count, metadata count.
	if _, err := cw.Write([]byte("GGUF")); err != nil {
		return err
	}
	if err := binary.Write(cw, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}
	if err := binary.Write(cw, binary.LittleEndian, int64(len(tensors))); err != nil {
		return err
	}
	if err := binary.Write(cw, binary.LittleEndian, int64(len(metadata))); err != nil {
		return err
	}

	// Metadata key-value pairs.
	for key, value := range metadata {
		if err := writeString(cw, key); err != nil {
			return err
		}
		if err := writeValue(cw, value); err != nil {
			return fmt.Errorf("metadata %q: %w", key, err)
		}
	}

	// Tensor info records. Offsets are relative to the start of the tensor data
	// blob, which begins after the header and its alignment padding.
	offset := uint64(0)
	for i := range tensors {
		t := &tensors[i]
		if err := writeString(cw, t.Name); err != nil {
			return err
		}
		if err := binary.Write(cw, binary.LittleEndian, uint32(len(t.Dimensions))); err != nil {
			return err
		}
		for _, d := range t.Dimensions {
			if err := binary.Write(cw, binary.LittleEndian, d); err != nil {
				return err
			}
		}
		if err := binary.Write(cw, binary.LittleEndian, uint32(t.Type)); err != nil {
			return err
		}
		if err := binary.Write(cw, binary.LittleEndian, offset); err != nil {
			return err
		}
		offset += uint64(len(t.Data))
	}

	// Pad the header to the alignment boundary.
	if err := padTo(cw, alignment); err != nil {
		return err
	}

	// Tensor data blob.
	for i := range tensors {
		if _, err := cw.Write(tensors[i].Data); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// writeString writes a GGUF string (uint64 length prefix + UTF-8 bytes).
func writeString(w io.Writer, s string) error {
	if err := binary.Write(w, binary.LittleEndian, uint64(len(s))); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

// writeValue writes a single GGUF metadata value, inferring its type.
func writeValue(w io.Writer, value interface{}) error {
	switch v := value.(type) {
	case string:
		if err := binary.Write(w, binary.LittleEndian, uint32(GGUFTypeString)); err != nil {
			return err
		}
		return writeString(w, v)
	case bool:
		if err := binary.Write(w, binary.LittleEndian, uint32(GGUFTypeBool)); err != nil {
			return err
		}
		b := int8(0)
		if v {
			b = 1
		}
		return binary.Write(w, binary.LittleEndian, b)
	case uint8:
		return writeScalar(w, GGUFTypeUint8, v)
	case int8:
		return writeScalar(w, GGUFTypeInt8, v)
	case uint16:
		return writeScalar(w, GGUFTypeUint16, v)
	case int16:
		return writeScalar(w, GGUFTypeInt16, v)
	case uint32:
		return writeScalar(w, GGUFTypeUint32, v)
	case int32:
		return writeScalar(w, GGUFTypeInt32, v)
	case uint64:
		return writeScalar(w, GGUFTypeUint64, v)
	case int64:
		return writeScalar(w, GGUFTypeInt64, v)
	case float32:
		return writeScalar(w, GGUFTypeFloat32, v)
	case float64:
		return writeScalar(w, GGUFTypeFloat64, v)
	case []string:
		return writeArray(w, GGUFTypeString, v)
	case []bool:
		return writeArray(w, GGUFTypeBool, v)
	case []uint8:
		return writeArray(w, GGUFTypeUint8, v)
	case []int8:
		return writeArray(w, GGUFTypeInt8, v)
	case []uint16:
		return writeArray(w, GGUFTypeUint16, v)
	case []int16:
		return writeArray(w, GGUFTypeInt16, v)
	case []uint32:
		return writeArray(w, GGUFTypeUint32, v)
	case []int32:
		return writeArray(w, GGUFTypeInt32, v)
	case []uint64:
		return writeArray(w, GGUFTypeUint64, v)
	case []int64:
		return writeArray(w, GGUFTypeInt64, v)
	case []float32:
		return writeArray(w, GGUFTypeFloat32, v)
	case []float64:
		return writeArray(w, GGUFTypeFloat64, v)
	default:
		return fmt.Errorf("unsupported metadata value type %T", value)
	}
}

func writeScalar(w io.Writer, t GGUFType, value interface{}) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(t)); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, value)
}

func writeArray(w io.Writer, elemType GGUFType, values interface{}) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(GGUFTypeArray)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(elemType)); err != nil {
		return err
	}
	switch v := values.(type) {
	case []string:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := writeString(w, e); err != nil {
				return err
			}
		}
	case []bool:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			b := int8(0)
			if e {
				b = 1
			}
			if err := binary.Write(w, binary.LittleEndian, b); err != nil {
				return err
			}
		}
	default:
		return writeArrayScalars(w, v)
	}
	return nil
}

func writeArrayScalars(w io.Writer, values interface{}) error {
	// Write the length then each element. The element type was already written.
	switch v := values.(type) {
	case []uint8:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []int8:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []uint16:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []int16:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []uint32:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []int32:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []uint64:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []int64:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []float32:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	case []float64:
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, e := range v {
			if err := binary.Write(w, binary.LittleEndian, e); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported array element type %T", values)
	}
	return nil
}

// padTo writes zero bytes until the counting writer position is a multiple of
// alignment.
func padTo(cw *countingWriter, alignment uint32) error {
	remainder := cw.count % int64(alignment)
	if remainder == 0 {
		return nil
	}
	pad := make([]byte, int64(alignment)-remainder)
	_, err := cw.Write(pad)
	return err
}

// Float32Bytes packs float32 values into their little-endian byte layout.
func Float32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// TensorSizeBytes returns the on-disk byte size of a tensor with the given type
// and dimensions.
func TensorSizeBytes(t GGMLType, dims []int64) (int64, error) {
	var elements int64 = 1
	for _, d := range dims {
		if d <= 0 {
			return 0, fmt.Errorf("invalid dimension %d", d)
		}
		elements *= d
	}
	var nbytes int64
	switch t {
	case GGMLTypeF32:
		nbytes = elements * 4
	case GGMLTypeF16, GGMLTypeBF16:
		nbytes = elements * 2
	case GGMLTypeF64:
		nbytes = elements * 8
	case GGMLTypeQ8_0:
		nbytes = (elements / 32) * 34
	case GGMLTypeQ4_0:
		nbytes = (elements / 32) * 18
	default:
		return 0, fmt.Errorf("unsupported tensor type %d for size computation", t)
	}
	return nbytes, nil
}

// ReadTensorData reads the raw on-disk bytes of one tensor from an opened GGUF
// file. The caller must interpret the bytes according to the tensor's Type.
func ReadTensorData(f *os.File, gf *GGUFFile, name string) ([]byte, error) {
	for _, ti := range gf.Tensors {
		if ti.Name != name {
			continue
		}
		nbytes, err := TensorSizeBytes(ti.Type, ti.Dimensions)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		data := make([]byte, nbytes)
		offset := gf.DataOffset + int64(ti.Offset)
		if _, err := f.ReadAt(data, offset); err != nil {
			return nil, fmt.Errorf("read tensor %q: %w", name, err)
		}
		return data, nil
	}
	return nil, fmt.Errorf("tensor %q not found", name)
}

// Float32Tensor unpacks little-endian F32 tensor bytes into float32 values.
func Float32Tensor(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("F32 tensor byte length %d is not a multiple of 4", len(data))
	}
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}
