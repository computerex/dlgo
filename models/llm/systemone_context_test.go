package llm

import (
	"context"
	"errors"
	"testing"
)

func TestSystemOneContextCancellation(t *testing.T) {
	if err := (SystemOneConfig{}).Err(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	cfg := SystemOneConfig{Context: ctx}
	var p *Pipeline
	if _, err := p.SystemOneMulti("", nil, cfg, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("multi: %v", err)
	}
	if _, _, _, err := p.SystemOneHidden("", SystemOneQuestion{}, cfg); !errors.Is(err, context.Canceled) {
		t.Fatalf("hidden: %v", err)
	}
}
