package server

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func assertSystemOneTiming(t *testing.T, header string) {
	t.Helper()
	metrics := strings.Split(header, ", ")
	if len(metrics) != 2 {
		t.Fatalf("Server-Timing = %q, want queue and decision metrics", header)
	}
	for i, name := range []string{"queue", "decision"} {
		prefix := name + ";dur="
		if !strings.HasPrefix(metrics[i], prefix) {
			t.Fatalf("metric = %q, want %s", metrics[i], prefix)
		}
		duration, err := strconv.ParseFloat(strings.TrimPrefix(metrics[i], prefix), 64)
		if err != nil || math.IsNaN(duration) || math.IsInf(duration, 0) || duration <= 0 {
			t.Fatalf("%s duration = %v, err = %v; want positive finite milliseconds", name, duration, err)
		}
	}
}

func TestSystemOneServerTiming(t *testing.T) {
	for _, failure := range []bool{false, true} {
		t.Run(fmt.Sprintf("failure=%v", failure), func(t *testing.T) {
			s := &Scheduler{}
			if err := s.acquireInference(context.Background()); err != nil {
				t.Fatal(err)
			}
			w := httptest.NewRecorder()
			started, done := make(chan struct{}), make(chan struct{})
			go func() {
				defer close(done)
				close(started)
				serveSystemOneResponse(context.Background(), w, s, func(buffer http.ResponseWriter) error {
					time.Sleep(2 * time.Millisecond)
					buffer.Header().Set("Content-Type", "application/json")
					// A buffered header must not overwrite the measured metrics.
					buffer.Header().Set("Server-Timing", "incorrect;dur=0")
					_, _ = buffer.Write([]byte(`{"value":"a"}`))
					if failure {
						return fmt.Errorf("wrapped: %w", &systemOneHTTPError{http.StatusBadRequest, "bad decision", "invalid_request_error"})
					}
					return nil
				})
			}()
			awaitInferenceTest(t, started)
			time.Sleep(2 * time.Millisecond)
			s.releaseInference()
			awaitInferenceTest(t, done)
			// Result checks the headers committed to the actual HTTP response,
			// not just the recorder's mutable header map.
			assertSystemOneTiming(t, w.Result().Header.Get("Server-Timing"))
			if failure {
				if w.Code != http.StatusBadRequest || strings.Contains(w.Body.String(), `"value"`) {
					t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
				}
			} else if w.Code != http.StatusOK || w.Body.String() != `{"value":"a"}` {
				t.Fatalf("response changed: status=%d body=%s", w.Code, w.Body.String())
			}
		})
	}
}

func TestSystemOneWrappedContextErrors(t *testing.T) {
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded} {
		t.Run(cause.Error(), func(t *testing.T) {
			s := &Scheduler{}
			w := httptest.NewRecorder()
			serveSystemOneResponse(context.Background(), w, s, func(buffer http.ResponseWriter) error {
				_, _ = buffer.Write([]byte("discard me"))
				return fmt.Errorf("engine: %w", fmt.Errorf("decision: %w", cause))
			})
			if w.Body.Len() != 0 || len(w.Header()) != 0 || w.Flushed {
				t.Fatalf("cancellation published a response: headers=%v body=%s", w.Header(), w.Body.String())
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if err := s.WithInferenceLockContext(ctx, func() error { return nil }); err != nil {
				t.Fatalf("cancellation did not release lock: %v", err)
			}
		})
	}
}
