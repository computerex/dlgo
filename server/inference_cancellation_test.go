package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/computerex/dlgo/models/llm"
)

func awaitInferenceTest(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for inference operation")
	}
}

func TestInferenceLockCancellation(t *testing.T) {
	s := &Scheduler{}
	if err := s.acquireInference(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer s.releaseInference()
	ctx, cancel := context.WithCancel(context.Background())
	started, done := make(chan struct{}), make(chan struct{})
	var err error
	called := false
	go func() {
		defer close(done)
		close(started)
		err = s.WithInferenceLockContext(ctx, func() error { called = true; return nil })
	}()
	awaitInferenceTest(t, started)
	cancel()
	awaitInferenceTest(t, done)
	if !errors.Is(err, context.Canceled) || called {
		t.Fatalf("err = %v, callback called = %v", err, called)
	}
}

func TestInferenceLockDeadline(t *testing.T) {
	s := &Scheduler{}
	if err := s.acquireInference(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer s.releaseInference()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := s.WithInferenceLockContext(ctx, func() error {
		t.Fatal("expired waiter ran inference")
		return nil
	}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
}

func TestInferenceLockPreCanceledAndReusable(t *testing.T) {
	s := &Scheduler{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for i := 0; i < 100; i++ {
		if err := s.WithInferenceLockContext(ctx, func() error { t.Fatal("canceled callback ran"); return nil }); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
	}
	want := errors.New("callback failure")
	if err := s.WithInferenceLock(func() error { return want }); err != want {
		t.Fatalf("err = %v", err)
	}
	ctx, cancel = context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := s.WithInferenceLockContext(ctx, func() error { return nil }); err != nil {
		t.Fatal(err)
	}
}

func TestInferenceLockExclusion(t *testing.T) {
	s := &Scheduler{}
	var wg sync.WaitGroup
	active, maximum := 0, 0 // protected by the inference gate under test
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			fn := func() error {
				active++
				if active > maximum {
					maximum = active
				}
				time.Sleep(time.Millisecond)
				active--
				return nil
			}
			if i%2 == 0 {
				_ = s.WithInferenceLock(fn)
			} else {
				_ = s.WithInferenceLockContext(context.Background(), fn)
			}
		}(i)
	}
	wg.Wait()
	if maximum != 1 || active != 0 {
		t.Fatalf("maximum = %d, active = %d", maximum, active)
	}
}

func TestGenerationCanceledWaitingForInferenceLock(t *testing.T) {
	// A nil model deliberately panics if generation reaches model access.
	s := &Scheduler{}
	if err := s.acquireInference(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer s.releaseInference()
	ctx, cancel := context.WithCancel(context.Background())
	req := &InferenceRequest{Ctx: ctx, Output: make(chan StreamEvent)}
	done := make(chan struct{})
	go func() { defer close(done); s.processRequest(req) }()
	cancel()
	awaitInferenceTest(t, done)
	if _, open := <-req.Output; open {
		t.Fatal("canceled generation output was not closed")
	}
}

type blockingSystemOneWriter struct {
	*httptest.ResponseRecorder
	entered chan struct{}
	release chan struct{}
}

func (w *blockingSystemOneWriter) Write(data []byte) (int, error) {
	close(w.entered)
	<-w.release
	return w.ResponseRecorder.Write(data)
}

func TestSystemOneResponseWriteOutsideInferenceLock(t *testing.T) {
	s := &Scheduler{}
	w := &blockingSystemOneWriter{httptest.NewRecorder(), make(chan struct{}), make(chan struct{})}
	defer close(w.release)
	questions := []parsedQuestion{{ID: "decision", Type: "choice"}}
	results := []*llm.SystemOneResult{{QuestionID: "decision", Value: "a", Confidence: .8, PromptTokens: 7, Choices: []llm.ChoiceScore{{Value: "a", Probability: .8}, {Value: "b", Probability: .2}}}}
	expected := httptest.NewRecorder()
	writeSystemOneResponse(expected, "m", questions, results, 7)
	done := make(chan struct{})
	go func() {
		defer close(done)
		serveSystemOneResponse(context.Background(), w, s, func(buffer http.ResponseWriter) error {
			buffer.Header().Set("Content-Type", "application/json")
			writeSystemOneResponse(buffer, "m", questions, results, 7)
			return nil
		})
	}()
	awaitInferenceTest(t, w.entered)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := s.WithInferenceLockContext(ctx, func() error { return nil }); err != nil {
		t.Fatalf("network write held inference lock: %v", err)
	}
	w.release <- struct{}{}
	awaitInferenceTest(t, done)
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "application/json" || w.Body.String() != expected.Body.String() {
		t.Fatalf("response changed: status=%d headers=%v body=%s", w.Code, w.Header(), w.Body.String())
	}
}

func TestSystemOneResponseCancellation(t *testing.T) {
	for _, phase := range []string{"waiting", "active"} {
		t.Run(phase, func(t *testing.T) {
			s := &Scheduler{}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			w := httptest.NewRecorder()
			called := false
			if phase == "waiting" {
				if err := s.acquireInference(context.Background()); err != nil {
					t.Fatal(err)
				}
				defer s.releaseInference()
			}
			done := make(chan struct{})
			go func() {
				defer close(done)
				serveSystemOneResponse(ctx, w, s, func(buffer http.ResponseWriter) error {
					called = true
					cancel()
					buffer.Header().Set("Content-Type", "application/json")
					_, _ = buffer.Write([]byte(`{"value":"a"}`))
					return nil
				})
			}()
			if phase == "waiting" {
				cancel()
			}
			awaitInferenceTest(t, done)
			if called != (phase == "active") || w.Body.Len() != 0 || len(w.Header()) != 0 {
				t.Fatalf("called=%v headers=%v body=%s", called, w.Header(), w.Body.String())
			}
		})
	}
}

func TestSystemOneBufferedErrorResponse(t *testing.T) {
	for _, typed := range []bool{false, true} {
		s := &Scheduler{}
		w := httptest.NewRecorder()
		serveSystemOneResponse(context.Background(), w, s, func(buffer http.ResponseWriter) error {
			_, _ = buffer.Write([]byte("partial response must be discarded"))
			if typed {
				return &systemOneHTTPError{http.StatusBadRequest, "bad decision", "invalid_request_error"}
			}
			return errors.New("engine failure")
		})
		wantStatus := http.StatusInternalServerError
		if typed {
			wantStatus = http.StatusBadRequest
		}
		if w.Code != wantStatus || !json.Valid(w.Body.Bytes()) {
			t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
		}
	}
}
