package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestHandleSystemOneValidation verifies request validation for the typed
// System One endpoint without requiring a loaded model.
func TestHandleSystemOneValidation(t *testing.T) {
	s := &Server{mux: http.NewServeMux(), manager: NewModelManager()}

	post := func(body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/v1/systemone", bytes.NewBufferString(body))
		rec := httptest.NewRecorder()
		s.handleSystemOne(rec, req)
		return rec
	}

	// Method guard.
	req := httptest.NewRequest(http.MethodGet, "/v1/systemone", nil)
	rec := httptest.NewRecorder()
	s.handleSystemOne(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want 405", rec.Code)
	}

	// Invalid JSON.
	if rec := post("{not json"); rec.Code != http.StatusBadRequest {
		t.Fatalf("bad JSON status = %d, want 400", rec.Code)
	}

	// Missing required fields.
	if rec := post(`{"model":"m"}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("missing fields status = %d, want 400", rec.Code)
	}

	// Unknown type.
	body := `{"model":"m","readout":"r","state":"s","instructions":"i","type":"bogus"}`
	rec = post(body)
	if rec.Code != http.StatusNotFound {
		// Unknown type is checked after model lookup; no model manager entry
		// means we expect the model-not-found error first.
		t.Fatalf("unknown type with no model: status = %d, want 404", rec.Code)
	}

	// Choice with fewer than two criteria is rejected before evaluation once a
	// model would be found; without a model the 404 wins. Verify the error body
	// mentions the model.
	rec = post(`{"model":"missing","readout":"r","state":"s","instructions":"i","criteria":{"a":"A"}}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("no model status = %d, want 404", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "model not loaded") {
		t.Fatalf("body = %s, want model-not-loaded error", rec.Body.String())
	}
}

// TestSystemOneRequestShape documents the wire shape accepted by the endpoint.
func TestSystemOneRequestShape(t *testing.T) {
	var req SystemOneRequest
	body := `{"model":"qwen35","readout":"ro.json","type":"score","state":"s","instructions":"i","min":0,"max":10,"min_confidence":0.5}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	if req.Type != "score" || req.Min != 0 || req.Max != 10 || req.MinConfidence != 0.5 {
		t.Fatalf("decoded = %+v", req)
	}
}

// TestConvertWireQuestion verifies per-type validation for multi-question requests.
func TestConvertWireQuestion(t *testing.T) {
	// Choice conversion sorts keys and maps labels.
	q, err := convertWireQuestion(SystemOneQuestion{ID: "route", Prompt: "Route?", Criteria: map[string]string{"b": "B label", "a": "A label"}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(q.Choices) != 2 || q.Choices[0].Value != "a" || q.Choices[1].Value != "b" {
		t.Fatalf("choices = %+v, want sorted a,b", q.Choices)
	}

	// Noul conversion builds exactly yes/no.
	q, err = convertWireQuestion(SystemOneQuestion{Prompt: "Yes or no?", Type: "noul"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(q.Choices) != 2 || q.Choices[0].Value != "no" || q.Choices[1].Value != "yes" {
		t.Fatalf("noul choices = %+v, want code-ascending no,yes", q.Choices)
	}

	// Score conversion discretizes the range.
	q, err = convertWireQuestion(SystemOneQuestion{Prompt: "Rate", Type: "score", Min: 1, Max: 5}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(q.Choices) != 5 || q.Choices[0].Value != "1" || q.Choices[4].Value != "5" {
		t.Fatalf("score choices = %+v", q.Choices)
	}

	// Missing prompt is rejected.
	if _, err := convertWireQuestion(SystemOneQuestion{Type: "noul"}, 3); err == nil {
		t.Fatal("expected missing-prompt error")
	}

	// Choice with fewer than two criteria is rejected.
	if _, err := convertWireQuestion(SystemOneQuestion{Prompt: "p", Criteria: map[string]string{"a": "A"}}, 4); err == nil {
		t.Fatal("expected too-few-criteria error")
	}

	// Inverted score range is rejected.
	if _, err := convertWireQuestion(SystemOneQuestion{Prompt: "p", Type: "score", Min: 5, Max: 1}, 5); err == nil {
		t.Fatal("expected inverted-range error")
	}

	// Unknown type is rejected.
	if _, err := convertWireQuestion(SystemOneQuestion{Prompt: "p", Type: "bogus"}, 6); err == nil {
		t.Fatal("expected unknown-type error")
	}
}