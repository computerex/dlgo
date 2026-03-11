package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

// Server is the main HTTP server for the dlgo inference engine.
type Server struct {
	mux     *http.ServeMux
	manager *ModelManager
	addr    string
}

// NewServer creates a new inference server.
func NewServer(addr string, manager *ModelManager) *Server {
	s := &Server{
		mux:     http.NewServeMux(),
		manager: manager,
		addr:    addr,
	}
	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/chat/completions", s.cors(s.handleChatCompletions))
	s.mux.HandleFunc("/v1/models", s.cors(s.handleModels))
	s.mux.HandleFunc("/health", s.cors(s.handleHealth))
}

// SetFrontendHandler sets a handler to serve the frontend app at /.
func (s *Server) SetFrontendHandler(h http.Handler) {
	s.mux.Handle("/", h)
}

func (s *Server) ListenAndServe() error {
	log.Printf("dlgo server listening on %s", s.addr)
	return http.ListenAndServe(s.addr, s.mux)
}

func (s *Server) cors(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.handleListModels(w, r)
	case http.MethodPost:
		s.handleLoadModel(w, r)
	case http.MethodDelete:
		s.handleUnloadModel(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := s.manager.ListModels()
	resp := ModelListResponse{
		Object: "list",
		Data:   models,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	var req LoadModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}
	if req.Path == "" {
		writeError(w, http.StatusBadRequest, "path is required", "invalid_request_error")
		return
	}
	if req.ID == "" {
		parts := strings.Split(strings.ReplaceAll(req.Path, "\\", "/"), "/")
		name := parts[len(parts)-1]
		name = strings.TrimSuffix(name, ".gguf")
		req.ID = name
	}
	if req.Context <= 0 {
		req.Context = 2048
	}

	if err := s.manager.LoadModel(req.ID, req.Path, req.GPU, req.Context); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load model: "+err.Error(), "server_error")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "loaded", "id": req.ID})
}

func (s *Server) handleUnloadModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}
	if err := s.manager.UnloadModel(req.ID); err != nil {
		writeError(w, http.StatusNotFound, err.Error(), "not_found_error")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "unloaded", "id": req.ID})
}

func writeError(w http.ResponseWriter, status int, msg, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	resp := ErrorResponse{}
	resp.Error.Message = msg
	resp.Error.Type = errType
	json.NewEncoder(w).Encode(resp)
}
