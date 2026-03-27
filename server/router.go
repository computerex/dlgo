package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

// Server is the main HTTP server for the dlgo inference engine.
type Server struct {
	mux         *http.ServeMux
	manager     *ModelManager
	chatManager *ChatManager
	addr        string
}

// NewServer creates a new inference server.
func NewServer(addr string, manager *ModelManager, chatManager *ChatManager) *Server {
	s := &Server{
		mux:         http.NewServeMux(),
		manager:     manager,
		chatManager: chatManager,
		addr:        addr,
	}
	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/chat/completions", s.cors(s.handleChatCompletions))
	s.mux.HandleFunc("/v1/models", s.cors(s.handleModels))
	s.mux.HandleFunc("/v1/models/available", s.cors(s.handleAvailableModels))
	s.mux.HandleFunc("/v1/chats", s.cors(s.handleChats))
	s.mux.HandleFunc("/v1/chats/", s.cors(s.handleChatDetail))
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

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"name":    "dlgo",
		"version": "1.0",
		"endpoints": []string{
			"GET  /health",
			"GET  /v1/models",
			"POST /v1/models",
			"DELETE /v1/models",
			"POST /v1/chat/completions",
		},
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	resp := map[string]interface{}{"status": "ok"}
	if vram := s.manager.GetVRAMStatus(); vram != nil {
		resp["vram"] = vram
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := s.manager.ListModels()
	available := s.manager.ListAvailableModels()

	// Convert AvailableModel to AvailableModelObject
	availableObjs := make([]AvailableModelObject, len(available))
	for i, m := range available {
		availableObjs[i] = AvailableModelObject{
			ID:   m.ID,
			Path: m.Path,
		}
	}

	resp := ModelListResponse{
		Object:    "list",
		Data:      models,
		Available: availableObjs,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleAvailableModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}
	available := s.manager.ListAvailableModels()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   available,
	})
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
	// Default to 0 = use model's native context length, automatically reduced
	// by the memory budget checker to fit available RAM/VRAM. Avoids the old
	// 2048-token default that causes thinking models to run out of context after
	// just 1-2 turns when users request large max_tokens.

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

func (s *Server) handleChats(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		chats := s.chatManager.ListChats()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data":   chats,
		})
	case http.MethodPost:
		var req struct {
			Title string `json:"title"`
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
			return
		}
		if req.Title == "" {
			req.Title = "New Chat"
		}
		chat := s.chatManager.CreateChat(req.Title, req.Model)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chat)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
	}
}

func (s *Server) handleChatDetail(w http.ResponseWriter, r *http.Request) {
	// Extract chat ID from path
	path := strings.TrimPrefix(r.URL.Path, "/v1/chats/")
	if path == "" || strings.Contains(path, "/") {
		writeError(w, http.StatusBadRequest, "invalid chat ID", "invalid_request_error")
		return
	}

	switch r.Method {
	case http.MethodGet:
		chat := s.chatManager.GetChat(path)
		if chat == nil {
			writeError(w, http.StatusNotFound, "chat not found", "not_found_error")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chat)
	case http.MethodDelete:
		if !s.chatManager.DeleteChat(path) {
			writeError(w, http.StatusNotFound, "chat not found", "not_found_error")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
	}
}

func writeError(w http.ResponseWriter, status int, msg, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	resp := ErrorResponse{}
	resp.Error.Message = msg
	resp.Error.Type = errType
	json.NewEncoder(w).Encode(resp)
}
