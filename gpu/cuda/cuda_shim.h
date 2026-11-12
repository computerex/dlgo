// CUDA backend — minimal CUDA Driver API shim.
//
// We talk to the driver API in nvcuda.dll directly (loaded at runtime), rather
// than linking cudart at build time. That keeps the Go/cgo build free of CUDA
// library paths and lets the same binary run on any machine whose driver is
// new enough — the exact same pattern dlgo already uses for vulkan-1.dll.
//
// Everything here is plain C (no .cu), so cgo can compile it directly. Kernels
// live in kernels.cu, are compiled to PTX offline by build_cuda.ps1, and are
// loaded with dlcuda_load_module (cuModuleLoadData on the PTX text).

#ifndef DLGO_CUDA_SHIM_H
#define DLGO_CUDA_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Ownership: returns 0 on success, non-zero CUresult on failure.
// For the *Alloc calls the returned value is the device pointer (0 = failure).

int  dlcuda_available(void);              // 1 if nvcuda.dll loaded and cuInit succeeded
int  dlcuda_init(void);                   // idempotent; 0 = OK
void dlcuda_shutdown(void);               // release context (module/buffers freed by caller)

const char* dlcuda_last_error(void);      // human readable; valid until next call
const char* dlcuda_device_name(void);
int dlcuda_sm_count(void);
unsigned long long dlcuda_total_mem(void);    // device-local, bytes
unsigned long long dlcuda_free_mem(void);     // device-local free, bytes
unsigned long long dlcuda_allocated(void);    // sum of live dlcuda_alloc()

unsigned long long dlcuda_alloc(unsigned long long bytes);       // cuMemAlloc (device-local)
unsigned long long dlcuda_alloc_host(unsigned long long bytes);  // cuMemAllocHost (pinned)
int  dlcuda_free(unsigned long long dptr);
int  dlcuda_free_host(unsigned long long hptr);

int  dlcuda_upload(unsigned long long dst, const void* src, unsigned long long n);
int  dlcuda_upload_at(unsigned long long dst, unsigned long long offset, const void* src, unsigned long long n);
int  dlcuda_upload_at_async(unsigned long long dst, unsigned long long offset, const void* src, unsigned long long n);
int  dlcuda_download(void* dst, unsigned long long src, unsigned long long n);
int  dlcuda_memset8(unsigned long long dst, int value, unsigned long long n);
int  dlcuda_sync(void);

// ---- CUDA graphs ---------------------------------------------------------
// Begin/end capture around a sequence of launches, then replay the recorded
// graph per token. Capture uses a dedicated non-default stream internally.
int  dlcuda_graph_begin(void);
unsigned long long dlcuda_graph_end(void);          // returns an exec handle, 0 on failure
int  dlcuda_graph_launch(unsigned long long exec);
int  dlcuda_graph_destroy(unsigned long long exec);

// Module + function handles are opaque pointers returned as 64-bit values.
unsigned long long dlcuda_load_module(const char* ptx_image);            // cuModuleLoadData
unsigned long long dlcuda_get_func(unsigned long long module, const char* name);

int dlcuda_launch(unsigned long long func,
                  unsigned gx, unsigned gy, unsigned gz,
                  unsigned bx, unsigned by, unsigned bz,
                  unsigned shared_bytes, void** kernel_params);

// dlcuda_launch_blob marshals kernel arguments from a flat buffer.
// `sizes[i]` is the ABI size of argument i (e.g. 8 for a pointer/u64, 4 for int
// or float); `blob` is the concatenation of each argument's bytes. The void*
// array is built on the C stack and the launch is synchronised before returning,
// so the caller's Go buffer stays valid for the whole operation. This avoids
// passing a Go pointer to memory that holds Go pointers (forbidden by cgo).
int dlcuda_launch_blob(unsigned long long func,
                       unsigned gx, unsigned gy, unsigned gz,
                       unsigned bx, unsigned by, unsigned bz,
                       unsigned shared_bytes,
                       int nargs, const unsigned int* sizes, const unsigned char* blob);

// ---------------------------------------------------------------------------
// Typed per-kernel launchers. Each builds the void* params array from C locals
// on its own stack and launches+synchronises inside the call. This is the
// reliable way to pass kernel arguments across the cgo boundary: no Go pointer
// arrays, no blob type-punning, no lifetime hazards.
// ---------------------------------------------------------------------------
int dlcuda_launch_vecadd(unsigned long long fn, unsigned grid, unsigned block,
                         unsigned long long a, unsigned long long b, unsigned long long out, int n);
int dlcuda_launch_kvstore(unsigned long long fn, unsigned grid, unsigned block,
                          unsigned long long kin, unsigned long long vin,
                          unsigned long long kout, unsigned long long vout,
                          int pos, int kv_dim, int ring_cap);
int dlcuda_launch_rmsnorm(unsigned long long fn, unsigned grid, unsigned block,
                          unsigned long long x, unsigned long long w, unsigned long long out,
                          int n, float eps);
int dlcuda_launch_silumul(unsigned long long fn, unsigned grid, unsigned block,
                          unsigned long long gate, unsigned long long up, unsigned long long out, int n);
int dlcuda_launch_softmax(unsigned long long fn, unsigned grid, unsigned block,
                          unsigned long long x, int n);
int dlcuda_launch_probe7(unsigned long long fn, unsigned grid, unsigned block,
                         unsigned long long a, unsigned long long b,
                         unsigned long long c, unsigned long long d,
                         int n1, int n2, int n3);
int dlcuda_launch_halfprobe(unsigned long long fn, unsigned grid, unsigned block,
                            unsigned long long in, unsigned long long out, int n);

#ifdef __cplusplus
}
#endif
#endif
