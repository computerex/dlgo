#include "cuda_shim.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int          CUresult;
typedef int          CUdevice;
typedef uint64_t     CUdeviceptr;
typedef void*        CUcontext;
typedef void*        CUmodule;
typedef void*        CUfunction;
typedef void*        CUstream;

#define CUDA_SUCCESS 0

typedef CUresult (*PFN_cuInit)(unsigned);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*PFN_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef CUresult (*PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*PFN_cuCtxSynchronize)(void);
typedef CUresult (*PFN_cuStreamSynchronize)(CUstream);
typedef CUresult (*PFN_cuMemGetInfo)(size_t*, size_t*);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*PFN_cuMemAllocHost)(void**, size_t);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (*PFN_cuMemFreeHost)(void*);
typedef CUresult (*PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned,
                                       unsigned, CUstream, void**, void**);
typedef CUresult (*PFN_cuGetErrorName)(CUresult, const char**);
// Streams + CUDA graphs (a per-token graph replay is what removes WDDM's
// ~10us/launch dispatch cost; capture cannot use the legacy default stream).
typedef CUresult (*PFN_cuStreamCreate)(CUstream*, unsigned);
typedef CUresult (*PFN_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*PFN_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*PFN_cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
typedef CUresult (*PFN_cuStreamBeginCapture)(CUstream, int);
typedef CUresult (*PFN_cuStreamEndCapture)(CUstream, void**);
typedef CUresult (*PFN_cuGraphInstantiate)(void**, void*, unsigned long long*, void*, void*); // v1, 5 args
typedef CUresult (*PFN_cuGraphInstantiate_v2)(void**, void*, unsigned long long);            // v2, 3 args
typedef CUresult (*PFN_cuGraphInstantiateWithFlags)(void**, void*, unsigned long long);      // 3 args
typedef CUresult (*PFN_cuGraphLaunch)(void*, CUstream);
typedef CUresult (*PFN_cuGraphDestroy)(void*);
typedef CUresult (*PFN_cuGraphExecDestroy)(void*);

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16

static struct {
    int inited;
    int available;
    void* lib;
    CUcontext ctx;
    CUdevice  dev;
    char name[256];
    int sms;
    unsigned long long allocated;
    CUstream stream;      // all work goes here (capture needs a non-default stream)
    int capturing;
    char err[512];

    PFN_cuInit                   Init;
    PFN_cuDeviceGet              DeviceGet;
    PFN_cuDeviceGetName          DeviceGetName;
    PFN_cuDeviceGetAttribute     DeviceGetAttribute;
    PFN_cuDevicePrimaryCtxRetain PrimaryCtxRetain;
    PFN_cuCtxSetCurrent          CtxSetCurrent;
    PFN_cuCtxSynchronize         CtxSynchronize;
    PFN_cuStreamSynchronize      StreamSynchronize;
    PFN_cuMemGetInfo             MemGetInfo;
    PFN_cuMemAlloc               MemAlloc;
    PFN_cuMemAllocHost           MemAllocHost;
    PFN_cuMemFree                MemFree;
    PFN_cuMemFreeHost            MemFreeHost;
    PFN_cuMemcpyHtoD             MemcpyHtoD;
    PFN_cuMemcpyDtoH             MemcpyDtoH;
    PFN_cuMemsetD8               MemsetD8;
    PFN_cuModuleLoadData         ModuleLoadData;
    PFN_cuModuleGetFunction      ModuleGetFunction;
    PFN_cuLaunchKernel           LaunchKernel;
    PFN_cuGetErrorName           GetErrorName;
    PFN_cuStreamCreate           StreamCreate;
    PFN_cuMemcpyHtoDAsync        MemcpyHtoDAsync;
    PFN_cuMemcpyDtoHAsync        MemcpyDtoHAsync;
    PFN_cuMemsetD8Async          MemsetD8Async;
    PFN_cuStreamBeginCapture     StreamBeginCapture;
    PFN_cuStreamEndCapture       StreamEndCapture;
    PFN_cuGraphInstantiate       GraphInstantiate;      // v1 (5 args)
    PFN_cuGraphInstantiate_v2    GraphInstantiateV2;    // v2 (3 args)
    PFN_cuGraphInstantiateWithFlags GraphInstantiateFlags;
    PFN_cuGraphLaunch            GraphLaunch;
    PFN_cuGraphDestroy           GraphDestroy;
    PFN_cuGraphExecDestroy       GraphExecDestroy;
} S;

static void set_err(const char* msg) {
    if (!msg) msg = "";
    strncpy(S.err, msg, sizeof(S.err) - 1);
    S.err[sizeof(S.err) - 1] = 0;
}

static void set_err_result(const char* what, CUresult r) {
    const char* nm = "?";
    if (S.GetErrorName) {
        const char* tmp = 0;
        if (S.GetErrorName(r, &tmp) == CUDA_SUCCESS && tmp) nm = tmp;
    }
    snprintf(S.err, sizeof(S.err), "%s failed: CUresult=%d (%s)", what, (int)r, nm);
}

// The primary context is current only on the OS thread that set it. Go
// goroutines migrate between threads, so re-bind at the start of every entry.
// (Caching this per-thread was tried and was much slower: certain driver calls
// leave no current context, so the re-bind must happen every call.)
static void ensure_ctx(void) {
    if (S.inited && S.ctx && S.CtxSetCurrent) S.CtxSetCurrent(S.ctx);
}

// All work is submitted to one dedicated stream (see dlcuda_init).
static CUstream work_stream(void) { return S.stream ? S.stream : (CUstream)0; }

static int finish_stream(void) {
    if (S.inited && S.StreamSynchronize) {
        CUresult r = S.StreamSynchronize(work_stream());
        if (r != CUDA_SUCCESS) { set_err_result("cuStreamSynchronize", r); return -1; }
    }
    return 0;
}

#ifdef _WIN32
static void* sym(const char* name) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_v2", name); // prefer the modern entry point
    void* p = (void*)GetProcAddress((HMODULE)S.lib, buf);
    if (!p) p = (void*)GetProcAddress((HMODULE)S.lib, name);
    return p;
}
#else
static void* sym(const char* name) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_v2", name);
    void* p = dlsym(S.lib, buf);
    if (!p) p = dlsym(S.lib, name);
    return p;
}
#endif

// Exact-name lookup (no _v2 preference): the cuGraphInstantiate family has
// different arities under the same base name, so the ABI must be chosen
// explicitly rather than by "prefer _v2".
static void* sym_exact(const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)S.lib, name);
#else
    return dlsym(S.lib, name);
#endif
}

#define NEED(field, name) do { \
    *(void**)(&S.field) = sym(name); \
    if (!S.field) { set_err("missing driver symbol: " name); return -1; } \
} while (0)

static int load_symbols(void) {
#ifdef _WIN32
    S.lib = (void*)LoadLibraryA("nvcuda.dll");
    if (!S.lib) { set_err("nvcuda.dll not found (no NVIDIA driver?)"); return -1; }
#else
    S.lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!S.lib) S.lib = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!S.lib) { set_err("libcuda.so not found"); return -1; }
#endif
    NEED(Init, "cuInit");
    NEED(DeviceGet, "cuDeviceGet");
    NEED(DeviceGetName, "cuDeviceGetName");
    NEED(DeviceGetAttribute, "cuDeviceGetAttribute");
    NEED(PrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    NEED(CtxSetCurrent, "cuCtxSetCurrent");
    NEED(CtxSynchronize, "cuCtxSynchronize");
    *(void**)(&S.StreamSynchronize) = sym("cuStreamSynchronize");
    NEED(MemGetInfo, "cuMemGetInfo");
    NEED(MemAlloc, "cuMemAlloc");
    NEED(MemAllocHost, "cuMemAllocHost");
    NEED(MemFree, "cuMemFree");
    NEED(MemFreeHost, "cuMemFreeHost");
    NEED(MemcpyHtoD, "cuMemcpyHtoD");
    NEED(MemcpyDtoH, "cuMemcpyDtoH");
    NEED(MemsetD8, "cuMemsetD8");
    NEED(ModuleLoadData, "cuModuleLoadData");
    NEED(ModuleGetFunction, "cuModuleGetFunction");
    NEED(LaunchKernel, "cuLaunchKernel");
    *(void**)(&S.GetErrorName) = sym("cuGetErrorName");
    // Stream + graph entry points are optional: without them this backend still
    // works (stream 0, no graph replay), it just pays the per-launch cost.
    *(void**)(&S.StreamCreate)       = sym("cuStreamCreate");
    *(void**)(&S.MemcpyHtoDAsync)    = sym("cuMemcpyHtoDAsync");
    *(void**)(&S.MemcpyDtoHAsync)    = sym("cuMemcpyDtoHAsync");
    *(void**)(&S.MemsetD8Async)      = sym("cuMemsetD8Async");
    *(void**)(&S.StreamBeginCapture) = sym("cuStreamBeginCapture");
    *(void**)(&S.StreamEndCapture)   = sym("cuStreamEndCapture");
    *(void**)(&S.GraphInstantiate)      = sym_exact("cuGraphInstantiate");
    *(void**)(&S.GraphInstantiateV2)    = sym_exact("cuGraphInstantiate_v2");
    *(void**)(&S.GraphInstantiateFlags) = sym_exact("cuGraphInstantiateWithFlags");
    *(void**)(&S.GraphLaunch)        = sym("cuGraphLaunch");
    *(void**)(&S.GraphDestroy)       = sym("cuGraphDestroy");
    *(void**)(&S.GraphExecDestroy)   = sym("cuGraphExecDestroy");
    return 0;
}

int dlcuda_init(void) {
    if (S.inited) return 0;
    if (load_symbols() != 0) { S.available = 0; return -1; }

    CUresult r = S.Init(0);
    if (r != CUDA_SUCCESS) { set_err_result("cuInit", r); return -1; }
    r = S.DeviceGet(&S.dev, 0);
    if (r != CUDA_SUCCESS) { set_err_result("cuDeviceGet", r); return -1; }

    char nm[256]; nm[0] = 0;
    if (S.DeviceGetName(nm, sizeof(nm), S.dev) == CUDA_SUCCESS) {
        strncpy(S.name, nm, sizeof(S.name) - 1);
        S.name[sizeof(S.name) - 1] = 0;
    }
    int sms = 0;
    if (S.DeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, S.dev) == CUDA_SUCCESS)
        S.sms = sms;

    r = S.PrimaryCtxRetain(&S.ctx, S.dev);
    if (r != CUDA_SUCCESS) { set_err_result("cuDevicePrimaryCtxRetain", r); return -1; }
    r = S.CtxSetCurrent(S.ctx);
    if (r != CUDA_SUCCESS) { set_err_result("cuCtxSetCurrent", r); return -1; }

    // One dedicated non-blocking stream for everything. The legacy default
    // stream cannot be captured into a CUDA graph, so all work uses this one.
    if (S.StreamCreate) {
        CUstream st = 0;
        if (S.StreamCreate(&st, 1 /*CU_STREAM_NON_BLOCKING*/) == CUDA_SUCCESS) S.stream = st;
    }

    S.inited = 1;
    S.available = 1;
    set_err("");
    return 0;
}

void dlcuda_shutdown(void) { S.inited = 0; }

int dlcuda_available(void) { return S.available ? 1 : 0; }
const char* dlcuda_last_error(void) { return S.err; }
const char* dlcuda_device_name(void) { return S.name; }
int dlcuda_sm_count(void) { return S.sms; }
unsigned long long dlcuda_allocated(void) { return S.allocated; }

unsigned long long dlcuda_total_mem(void) {
    size_t freeB = 0, totalB = 0;
    ensure_ctx();
    if (!S.MemGetInfo) return 0;
    if (S.MemGetInfo(&freeB, &totalB) != CUDA_SUCCESS) return 0;
    return (unsigned long long)totalB;
}

unsigned long long dlcuda_free_mem(void) {
    size_t freeB = 0, totalB = 0;
    ensure_ctx();
    if (!S.MemGetInfo) return 0;
    if (S.MemGetInfo(&freeB, &totalB) != CUDA_SUCCESS) return 0;
    return (unsigned long long)freeB;
}

unsigned long long dlcuda_alloc(unsigned long long bytes) {
    if (!S.inited) return 0;
    ensure_ctx();
    CUdeviceptr p = 0;
    CUresult r = S.MemAlloc(&p, (size_t)bytes);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemAlloc", r); return 0; }
    S.allocated += bytes;
    return (unsigned long long)p;
}

unsigned long long dlcuda_alloc_host(unsigned long long bytes) {
    if (!S.inited) return 0;
    ensure_ctx();
    void* p = 0;
    CUresult r = S.MemAllocHost(&p, (size_t)bytes);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemAllocHost", r); return 0; }
    return (unsigned long long)(uintptr_t)p;
}

int dlcuda_free(unsigned long long dptr) {
    if (!S.inited || !dptr) return -1;
    ensure_ctx();
    CUresult r = S.MemFree((CUdeviceptr)dptr);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemFree", r); return -1; }
    return 0;
}

int dlcuda_free_host(unsigned long long hptr) {
    if (!S.inited || !hptr) return -1;
    ensure_ctx();
    CUresult r = S.MemFreeHost((void*)(uintptr_t)hptr);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemFreeHost", r); return -1; }
    return 0;
}

int dlcuda_upload(unsigned long long dst, const void* src, unsigned long long n) {
    ensure_ctx();
    CUresult r;
    if (S.stream && S.MemcpyHtoDAsync) {
        r = S.MemcpyHtoDAsync((CUdeviceptr)dst, src, (size_t)n, S.stream);
        if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyHtoDAsync", r); return -1; }
        return finish_stream();
    }
    r = S.MemcpyHtoD((CUdeviceptr)dst, src, (size_t)n);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyHtoD", r); return -1; }
    return 0;
}

int dlcuda_upload_at(unsigned long long dst, unsigned long long offset, const void* src, unsigned long long n) {
    ensure_ctx();
    CUresult r;
    if (S.stream && S.MemcpyHtoDAsync) {
        r = S.MemcpyHtoDAsync((CUdeviceptr)(dst + offset), src, (size_t)n, S.stream);
        if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyHtoDAsync", r); return -1; }
        return finish_stream();
    }
    r = S.MemcpyHtoD((CUdeviceptr)(dst + offset), src, (size_t)n);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyHtoD", r); return -1; }
    return 0;
}

// dlcuda_upload_at_async queues the copy on the work stream without waiting, so
// it can be ordered immediately before a graph launch (graph replay scalars).
int dlcuda_upload_at_async(unsigned long long dst, unsigned long long offset, const void* src, unsigned long long n) {
    ensure_ctx();
    CUresult r;
    if (S.stream && S.MemcpyHtoDAsync) {
        r = S.MemcpyHtoDAsync((CUdeviceptr)(dst + offset), src, (size_t)n, S.stream);
        if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyHtoDAsync", r); return -1; }
        return 0;
    }
    r = S.MemcpyHtoD((CUdeviceptr)(dst + offset), src, (size_t)n);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyHtoD", r); return -1; }
    return 0;
}

int dlcuda_download(void* dst, unsigned long long src, unsigned long long n) {
    ensure_ctx();
    CUresult r;
    if (S.stream && S.MemcpyDtoHAsync) {
        r = S.MemcpyDtoHAsync(dst, (CUdeviceptr)src, (size_t)n, S.stream);
        if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyDtoHAsync", r); return -1; }
        return finish_stream();
    }
    r = S.MemcpyDtoH(dst, (CUdeviceptr)src, (size_t)n);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemcpyDtoH", r); return -1; }
    return 0;
}

int dlcuda_memset8(unsigned long long dst, int value, unsigned long long n) {
    ensure_ctx();
    // Queued on the work stream (not synchronised here) so it can be captured
    // as part of a graph.
    CUresult r;
    if (S.stream && S.MemsetD8Async) {
        r = S.MemsetD8Async((CUdeviceptr)dst, (unsigned char)value, (size_t)n, S.stream);
        if (r != CUDA_SUCCESS) { set_err_result("cuMemsetD8Async", r); return -1; }
        return 0;
    }
    r = S.MemsetD8((CUdeviceptr)dst, (unsigned char)value, (size_t)n);
    if (r != CUDA_SUCCESS) { set_err_result("cuMemsetD8", r); return -1; }
    return 0;
}

int dlcuda_sync(void) {
    ensure_ctx();
    // cuCtxSynchronize crashes this WDDM driver; synchronise our own stream,
    // which is where all kernels are queued.
    CUresult r;
    if (S.StreamSynchronize) r = S.StreamSynchronize(work_stream());
    else                     r = S.CtxSynchronize();
    if (r != CUDA_SUCCESS) { set_err_result("cuStreamSynchronize", r); return -1; }
    return 0;
}

// ---- CUDA graph capture / replay -------------------------------------------
// Capture the whole per-token layer loop once, then replay it per token. This
// is what removes the measured ~10us/launch WDDM dispatch cost (decode fires
// ~480 kernels/token).
int dlcuda_graph_begin(void) {
    ensure_ctx();
    if (!S.inited || !S.stream || !S.StreamBeginCapture || !S.StreamEndCapture || !S.GraphInstantiate) {
        set_err("cuda graphs unavailable (missing driver symbols)");
        return -1;
    }
    if (S.capturing) { set_err("graph capture already open"); return -1; }
    CUresult r = S.StreamBeginCapture(S.stream, 0 /*CU_STREAM_CAPTURE_MODE_GLOBAL*/);
    if (r != CUDA_SUCCESS) { set_err_result("cuStreamBeginCapture", r); return -1; }
    S.capturing = 1;
    return 0;
}

// graph_instantiate picks an entry point whose arity matches how we call it. The
// base name cuGraphInstantiate is the 5-arg v1 on CUDA 12, so preferring "_v2"
// blindly would corrupt the stack.
static int graph_instantiate(void** exec, void* graph) {
    CUresult r;
    if (S.GraphInstantiateFlags) {
        r = S.GraphInstantiateFlags(exec, graph, 0);
        if (r != CUDA_SUCCESS) { set_err_result("cuGraphInstantiateWithFlags", r); return -1; }
        return 0;
    }
    if (S.GraphInstantiateV2) {
        r = S.GraphInstantiateV2(exec, graph, 0);
        if (r != CUDA_SUCCESS) { set_err_result("cuGraphInstantiate_v2", r); return -1; }
        return 0;
    }
    if (S.GraphInstantiate) {
        unsigned long long badNode = 0;
        char log[256]; log[0] = 0;
        r = S.GraphInstantiate(exec, graph, &badNode, log, sizeof(log));
        if (r != CUDA_SUCCESS) { set_err_result("cuGraphInstantiate", r); return -1; }
        return 0;
    }
    set_err("no cuGraphInstantiate entry point available");
    return -1;
}

unsigned long long dlcuda_graph_end(void) {
    ensure_ctx();
    if (!S.capturing) { set_err("graph capture not open"); return 0; }
    S.capturing = 0;
    void* graph = 0;
    CUresult r = S.StreamEndCapture(S.stream, &graph);
    if (r != CUDA_SUCCESS) { set_err_result("cuStreamEndCapture", r); return 0; }
    if (!graph) { set_err("cuStreamEndCapture returned no graph"); return 0; }
    void* exec = 0;
    int rc = graph_instantiate(&exec, graph);
    if (S.GraphDestroy) S.GraphDestroy(graph);
    if (rc != 0 || !exec) return 0;
    return (unsigned long long)(uintptr_t)exec;
}

int dlcuda_graph_launch(unsigned long long exec) {
    ensure_ctx();
    if (!exec || !S.GraphLaunch) { set_err("graph launch: no exec handle"); return -1; }
    CUresult r = S.GraphLaunch((void*)(uintptr_t)exec, work_stream());
    if (r != CUDA_SUCCESS) { set_err_result("cuGraphLaunch", r); return -1; }
    return 0;
}

int dlcuda_graph_destroy(unsigned long long exec) {
    ensure_ctx();
    if (!exec || !S.GraphExecDestroy) return 0;
    S.GraphExecDestroy((void*)(uintptr_t)exec);
    return 0;
}

unsigned long long dlcuda_load_module(const char* image) {
    if (!S.inited || !image) return 0;
    ensure_ctx();
    CUmodule m = 0;
    CUresult r = S.ModuleLoadData(&m, (const void*)image);
    if (r != CUDA_SUCCESS) { set_err_result("cuModuleLoadData", r); return 0; }
    return (unsigned long long)(uintptr_t)m;
}

unsigned long long dlcuda_get_func(unsigned long long module, const char* name) {
    if (!S.inited || !module) return 0;
    ensure_ctx();
    CUfunction f = 0;
    CUresult r = S.ModuleGetFunction(&f, (CUmodule)(uintptr_t)module, name);
    if (r != CUDA_SUCCESS) { set_err_result("cuModuleGetFunction", r); return 0; }
    return (unsigned long long)(uintptr_t)f;
}

int dlcuda_launch(unsigned long long func,
                  unsigned gx, unsigned gy, unsigned gz,
                  unsigned bx, unsigned by, unsigned bz,
                  unsigned shared_bytes, void** kernel_params) {
    ensure_ctx();
    CUresult r = S.LaunchKernel((CUfunction)(uintptr_t)func,
                                gx, gy, gz, bx, by, bz,
                                shared_bytes, work_stream(), kernel_params, (void**)0);
    if (r != CUDA_SUCCESS) { set_err_result("cuLaunchKernel", r); return -1; }
    return 0;
}
