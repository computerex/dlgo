#include "vulkan_gpu.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Performance counters */
static double g_perf_gpu_us = 0.0;
static int    g_perf_submits = 0;
static int    g_perf_dispatches = 0;
static int    g_perf_barriers = 0;

void gpu_perf_reset(void) { g_perf_gpu_us = 0; g_perf_submits = 0; g_perf_dispatches = 0; g_perf_barriers = 0; }
double gpu_perf_get_gpu_us(void) { return g_perf_gpu_us; }
int    gpu_perf_get_dispatches(void) { return g_perf_dispatches; }
int    gpu_perf_get_barriers(void) { return g_perf_barriers; }
void gpu_perf_print(void) {
    fprintf(stderr, "[PERF] GPU fence-wait: %.1f ms, submits: %d, dispatches: %d, barriers: %d, avg: %.2f ms/submit\n",
            g_perf_gpu_us / 1000.0, g_perf_submits, g_perf_dispatches, g_perf_barriers,
            g_perf_submits > 0 ? g_perf_gpu_us / 1000.0 / g_perf_submits : 0.0);
}

#ifdef _WIN32
#include <windows.h>
#define LOAD_VULKAN() LoadLibraryA("vulkan-1.dll")
#define GET_PROC(lib, name) (void*)GetProcAddress((HMODULE)(lib), name)
#define CLOSE_LIB(lib) FreeLibrary((HMODULE)(lib))
typedef HMODULE LibHandle;
#elif defined(__APPLE__)
#include <dlfcn.h>
#define LOAD_VULKAN() dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL)
#define GET_PROC(lib, name) dlsym(lib, name)
#define CLOSE_LIB(lib) dlclose(lib)
typedef void* LibHandle;
#else
#include <dlfcn.h>
#define LOAD_VULKAN() dlopen("libvulkan.so.1", RTLD_NOW)
#define GET_PROC(lib, name) dlsym(lib, name)
#define CLOSE_LIB(lib) dlclose(lib)
typedef void* LibHandle;
#endif

// ---------------------------------------------------------------------------
// Vulkan function pointers (loaded dynamically, no link-time dependency)
// ---------------------------------------------------------------------------
static LibHandle vk_lib = NULL;
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_ = NULL;

#define VK_FUNC(name) static PFN_##name name##_ = NULL;
VK_FUNC(vkCreateInstance)
VK_FUNC(vkDestroyInstance)
VK_FUNC(vkEnumerateInstanceExtensionProperties)
VK_FUNC(vkEnumeratePhysicalDevices)
VK_FUNC(vkEnumerateDeviceExtensionProperties)
VK_FUNC(vkGetPhysicalDeviceProperties)
VK_FUNC(vkGetPhysicalDeviceMemoryProperties)
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties)
VK_FUNC(vkGetPhysicalDeviceFeatures2)
VK_FUNC(vkCreateDevice)
VK_FUNC(vkDestroyDevice)
VK_FUNC(vkGetDeviceQueue)
VK_FUNC(vkCreateCommandPool)
VK_FUNC(vkDestroyCommandPool)
VK_FUNC(vkAllocateCommandBuffers)
VK_FUNC(vkFreeCommandBuffers)
VK_FUNC(vkBeginCommandBuffer)
VK_FUNC(vkEndCommandBuffer)
VK_FUNC(vkQueueSubmit)
VK_FUNC(vkQueueWaitIdle)
VK_FUNC(vkCreateFence)
VK_FUNC(vkDestroyFence)
VK_FUNC(vkWaitForFences)
VK_FUNC(vkResetFences)
VK_FUNC(vkResetCommandBuffer)
VK_FUNC(vkAllocateMemory)
VK_FUNC(vkFreeMemory)
VK_FUNC(vkCreateBuffer)
VK_FUNC(vkDestroyBuffer)
VK_FUNC(vkGetBufferMemoryRequirements)
VK_FUNC(vkBindBufferMemory)
VK_FUNC(vkMapMemory)
VK_FUNC(vkUnmapMemory)
VK_FUNC(vkFlushMappedMemoryRanges)
VK_FUNC(vkInvalidateMappedMemoryRanges)
VK_FUNC(vkCreateShaderModule)
VK_FUNC(vkDestroyShaderModule)
VK_FUNC(vkCreateComputePipelines)
VK_FUNC(vkDestroyPipeline)
VK_FUNC(vkCreatePipelineLayout)
VK_FUNC(vkDestroyPipelineLayout)
VK_FUNC(vkCreateDescriptorSetLayout)
VK_FUNC(vkDestroyDescriptorSetLayout)
VK_FUNC(vkCreateDescriptorPool)
VK_FUNC(vkDestroyDescriptorPool)
VK_FUNC(vkAllocateDescriptorSets)
VK_FUNC(vkUpdateDescriptorSets)
VK_FUNC(vkResetDescriptorPool)
VK_FUNC(vkCmdBindPipeline)
VK_FUNC(vkCmdBindDescriptorSets)
VK_FUNC(vkCmdDispatch)
VK_FUNC(vkCmdPushConstants)
VK_FUNC(vkCmdPipelineBarrier)
VK_FUNC(vkCmdCopyBuffer)
VK_FUNC(vkDeviceWaitIdle)
static PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ = NULL;
static PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2_ = NULL;

/* VK_KHR_cooperative_matrix — use types from SDK headers */
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_fn)(
    VkPhysicalDevice physicalDevice,
    uint32_t* pPropertyCount,
    VkCooperativeMatrixPropertiesKHR* pProperties);
static PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_fn
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_ = NULL;
#undef VK_FUNC

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t size;
    int host_visible;
    int pooled;  // 1 = memory owned by pool, don't free individually
} BufferAlloc;

#define MAX_BUFFERS 8192
#define MAX_DESCRIPTORS_PER_POOL 65536
#define STAGING_SIZE (128 * 1024 * 1024) // 128MB staging buffer

static struct {
    int initialized;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;

    VkPhysicalDeviceProperties dev_props;
    VkPhysicalDeviceMemoryProperties mem_props;

    // Staging buffer for uploads/downloads
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;
    void* staging_mapped;
    uint64_t staging_size;
    uint64_t staging_offset;  // ring-buffer offset within a batch (reset at end_batch)

    // Buffer allocator
    BufferAlloc buffers[MAX_BUFFERS];
    int buf_count;

    // Pipelines
    VkPipeline pipelines[PIPE_COUNT];
    VkPipelineLayout pipe_layouts[PIPE_COUNT];
    VkDescriptorSetLayout desc_layouts[PIPE_COUNT];
    VkDescriptorPool desc_pool;
    int pipelines_ready;

    // Subgroup size (for shader workgroup optimization)
    uint32_t subgroup_size;

    // Command buffer batching mode
    int recording; // 1 = batching dispatches, 0 = immediate submit
    int dispatch_count; // number of dispatches in current batch
    int need_barrier; // 1 = insert barrier before next dispatch
    PipelineID last_bound_pipe; // pipeline bind cache (skip redundant binds)

    // dp4a capability (VK_KHR_shader_integer_dot_product)
    int has_dp4a;

    // Cooperative matrix (VK_KHR_cooperative_matrix — tensor cores)
    int has_coopmat;

    // VK_EXT_memory_budget support for querying free VRAM
    int has_memory_budget;

    // GPU timestamp profiling
    VkQueryPool ts_pool;
    int ts_batch_count;         // how many batches profiled so far
    int ts_dispatch_idx;        // current dispatch index within batch

    // Cumulative VRAM allocation tracking (our own counter)
    uint64_t allocated_bytes;

    // Pool allocator: single VkDeviceMemory, bump pointer
    VkDeviceMemory pool_mem;
    uint64_t pool_size;
    uint64_t pool_offset;   // bump pointer (aligned)
    int pool_active;        // 1 = pool mode, gpu_alloc suballocates
    uint32_t pool_mem_type; // memory type used for pool
} g = {0};

// Safety floor for the budget API pre-check (secondary guard).
#define VRAM_SAFETY_FLOOR_BYTES (512ULL * 1024 * 1024)

// VRAM reserve: bytes kept free for OS (DWM), browsers, Vulkan driver overhead
// (pipelines, descriptors, staging buffer), and other GPU-using apps.
// On a 16 GB GPU this leaves ~13 GB for model weights + KV cache.
// Overflow goes to host-visible (system RAM accessible by GPU shaders via PCIe).
#define VRAM_RESERVE_BYTES (3ULL * 1024 * 1024 * 1024)

// ---------------------------------------------------------------------------
// Dynamic Vulkan loading
// ---------------------------------------------------------------------------
static int load_vulkan(void) {
    vk_lib = LOAD_VULKAN();
    if (!vk_lib) return GPU_ERR_NO_VULKAN;

    vkGetInstanceProcAddr_ = (PFN_vkGetInstanceProcAddr)GET_PROC(vk_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr_) { CLOSE_LIB(vk_lib); vk_lib = NULL; return GPU_ERR_NO_VULKAN; }

    vkCreateInstance_ = (PFN_vkCreateInstance)vkGetInstanceProcAddr_(NULL, "vkCreateInstance");
    vkEnumerateInstanceExtensionProperties_ = (PFN_vkEnumerateInstanceExtensionProperties)
        vkGetInstanceProcAddr_(NULL, "vkEnumerateInstanceExtensionProperties");
    vkEnumeratePhysicalDevices_ = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr_(NULL, "vkEnumeratePhysicalDevices");
    return GPU_OK;
}

static void load_instance_funcs(VkInstance inst) {
    #define LOAD(name) name##_ = (PFN_##name)vkGetInstanceProcAddr_(inst, #name);
    LOAD(vkDestroyInstance)
    LOAD(vkEnumerateDeviceExtensionProperties)
    LOAD(vkEnumeratePhysicalDevices)
    LOAD(vkGetPhysicalDeviceProperties)
    LOAD(vkGetPhysicalDeviceMemoryProperties)
    LOAD(vkGetPhysicalDeviceQueueFamilyProperties)
    LOAD(vkGetPhysicalDeviceFeatures2)
    LOAD(vkCreateDevice)
    LOAD(vkDestroyDevice)
    LOAD(vkGetDeviceQueue)
    LOAD(vkCreateCommandPool)
    LOAD(vkDestroyCommandPool)
    LOAD(vkAllocateCommandBuffers)
    LOAD(vkFreeCommandBuffers)
    LOAD(vkBeginCommandBuffer)
    LOAD(vkEndCommandBuffer)
    LOAD(vkQueueSubmit)
    LOAD(vkQueueWaitIdle)
    LOAD(vkCreateFence)
    LOAD(vkDestroyFence)
    LOAD(vkWaitForFences)
    LOAD(vkResetFences)
    LOAD(vkResetCommandBuffer)
    LOAD(vkAllocateMemory)
    LOAD(vkFreeMemory)
    LOAD(vkCreateBuffer)
    LOAD(vkDestroyBuffer)
    LOAD(vkGetBufferMemoryRequirements)
    LOAD(vkBindBufferMemory)
    LOAD(vkMapMemory)
    LOAD(vkUnmapMemory)
    LOAD(vkFlushMappedMemoryRanges)
    LOAD(vkInvalidateMappedMemoryRanges)
    LOAD(vkCreateShaderModule)
    LOAD(vkDestroyShaderModule)
    LOAD(vkCreateComputePipelines)
    LOAD(vkDestroyPipeline)
    LOAD(vkCreatePipelineLayout)
    LOAD(vkDestroyPipelineLayout)
    LOAD(vkCreateDescriptorSetLayout)
    LOAD(vkDestroyDescriptorSetLayout)
    LOAD(vkCreateDescriptorPool)
    LOAD(vkDestroyDescriptorPool)
    LOAD(vkAllocateDescriptorSets)
    LOAD(vkUpdateDescriptorSets)
    LOAD(vkResetDescriptorPool)
    LOAD(vkCmdBindPipeline)
    LOAD(vkCmdBindDescriptorSets)
    LOAD(vkCmdDispatch)
    LOAD(vkCmdPushConstants)
    LOAD(vkCmdPipelineBarrier)
    LOAD(vkCmdCopyBuffer)
    LOAD(vkDeviceWaitIdle)
    #undef LOAD

    vkCmdPushDescriptorSetKHR_ = (PFN_vkCmdPushDescriptorSetKHR)
        vkGetInstanceProcAddr_(g.instance, "vkCmdPushDescriptorSetKHR");
    vkGetPhysicalDeviceMemoryProperties2_ = (PFN_vkGetPhysicalDeviceMemoryProperties2)
        vkGetInstanceProcAddr_(g.instance, "vkGetPhysicalDeviceMemoryProperties2");
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_ =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_fn)
        vkGetInstanceProcAddr_(g.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
}

static int has_instance_extension(const char* name) {
    if (!vkEnumerateInstanceExtensionProperties_) return 0;
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties_(NULL, &count, NULL) != VK_SUCCESS || count == 0) {
        return 0;
    }
    VkExtensionProperties* exts = (VkExtensionProperties*)calloc(count, sizeof(VkExtensionProperties));
    if (!exts) return 0;
    int found = 0;
    if (vkEnumerateInstanceExtensionProperties_(NULL, &count, exts) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            if (strcmp(exts[i].extensionName, name) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(exts);
    return found;
}

static int has_device_extension(VkPhysicalDevice dev, const char* name) {
    if (!vkEnumerateDeviceExtensionProperties_) return 0;
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties_(dev, NULL, &count, NULL) != VK_SUCCESS || count == 0) {
        return 0;
    }
    VkExtensionProperties* exts = (VkExtensionProperties*)calloc(count, sizeof(VkExtensionProperties));
    if (!exts) return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties_(dev, NULL, &count, exts) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            if (strcmp(exts[i].extensionName, name) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(exts);
    return found;
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------
static uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < g.mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1 << i)) && (g.mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Check if the VRAM budget allows a device-local allocation of req_size bytes.
// Returns 1 if there's room, 0 if we should skip to host-visible fallback.
static int vram_budget_allows(uint64_t req_size, uint32_t mem_type_idx) {
    if (!g.has_memory_budget || !vkGetPhysicalDeviceMemoryProperties2_) return 1;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
    };
    VkPhysicalDeviceMemoryProperties2 mp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = &bp
    };
    vkGetPhysicalDeviceMemoryProperties2_(g.physical_device, &mp2);

    uint32_t heap_idx = g.mem_props.memoryTypes[mem_type_idx].heapIndex;
    uint64_t budget = bp.heapBudget[heap_idx];
    uint64_t usage  = bp.heapUsage[heap_idx];
    uint64_t avail  = (budget > usage) ? (budget - usage) : 0;

    return avail >= req_size + VRAM_SAFETY_FLOOR_BYTES;
}

static int create_buffer(VkBuffer* buf, VkDeviceMemory* mem, uint64_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer_(g.device, &ci, NULL, buf) != VK_SUCCESS) return GPU_ERR_OOM;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements_(g.device, *buf, &req);

    uint32_t mt = find_memory_type(req.memoryTypeBits, mem_flags);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer_(g.device, *buf, NULL);
        return GPU_ERR_OOM;
    }

    // WDDM overcommit protection: on Windows, vkAllocateMemory for device-local
    // almost never fails — it silently overcommits and crashes later. Use the
    // budget API as a guard: if VRAM is too low, return OOM so the caller can
    // fall back to host-visible memory.
    if (mem_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        if (!vram_budget_allows(req.size, mt)) {
            vkDestroyBuffer_(g.device, *buf, NULL);
            return GPU_ERR_OOM;
        }
    }

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;

    VkResult alloc_rc = vkAllocateMemory_(g.device, &ai, NULL, mem);
    if (alloc_rc != VK_SUCCESS) {
        vkDestroyBuffer_(g.device, *buf, NULL);
        return GPU_ERR_OOM;
    }

    vkBindBufferMemory_(g.device, *buf, *mem, 0);
    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Command buffer helpers
// ---------------------------------------------------------------------------
static void begin_cmd(void) {
    vkResetCommandBuffer_(g.cmd_buf, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer_(g.cmd_buf, &bi);
}

static void submit_and_wait(void) {
    vkEndCommandBuffer_(g.cmd_buf);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd_buf;

    LARGE_INTEGER t0, t1, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    vkResetFences_(g.device, 1, &g.fence);
    vkQueueSubmit_(g.queue, 1, &si, g.fence);

    /* Timeout: 10 seconds in nanoseconds.  If the GPU hangs, we must exit
       BEFORE Windows TDR (default 2s per dispatch) resets the adapter —
       otherwise every GPU-using app (including VS Code) crashes too.
       10s is generous for legitimate batches; a single hung dispatch will
       trigger TDR at ~2s and return VK_ERROR_DEVICE_LOST well before this. */
    VkResult fence_result = vkWaitForFences_(g.device, 1, &g.fence, VK_TRUE,
                                             (uint64_t)10 * 1000000000ULL);
    if (fence_result == VK_TIMEOUT) {
        fprintf(stderr, "\n[GPU FATAL] Fence timeout after 10s — GPU likely hung. "
                        "Exiting immediately to prevent TDR crash.\n");
        fflush(stderr);
        exit(96);
    }
    if (fence_result == VK_ERROR_DEVICE_LOST) {
        fprintf(stderr, "\n[GPU FATAL] Device lost (TDR fired). Exiting.\n");
        fflush(stderr);
        exit(95);
    }

    QueryPerformanceCounter(&t1);
    g_perf_gpu_us += (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart * 1e6;
    g_perf_submits++;
}

static void buffer_barrier(VkBuffer buf, uint64_t size) {
    VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.buffer = buf;
    b.offset = 0;
    b.size = size;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier_(g.cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &b, 0, NULL);
}

// ---------------------------------------------------------------------------
// Buffer ID <-> Vulkan buffer mapping
// ---------------------------------------------------------------------------
static GpuBuf register_buffer(VkBuffer buf, VkDeviceMemory mem, uint64_t size, int host_vis) {
    if (g.buf_count >= MAX_BUFFERS) return 0;
    int idx = g.buf_count++;
    g.buffers[idx].buffer = buf;
    g.buffers[idx].memory = mem;
    g.buffers[idx].size = size;
    g.buffers[idx].host_visible = host_vis;
    return (GpuBuf)(idx + 1);
}

static BufferAlloc* get_buf(GpuBuf id) {
    if (id == 0 || id > (GpuBuf)g.buf_count) return NULL;
    return &g.buffers[id - 1];
}

// ---------------------------------------------------------------------------
// Public API: init / shutdown
// ---------------------------------------------------------------------------
int gpu_init(void) {
    if (g.initialized) return GPU_OK;

    int rc = load_vulkan();
    if (rc != GPU_OK) return rc;

    // Create instance
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "dlgo";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
#ifdef __APPLE__
    const char* instance_extensions[2];
    uint32_t instance_extension_count = 0;
    if (has_instance_extension("VK_KHR_get_physical_device_properties2")) {
        instance_extensions[instance_extension_count++] = "VK_KHR_get_physical_device_properties2";
    }
    if (has_instance_extension("VK_KHR_portability_enumeration")) {
        instance_extensions[instance_extension_count++] = "VK_KHR_portability_enumeration";
        ici.flags |= 0x00000001; /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */
    }
    ici.enabledExtensionCount = instance_extension_count;
    ici.ppEnabledExtensionNames = instance_extensions;
#endif

    VkResult vkr = vkCreateInstance_(&ici, NULL, &g.instance);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "[dlgo/gpu] vkCreateInstance failed: %d\n", (int)vkr);
        return GPU_ERR_INIT_FAIL;
    }
    load_instance_funcs(g.instance);

    // Pick physical device (prefer discrete GPU)
    uint32_t dev_count = 0;
    vkr = vkEnumeratePhysicalDevices_(g.instance, &dev_count, NULL);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "[dlgo/gpu] vkEnumeratePhysicalDevices(count) failed: %d\n", (int)vkr);
        return GPU_ERR_INIT_FAIL;
    }
    if (dev_count == 0) return GPU_ERR_NO_DEVICE;

    VkPhysicalDevice* devs = (VkPhysicalDevice*)calloc(dev_count, sizeof(VkPhysicalDevice));
    vkr = vkEnumeratePhysicalDevices_(g.instance, &dev_count, devs);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "[dlgo/gpu] vkEnumeratePhysicalDevices(list) failed: %d\n", (int)vkr);
        free(devs);
        return GPU_ERR_INIT_FAIL;
    }

    g.physical_device = devs[0];
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties_(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g.physical_device = devs[i];
            break;
        }
    }
    free(devs);

    vkGetPhysicalDeviceProperties_(g.physical_device, &g.dev_props);
    vkGetPhysicalDeviceMemoryProperties_(g.physical_device, &g.mem_props);

    // Dump memory layout for diagnostics
    fprintf(stderr, "[dlgo/gpu] Memory heaps: %u\n", g.mem_props.memoryHeapCount);
    for (uint32_t i = 0; i < g.mem_props.memoryHeapCount; i++) {
        fprintf(stderr, "[dlgo/gpu]   heap %u: %llu MB, flags=0x%x%s\n", i,
                (unsigned long long)(g.mem_props.memoryHeaps[i].size / (1024*1024)),
                g.mem_props.memoryHeaps[i].flags,
                (g.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " (device-local)" : "");
    }
    fprintf(stderr, "[dlgo/gpu] Memory types: %u\n", g.mem_props.memoryTypeCount);
    for (uint32_t i = 0; i < g.mem_props.memoryTypeCount; i++) {
        fprintf(stderr, "[dlgo/gpu]   type %u: heap=%u, flags=0x%x", i,
                g.mem_props.memoryTypes[i].heapIndex, g.mem_props.memoryTypes[i].propertyFlags);
        VkMemoryPropertyFlags f = g.mem_props.memoryTypes[i].propertyFlags;
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) fprintf(stderr, " DEV");
        if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) fprintf(stderr, " HOST_VIS");
        if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) fprintf(stderr, " HOST_COH");
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) fprintf(stderr, " HOST_CACH");
        fprintf(stderr, "\n");
    }

    // Find compute queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties_(g.physical_device, &qf_count, NULL);
    VkQueueFamilyProperties* qf = (VkQueueFamilyProperties*)calloc(qf_count, sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties_(g.physical_device, &qf_count, qf);

    g.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g.queue_family = i;
            break;
        }
    }
    free(qf);
    if (g.queue_family == UINT32_MAX) return GPU_ERR_NO_DEVICE;

    // Create logical device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo dqci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = g.queue_family;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &priority;

    int coopmat_ext = has_device_extension(g.physical_device, "VK_KHR_cooperative_matrix");

    VkPhysicalDeviceVulkan12Features avail12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features avail11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 availFeatures2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    availFeatures2.pNext = &avail11;
    avail11.pNext = &avail12;
    if (vkGetPhysicalDeviceFeatures2_) {
        vkGetPhysicalDeviceFeatures2_(g.physical_device, &availFeatures2);
    }

    VkPhysicalDeviceVulkan12Features vk12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    vk12.storageBuffer8BitAccess = avail12.storageBuffer8BitAccess;
    vk12.uniformAndStorageBuffer8BitAccess = avail12.uniformAndStorageBuffer8BitAccess;
    vk12.shaderInt8 = avail12.shaderInt8;
    vk12.scalarBlockLayout = avail12.scalarBlockLayout;
    /* Note: vulkanMemoryModel is required by VK_KHR_cooperative_matrix but
       crashes the NVIDIA driver on this system. The cooperative matrix shader
       may still work since NVIDIA implements the memory model internally. */

    VkPhysicalDeviceVulkan11Features vk11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    vk11.storageBuffer16BitAccess = avail11.storageBuffer16BitAccess;
    vk11.pNext = &vk12;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &vk11;

    int dp4a_ext = has_device_extension(g.physical_device, "VK_KHR_shader_integer_dot_product");
    g.has_dp4a = dp4a_ext;
    g.has_coopmat = coopmat_ext;
    fprintf(stderr, "[dlgo/gpu] VK_KHR_cooperative_matrix: %s\n", coopmat_ext ? "YES" : "NO");

    /* Query cooperative matrix types */
    if (coopmat_ext && vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_) {
        uint32_t cm_count = 0;
        VkResult cmr = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_(g.physical_device, &cm_count, NULL);
        if (cmr == VK_SUCCESS && cm_count > 0 && cm_count < 256) {
            VkCooperativeMatrixPropertiesKHR* cm = (VkCooperativeMatrixPropertiesKHR*)
                calloc(cm_count, sizeof(VkCooperativeMatrixPropertiesKHR));
            for (uint32_t i = 0; i < cm_count; i++)
                cm[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            cmr = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR_(g.physical_device, &cm_count, cm);
            if (cmr == VK_SUCCESS) {
                static const char* ct[] = {"f16","f32","f64","s8","s16","s32","s64","u8","u16","u32","u64"};
                for (uint32_t i = 0; i < cm_count; i++) {
                    uint32_t at = cm[i].AType < 11 ? cm[i].AType : 0;
                    uint32_t bt = cm[i].BType < 11 ? cm[i].BType : 0;
                    uint32_t cct = cm[i].CType < 11 ? cm[i].CType : 0;
                    uint32_t rt = cm[i].ResultType < 11 ? cm[i].ResultType : 0;
                    fprintf(stderr, "[dlgo/gpu] coopmat[%u]: M=%u N=%u K=%u A=%s B=%s C=%s R=%s sat=%d scope=%u\n",
                        i, cm[i].MSize, cm[i].NSize, cm[i].KSize,
                        ct[at], ct[bt], ct[cct], ct[rt],
                        cm[i].saturatingAccumulation, (uint32_t)cm[i].scope);
                }
            }
            free(cm);
        } else {
            fprintf(stderr, "[dlgo/gpu] coopmat query: count=%u result=%d\n", cm_count, (int)cmr);
        }
    }

    const char* device_extensions[8];
    uint32_t device_extension_count = 0;
    if (has_device_extension(g.physical_device, "VK_KHR_push_descriptor")) {
        device_extensions[device_extension_count++] = "VK_KHR_push_descriptor";
    }
    if (has_device_extension(g.physical_device, "VK_KHR_portability_subset")) {
        device_extensions[device_extension_count++] = "VK_KHR_portability_subset";
    }
    if (dp4a_ext) {
        device_extensions[device_extension_count++] = "VK_KHR_shader_integer_dot_product";
    }
    g.has_memory_budget = has_device_extension(g.physical_device, "VK_EXT_memory_budget");
    if (g.has_memory_budget) {
        device_extensions[device_extension_count++] = "VK_EXT_memory_budget";
    }
    if (coopmat_ext) {
        device_extensions[device_extension_count++] = "VK_KHR_cooperative_matrix";
    }

    /* Chain cooperative matrix features if available */
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm_features = {0};
    cm_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    if (coopmat_ext) {
        cm_features.cooperativeMatrix = VK_TRUE;
        /* Chain at top of features2 pNext list */
        cm_features.pNext = features2.pNext;
        features2.pNext = &cm_features;
    }

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    dci.enabledExtensionCount = device_extension_count;
    dci.ppEnabledExtensionNames = device_extensions;

    fprintf(stderr, "[dlgo/gpu] about to call vkCreateDevice (coopmat=%d, ext_count=%u)\n",
        coopmat_ext, device_extension_count);

    vkr = vkCreateDevice_(g.physical_device, &dci, NULL, &g.device);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr,
            "[dlgo/gpu] vkCreateDevice failed: %d (push_descriptor=%d portability_subset=%d int8=%d sb8=%d usb8=%d sb16=%d scalar=%d)\n",
            (int)vkr,
            has_device_extension(g.physical_device, "VK_KHR_push_descriptor"),
            has_device_extension(g.physical_device, "VK_KHR_portability_subset"),
            (int)avail12.shaderInt8,
            (int)avail12.storageBuffer8BitAccess,
            (int)avail12.uniformAndStorageBuffer8BitAccess,
            (int)avail11.storageBuffer16BitAccess,
            (int)avail12.scalarBlockLayout);
        return GPU_ERR_INIT_FAIL;
    }

    vkGetDeviceQueue_(g.device, g.queue_family, 0, &g.queue);

    // Command pool + buffer
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = g.queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkr = vkCreateCommandPool_(g.device, &cpci, NULL, &g.cmd_pool);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "[dlgo/gpu] vkCreateCommandPool failed: %d\n", (int)vkr);
        return GPU_ERR_INIT_FAIL;
    }

    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g.cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkr = vkAllocateCommandBuffers_(g.device, &cbai, &g.cmd_buf);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "[dlgo/gpu] vkAllocateCommandBuffers failed: %d\n", (int)vkr);
        return GPU_ERR_INIT_FAIL;
    }

    // Fence
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkr = vkCreateFence_(g.device, &fci, NULL, &g.fence);
    if (vkr != VK_SUCCESS) {
        fprintf(stderr, "[dlgo/gpu] vkCreateFence failed: %d\n", (int)vkr);
        return GPU_ERR_INIT_FAIL;
    }

    // Staging buffer — prefer HOST_CACHED for fast CPU reads (download path).
    // HOST_CACHED + HOST_COHERENT uses system RAM which the CPU can read at
    // memory bandwidth instead of slow uncached PCIe BAR reads.
    g.staging_size = STAGING_SIZE;
    rc = create_buffer(&g.staging_buf, &g.staging_mem, g.staging_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (rc != GPU_OK) {
        rc = create_buffer(&g.staging_buf, &g.staging_mem, g.staging_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (rc != GPU_OK) {
        fprintf(stderr, "[dlgo/gpu] create staging buffer failed: %d\n", rc);
        return rc;
    }
    vkMapMemory_(g.device, g.staging_mem, 0, g.staging_size, 0, &g.staging_mapped);

    // Descriptor pool
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_DESCRIPTORS_PER_POOL * 8},
    };
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = MAX_DESCRIPTORS_PER_POOL;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = pool_sizes;
    dpci.flags = 0;
    vkCreateDescriptorPool_(g.device, &dpci, NULL, &g.desc_pool);

    g.subgroup_size = 32; // NVIDIA default
    g.recording = 0;
    g.initialized = 1;

    fprintf(stderr, "[dlgo/gpu] Initialized Vulkan on %s (%.0f MB VRAM)\n",
        g.dev_props.deviceName,
        (double)gpu_vram_bytes() / (1024.0 * 1024.0));

    return GPU_OK;
}

void gpu_shutdown(void) {
    if (!g.initialized) return;
    vkDeviceWaitIdle_(g.device);

    for (int i = 0; i < g.buf_count; i++) {
        if (g.buffers[i].buffer) vkDestroyBuffer_(g.device, g.buffers[i].buffer, NULL);
        if (g.buffers[i].memory) vkFreeMemory_(g.device, g.buffers[i].memory, NULL);
    }

    for (int i = 0; i < PIPE_COUNT; i++) {
        if (g.pipelines[i]) vkDestroyPipeline_(g.device, g.pipelines[i], NULL);
        if (g.pipe_layouts[i]) vkDestroyPipelineLayout_(g.device, g.pipe_layouts[i], NULL);
        if (g.desc_layouts[i]) vkDestroyDescriptorSetLayout_(g.device, g.desc_layouts[i], NULL);
    }

    if (g.desc_pool) vkDestroyDescriptorPool_(g.device, g.desc_pool, NULL);

    if (g.staging_mapped) vkUnmapMemory_(g.device, g.staging_mem);
    if (g.staging_buf) vkDestroyBuffer_(g.device, g.staging_buf, NULL);
    if (g.staging_mem) vkFreeMemory_(g.device, g.staging_mem, NULL);

    if (g.fence) vkDestroyFence_(g.device, g.fence, NULL);
    if (g.cmd_pool) vkDestroyCommandPool_(g.device, g.cmd_pool, NULL);
    if (g.device) vkDestroyDevice_(g.device, NULL);
    if (g.instance) vkDestroyInstance_(g.instance, NULL);
    if (vk_lib) CLOSE_LIB(vk_lib);

    memset(&g, 0, sizeof(g));
}

const char* gpu_device_name(void) {
    return g.initialized ? g.dev_props.deviceName : "none";
}

uint64_t gpu_vram_bytes(void) {
    if (!g.initialized) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < g.mem_props.memoryHeapCount; i++) {
        if (g.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += g.mem_props.memoryHeaps[i].size;
    }
    return total;
}

uint64_t gpu_vram_free_bytes(void) {
    if (!g.initialized) return 0;
    if (!g.has_memory_budget || !vkGetPhysicalDeviceMemoryProperties2_) {
        // Fallback: assume 90% of total is available (conservative)
        return gpu_vram_bytes() * 9 / 10;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
    };
    VkPhysicalDeviceMemoryProperties2 mem_props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = &budget_props
    };
    vkGetPhysicalDeviceMemoryProperties2_(g.physical_device, &mem_props2);
    uint64_t free_bytes = 0;
    for (uint32_t i = 0; i < mem_props2.memoryProperties.memoryHeapCount; i++) {
        if (mem_props2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            uint64_t budget = budget_props.heapBudget[i];
            uint64_t usage  = budget_props.heapUsage[i];
            if (budget > usage)
                free_bytes += budget - usage;
        }
    }
    return free_bytes;
}

int gpu_is_initialized(void) { return g.initialized; }
int gpu_has_dp4a(void) { return g.has_dp4a; }
uint64_t gpu_allocated_bytes(void) { return g.allocated_bytes; }

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

// Pool allocator: create a single large device memory, suballocate from it.
int gpu_pool_create(uint64_t size_bytes) {
    if (!g.initialized || g.pool_active) return GPU_ERR_INIT_FAIL;

    // Find device-local memory type
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size_bytes;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer tmp_buf;
    if (vkCreateBuffer_(g.device, &ci, NULL, &tmp_buf) != VK_SUCCESS) return GPU_ERR_OOM;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements_(g.device, tmp_buf, &req);
    vkDestroyBuffer_(g.device, tmp_buf, NULL);

    uint32_t mt = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) return GPU_ERR_OOM;

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = size_bytes;
    ai.memoryTypeIndex = mt;

    VkResult rc = vkAllocateMemory_(g.device, &ai, NULL, &g.pool_mem);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "[gpu] Pool alloc failed: VkResult=%d, size=%.1f MB\n",
                (int)rc, (double)size_bytes/(1024*1024));
        fflush(stderr);
        return GPU_ERR_OOM;
    }

    g.pool_size = size_bytes;
    g.pool_offset = 0;
    g.pool_active = 1;
    g.pool_mem_type = mt;
    g.allocated_bytes += size_bytes;
    return GPU_OK;
}

void gpu_pool_destroy(void) {
    if (!g.pool_mem) return;
    // Free all pool-backed buffers
    for (int i = 0; i < g.buf_count; i++) {
        if (g.buffers[i].pooled && g.buffers[i].buffer) {
            vkDestroyBuffer_(g.device, g.buffers[i].buffer, NULL);
            g.buffers[i].buffer = VK_NULL_HANDLE;
            g.buffers[i].memory = VK_NULL_HANDLE;
            g.buffers[i].size = 0;
            g.buffers[i].pooled = 0;
        }
    }
    if (g.pool_mem) {
        vkFreeMemory_(g.device, g.pool_mem, NULL);
        g.pool_mem = VK_NULL_HANDLE;
    }
    if (g.pool_size <= g.allocated_bytes)
        g.allocated_bytes -= g.pool_size;
    g.pool_active = 0;
    g.pool_size = 0;
    g.pool_offset = 0;
}

// Seal pool: stop suballocating but keep memory alive. Further allocations use normal path.
void gpu_pool_seal(void) {
    g.pool_active = 0;
}

GpuBuf gpu_alloc(uint64_t size_bytes, int usage) {
    if (!g.initialized || size_bytes == 0) return 0;

    // Pool mode: suballocate from single memory
    if (g.pool_active) {
        uint64_t aligned_offset = (g.pool_offset + 255) & ~(uint64_t)255;
        if (aligned_offset + size_bytes > g.pool_size) {
            fprintf(stderr, "[gpu] Pool exhausted: need %llu at offset %llu, pool size %llu\n",
                    (unsigned long long)size_bytes, (unsigned long long)aligned_offset,
                    (unsigned long long)g.pool_size);
            fflush(stderr);
            return 0;
        }

        VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = size_bytes;
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buf;
        if (vkCreateBuffer_(g.device, &ci, NULL, &buf) != VK_SUCCESS) return 0;
        if (vkBindBufferMemory_(g.device, buf, g.pool_mem, aligned_offset) != VK_SUCCESS) {
            vkDestroyBuffer_(g.device, buf, NULL);
            return 0;
        }

        g.pool_offset = aligned_offset + size_bytes;

        if (g.buf_count >= MAX_BUFFERS) { vkDestroyBuffer_(g.device, buf, NULL); return 0; }
        int idx = g.buf_count++;
        g.buffers[idx].buffer = buf;
        g.buffers[idx].memory = g.pool_mem;
        g.buffers[idx].size = size_bytes;
        g.buffers[idx].host_visible = 0;
        g.buffers[idx].pooled = 1;
        return (GpuBuf)(idx + 1);
    }

    // Fallback chain with counter-based VRAM ceiling.
    // PRIMARY guard: our own allocated_bytes counter (updated immediately).
    // The Vulkan budget API is stale on NVIDIA Windows and vkAllocateMemory
    // almost never fails under WDDM (it silently overcommits and freezes).
    // So we MUST decide whether to try device-local BEFORE calling Vulkan.
    VkBuffer buf;
    VkDeviceMemory mem;
    VkBufferUsageFlags vk_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    uint64_t total_vram = gpu_vram_bytes();
    uint64_t vram_ceiling = (total_vram > VRAM_RESERVE_BYTES)
        ? (total_vram - VRAM_RESERVE_BYTES) : (total_vram / 2);
    int under_ceiling = (g.allocated_bytes + size_bytes <= vram_ceiling);

    if (under_ceiling) {
        // Try ReBAR (device-local + host-visible) — best of both worlds
        int rc = create_buffer(&buf, &mem, size_bytes, vk_usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (rc == GPU_OK) {
            g.allocated_bytes += size_bytes;
            return register_buffer(buf, mem, size_bytes, 0);
        }

        // Try device-local only (standard VRAM)
        rc = create_buffer(&buf, &mem, size_bytes, vk_usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (rc == GPU_OK) {
            g.allocated_bytes += size_bytes;
            return register_buffer(buf, mem, size_bytes, 0);
        }
    }

    // Over ceiling or device-local failed: host-visible (system RAM, GPU-accessible)
    int rc = create_buffer(&buf, &mem, size_bytes, vk_usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (rc == GPU_OK) {
        g.allocated_bytes += size_bytes;
        return register_buffer(buf, mem, size_bytes, 1);
    }

    fprintf(stderr, "[gpu] gpu_alloc FAILED: %llu bytes, allocated=%llu MB, ceiling=%llu MB, total=%llu MB\n",
            (unsigned long long)size_bytes,
            (unsigned long long)(g.allocated_bytes/(1024*1024)),
            (unsigned long long)(vram_ceiling/(1024*1024)),
            (unsigned long long)(total_vram/(1024*1024)));
    return 0;
}

uint64_t gpu_host_visible_bytes(void) {
    uint64_t total = 0;
    for (int i = 0; i < g.buf_count; i++) {
        if (g.buffers[i].host_visible && g.buffers[i].buffer)
            total += g.buffers[i].size;
    }
    return total;
}

void gpu_free(GpuBuf id) {
    BufferAlloc* ba = get_buf(id);
    if (!ba) return;
    if (ba->pooled) {
        // Pool-backed: only destroy buffer, memory is owned by pool
        if (ba->buffer) vkDestroyBuffer_(g.device, ba->buffer, NULL);
        ba->buffer = VK_NULL_HANDLE;
        ba->memory = VK_NULL_HANDLE;
        ba->size = 0;
        ba->pooled = 0;
        return;
    }
    if (ba->size <= g.allocated_bytes)
        g.allocated_bytes -= ba->size;
    if (ba->buffer) vkDestroyBuffer_(g.device, ba->buffer, NULL);
    if (ba->memory) vkFreeMemory_(g.device, ba->memory, NULL);
    ba->buffer = VK_NULL_HANDLE;
    ba->memory = VK_NULL_HANDLE;
    ba->size = 0;
}

void gpu_reset_buffer_table(void) {
    // Compact: if all buffers have been freed, reset the counter so new
    // allocations reuse slots from the beginning of the table.
    int highest_live = -1;
    for (int i = 0; i < g.buf_count; i++) {
        if (g.buffers[i].buffer != VK_NULL_HANDLE) {
            highest_live = i;
        }
    }
    g.buf_count = highest_live + 1;
}

int gpu_upload(GpuBuf dst, const void* src, uint64_t size_bytes, uint64_t offset) {
    BufferAlloc* ba = get_buf(dst);
    if (!ba || !src) return GPU_ERR_DISPATCH;

    const uint8_t* p = (const uint8_t*)src;
    uint64_t remaining = size_bytes;
    uint64_t dst_off = offset;

    while (remaining > 0) {
        uint64_t avail = g.staging_size - g.staging_offset;
        uint64_t chunk = remaining < avail ? remaining : avail;

        memcpy((uint8_t*)g.staging_mapped + g.staging_offset, p, chunk);

        if (g.recording) {
            VkBufferCopy region = {g.staging_offset, dst_off, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, g.staging_buf, ba->buffer, 1, &region);
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            g.dispatch_count++;
            /* Advance staging ring offset (align to 256B) */
            g.staging_offset = (g.staging_offset + chunk + 255) & ~(uint64_t)255;
            if (g.staging_offset >= g.staging_size) g.staging_offset = 0;
        } else {
            begin_cmd();
            VkBufferCopy region = {g.staging_offset, dst_off, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, g.staging_buf, ba->buffer, 1, &region);
            submit_and_wait();
            g.staging_offset = 0;
        }

        p += chunk;
        dst_off += chunk;
        remaining -= chunk;
    }
    return GPU_OK;
}

int gpu_download(void* dst, GpuBuf src, uint64_t size_bytes, uint64_t offset) {
    BufferAlloc* ba = get_buf(src);
    if (!ba || !dst) return GPU_ERR_DISPATCH;

    uint8_t* p = (uint8_t*)dst;
    uint64_t remaining = size_bytes;
    uint64_t src_off = offset;

    while (remaining > 0) {
        uint64_t chunk = remaining < g.staging_size ? remaining : g.staging_size;

        if (g.recording) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            VkBufferCopy region = {src_off, 0, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, ba->buffer, g.staging_buf, 1, &region);
            g.dispatch_count++;
            // Must end batch to get the data, then read staging
            gpu_end_batch();
            memcpy(p, g.staging_mapped, chunk);
        } else {
            begin_cmd();
            VkBufferCopy region = {src_off, 0, chunk};
            vkCmdCopyBuffer_(g.cmd_buf, ba->buffer, g.staging_buf, 1, &region);
            submit_and_wait();
            memcpy(p, g.staging_mapped, chunk);
        }

        p += chunk;
        src_off += chunk;
        remaining -= chunk;
    }
    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Shader / pipeline creation
// ---------------------------------------------------------------------------
static VkShaderModule create_shader(const uint32_t* code, size_t code_size) {
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code_size;
    ci.pCode = code;

    VkShaderModule mod;
    if (vkCreateShaderModule_(g.device, &ci, NULL, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    return mod;
}

static int create_compute_pipeline(PipelineID id, const uint32_t* spirv, size_t spirv_size,
                                   int num_buffers, int push_const_size) {
    VkShaderModule mod = create_shader(spirv, spirv_size);
    if (mod == VK_NULL_HANDLE) return GPU_ERR_SHADER;

    // Descriptor set layout: N storage buffers
    VkDescriptorSetLayoutBinding* bindings = (VkDescriptorSetLayoutBinding*)calloc(
        num_buffers, sizeof(VkDescriptorSetLayoutBinding));
    for (int i = 0; i < num_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = num_buffers;
    dslci.pBindings = bindings;
    if (vkCmdPushDescriptorSetKHR_) {
        dslci.flags = 0x00000001; /* VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR */
    }
    vkCreateDescriptorSetLayout_(g.device, &dslci, NULL, &g.desc_layouts[id]);
    free(bindings);

    // Pipeline layout with push constants
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_const_size > 0 ? push_const_size : 4};

    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.desc_layouts[id];
    if (push_const_size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    vkCreatePipelineLayout_(g.device, &plci, NULL, &g.pipe_layouts[id]);

    // Compute pipeline
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = g.pipe_layouts[id];

    VkResult r = vkCreateComputePipelines_(g.device, VK_NULL_HANDLE, 1, &cpci, NULL, &g.pipelines[id]);
    vkDestroyShaderModule_(g.device, mod, NULL);

    return r == VK_SUCCESS ? GPU_OK : GPU_ERR_SHADER;
}

// ---------------------------------------------------------------------------
// Dispatch helper: bind pipeline, descriptors, push constants, dispatch
// ---------------------------------------------------------------------------
typedef struct {
    PipelineID pipe;
    GpuBuf bufs[8];
    uint64_t buf_offsets[8];
    int num_bufs;
    void* push_data;
    int push_size;
    uint32_t groups_x, groups_y, groups_z;
} DispatchParams;

static int dispatch_compute(DispatchParams* p) {
    if (!g.pipelines[p->pipe]) return GPU_ERR_SHADER;

    // Prepare descriptor writes
    VkDescriptorBufferInfo buf_infos[8];
    VkWriteDescriptorSet writes[8];
    for (int i = 0; i < p->num_bufs; i++) {
        BufferAlloc* ba = get_buf(p->bufs[i]);
        if (!ba) return GPU_ERR_DISPATCH;
        buf_infos[i].buffer = ba->buffer;
        buf_infos[i].offset = p->buf_offsets[i];
        buf_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = NULL;
        writes[i].dstSet = VK_NULL_HANDLE;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
        writes[i].pImageInfo = NULL;
        writes[i].pTexelBufferView = NULL;
    }

    if (g.recording) {
        if (g.need_barrier) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, NULL, 0, NULL);
            g.need_barrier = 0;
            g_perf_barriers++;
        }
        if (g.last_bound_pipe != p->pipe) {
            vkCmdBindPipeline_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipelines[p->pipe]);
            g.last_bound_pipe = p->pipe;
        }
        if (vkCmdPushDescriptorSetKHR_) {
            vkCmdPushDescriptorSetKHR_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, p->num_bufs, writes);
        } else {
            VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = g.desc_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &g.desc_layouts[p->pipe];
            VkDescriptorSet ds;
            if (vkAllocateDescriptorSets_(g.device, &dsai, &ds) != VK_SUCCESS) return GPU_ERR_DISPATCH;
            for (int i = 0; i < p->num_bufs; i++) writes[i].dstSet = ds;
            vkUpdateDescriptorSets_(g.device, p->num_bufs, writes, 0, NULL);
            vkCmdBindDescriptorSets_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, 1, &ds, 0, NULL);
        }
        if (p->push_data && p->push_size > 0) {
            vkCmdPushConstants_(g.cmd_buf, g.pipe_layouts[p->pipe],
                VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_size, p->push_data);
        }
        vkCmdDispatch_(g.cmd_buf, p->groups_x, p->groups_y, p->groups_z);
        g.dispatch_count++;
        g_perf_dispatches++;
    } else {
        begin_cmd();
        vkCmdBindPipeline_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipelines[p->pipe]);
        if (vkCmdPushDescriptorSetKHR_) {
            vkCmdPushDescriptorSetKHR_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, p->num_bufs, writes);
        } else {
            VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = g.desc_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &g.desc_layouts[p->pipe];
            VkDescriptorSet ds;
            if (vkAllocateDescriptorSets_(g.device, &dsai, &ds) != VK_SUCCESS) return GPU_ERR_DISPATCH;
            for (int i = 0; i < p->num_bufs; i++) writes[i].dstSet = ds;
            vkUpdateDescriptorSets_(g.device, p->num_bufs, writes, 0, NULL);
            vkCmdBindDescriptorSets_(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                g.pipe_layouts[p->pipe], 0, 1, &ds, 0, NULL);
        }
        if (p->push_data && p->push_size > 0) {
            vkCmdPushConstants_(g.cmd_buf, g.pipe_layouts[p->pipe],
                VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push_size, p->push_data);
        }
        vkCmdDispatch_(g.cmd_buf, p->groups_x, p->groups_y, p->groups_z);
        submit_and_wait();
        vkResetDescriptorPool_(g.device, g.desc_pool, 0);
    }

    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Compute shader SPIR-V will be loaded from embedded data
// We use a separate file for the compiled shaders
// ---------------------------------------------------------------------------
#include "shaders_spirv.h"

int gpu_load_pipelines(void) {
    int total = sizeof(shader_registry) / sizeof(shader_registry[0]);
    if (total > PIPE_COUNT) total = PIPE_COUNT;

    for (int i = 0; i < total; i++) {
        const ShaderInfo* si = &shader_registry[i];
        if (!si->code || si->code_size == 0) continue;
        int rc = create_compute_pipeline(i, si->code, si->code_size,
                                         si->num_buffers, si->push_const_size);
        if (rc != GPU_OK) {
            fprintf(stderr, "[dlgo/gpu] Failed to create pipeline %s (id=%d): %d\n",
                    si->name, i, rc);
        }
    }
    g.pipelines_ready = 1;
    fprintf(stderr, "[dlgo/gpu] Loaded %d compute pipelines\n", total);
    return GPU_OK;
}

void gpu_sync(void) {
    if (g.initialized) vkDeviceWaitIdle_(g.device);
}

void gpu_begin_batch(void) {
    if (!g.initialized || g.recording) return;
    begin_cmd();
    g.recording = 1;
    g.dispatch_count = 0;
    g.last_bound_pipe = PIPE_COUNT;
}

void gpu_end_batch(void) {
    if (!g.initialized || !g.recording) return;
    g.recording = 0;
    if (g.dispatch_count > 0) {
        submit_and_wait();
    }
    vkResetDescriptorPool_(g.device, g.desc_pool, 0);
    g.dispatch_count = 0;
    g.need_barrier = 0;
    g.staging_offset = 0;
}

void gpu_barrier(void) {
    if (g.recording && g.dispatch_count > 0) {
        g.need_barrier = 1;
    }
}

// ---------------------------------------------------------------------------
// IQ lookup table buffers (uploaded once, bound automatically for IQ matvec)
// ---------------------------------------------------------------------------
static GpuBuf g_iq_tables[5] = {0, 0, 0, 0, 0};
#define IQ_TABLE_IQ1S    0
#define IQ_TABLE_IQ2XXS  1
#define IQ_TABLE_IQ2S    2
#define IQ_TABLE_IQ3XXS  3
#define IQ_TABLE_IQ3S    4

int gpu_set_iq_tables(GpuBuf iq1s_buf, GpuBuf iq2xxs_buf, GpuBuf iq2s_buf, GpuBuf iq3xxs_buf, GpuBuf iq3s_buf) {
    g_iq_tables[IQ_TABLE_IQ1S]   = iq1s_buf;
    g_iq_tables[IQ_TABLE_IQ2XXS] = iq2xxs_buf;
    g_iq_tables[IQ_TABLE_IQ2S]   = iq2s_buf;
    g_iq_tables[IQ_TABLE_IQ3XXS] = iq3xxs_buf;
    g_iq_tables[IQ_TABLE_IQ3S]   = iq3s_buf;
    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Operations (implemented after shaders are compiled)
// ---------------------------------------------------------------------------
int gpu_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
               int rows, int cols, int qtype) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    GpuBuf table_buf = 0;
    switch (qtype) {
        case QTYPE_F32:     pipe = PIPE_MATVEC_F32; break;
        case QTYPE_F16:     pipe = PIPE_MATVEC_F16; break;
        case QTYPE_Q4_0:    pipe = PIPE_MATVEC_Q4_0; break;
        case QTYPE_Q8_0:    pipe = PIPE_MATVEC_Q8_0; break;
        case QTYPE_Q3_K:    pipe = PIPE_MATVEC_Q3_K; break;
        case QTYPE_Q4_K:    pipe = PIPE_MATVEC_Q4_K; break;
        case QTYPE_Q5_K:    pipe = PIPE_MATVEC_Q5_K; break;
        case QTYPE_Q5_0:    pipe = PIPE_MATVEC_Q5_0; break;
        case QTYPE_Q5_1:    pipe = PIPE_MATVEC_Q5_1; break;
        case QTYPE_Q6_K:    pipe = PIPE_MATVEC_Q6_K; break;
        case QTYPE_Q2_K:    pipe = PIPE_MATVEC_Q2_K; break;
        case QTYPE_TQ1_0:   pipe = PIPE_MATVEC_TQ1_0; break;
        case QTYPE_IQ1_S:   pipe = PIPE_MATVEC_IQ1_S;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; break;
        case QTYPE_IQ1_M:   pipe = PIPE_MATVEC_IQ1_M;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; break;
        case QTYPE_IQ2_XXS: pipe = PIPE_MATVEC_IQ2_XXS; table_buf = g_iq_tables[IQ_TABLE_IQ2XXS]; break;
        case QTYPE_IQ2_S:   pipe = PIPE_MATVEC_IQ2_S;   table_buf = g_iq_tables[IQ_TABLE_IQ2S]; break;
        case QTYPE_IQ3_XXS: pipe = PIPE_MATVEC_IQ3_XXS; break; // grid in shared mem
        case QTYPE_IQ3_S:   pipe = PIPE_MATVEC_IQ3_S;   table_buf = g_iq_tables[IQ_TABLE_IQ3S]; break;
        case QTYPE_IQ4_XS:  pipe = PIPE_MATVEC_IQ4_XS;  break;
        case 20:            pipe = PIPE_MATVEC_IQ4_NL;  break;  // IQ4_NL
        case 39:            pipe = PIPE_MATVEC_MXFP4;   break;  // MXFP4
        case QTYPE_BF16:    pipe = PIPE_MATVEC_BF16;    break;
        default: return GPU_ERR_DISPATCH;
    }

    // Rows per workgroup by quant type.
    // Q4_0, Q8_0: 2 rows/WG (all-threads-per-row, llama.cpp style).
    // Q5_0: 4 rows/WG (subgroup-per-row, 32 threads each).
    // K-quants and IQ: 2 rows/WG (16-thread cooperation per super-block).
    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_Q4_0: case QTYPE_Q8_0:
        case QTYPE_Q4_K: case QTYPE_Q6_K: case QTYPE_Q3_K: case QTYPE_Q5_K:
        case QTYPE_Q2_K: case QTYPE_TQ1_0:
        case QTYPE_IQ1_S: case QTYPE_IQ1_M:
        case QTYPE_IQ2_XXS:
        case QTYPE_IQ4_XS:
        case 20: // IQ4_NL
            rows_per_wg = 2;
            break;
        case 39: // MXFP4: 1 row per WG (128 threads, simple accumulation)
            rows_per_wg = 1;
            break;
        case QTYPE_BF16: // BF16: 8 rows per WG (128 threads, 4 subgroups x 2 rows)
            rows_per_wg = 8;
            break;
    }

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    if (table_buf) {
        dp.bufs[3] = table_buf;
        dp.num_bufs = 4;
    }
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = 1;
    dp.groups_z = 1;

    return dispatch_compute(&dp);
}

// gpu_matvec_offset: matrix-vector multiply starting from row_offset into weights.
// Used for MoE expert projections from packed expert tensors.
int gpu_matvec_offset(GpuBuf out_buf, int out_offset_bytes,
                      GpuBuf weights_buf, int weights_offset_bytes,
                      GpuBuf x_buf, int rows, int cols, int qtype) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    GpuBuf table_buf = 0;
    switch (qtype) {
        case QTYPE_F32:     pipe = PIPE_MATVEC_F32; break;
        case QTYPE_F16:     pipe = PIPE_MATVEC_F16; break;
        case QTYPE_Q4_0:    pipe = PIPE_MATVEC_Q4_0; break;
        case QTYPE_Q8_0:    pipe = PIPE_MATVEC_Q8_0; break;
        case QTYPE_Q3_K:    pipe = PIPE_MATVEC_Q3_K; break;
        case QTYPE_Q4_K:    pipe = PIPE_MATVEC_Q4_K; break;
        case QTYPE_Q5_K:    pipe = PIPE_MATVEC_Q5_K; break;
        case QTYPE_Q5_0:    pipe = PIPE_MATVEC_Q5_0; break;
        case QTYPE_Q5_1:    pipe = PIPE_MATVEC_Q5_1; break;
        case QTYPE_Q6_K:    pipe = PIPE_MATVEC_Q6_K; break;
        case QTYPE_Q2_K:    pipe = PIPE_MATVEC_Q2_K; break;
        case QTYPE_TQ1_0:   pipe = PIPE_MATVEC_TQ1_0; break;
        case QTYPE_IQ1_S:   pipe = PIPE_MATVEC_IQ1_S;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; break;
        case QTYPE_IQ1_M:   pipe = PIPE_MATVEC_IQ1_M;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; break;
        case QTYPE_IQ2_XXS: pipe = PIPE_MATVEC_IQ2_XXS; table_buf = g_iq_tables[IQ_TABLE_IQ2XXS]; break;
        case QTYPE_IQ2_S:   pipe = PIPE_MATVEC_IQ2_S;   table_buf = g_iq_tables[IQ_TABLE_IQ2S]; break;
        case QTYPE_IQ3_XXS: pipe = PIPE_MATVEC_IQ3_XXS; break;
        case QTYPE_IQ3_S:   pipe = PIPE_MATVEC_IQ3_S;   table_buf = g_iq_tables[IQ_TABLE_IQ3S]; break;
        case QTYPE_IQ4_XS:  pipe = PIPE_MATVEC_IQ4_XS;  break;
        case 20:            pipe = PIPE_MATVEC_IQ4_NL;  break;
        case 39:            pipe = PIPE_MATVEC_MXFP4;   break;
        case QTYPE_BF16:    pipe = PIPE_MATVEC_BF16;    break;
        default: return GPU_ERR_DISPATCH;
    }

    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_Q4_K: case QTYPE_Q6_K: case QTYPE_Q3_K: case QTYPE_Q5_K:
        case QTYPE_Q2_K: case QTYPE_TQ1_0:
        case QTYPE_IQ1_S: case QTYPE_IQ1_M:
        case QTYPE_IQ2_XXS:
        case QTYPE_IQ4_XS:
        case 20:
            rows_per_wg = 2;
            break;
        case 39:
            rows_per_wg = 1;
            break;
        case QTYPE_BF16:
            rows_per_wg = 8;
            break;
    }

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.buf_offsets[0] = out_offset_bytes;
    dp.buf_offsets[1] = weights_offset_bytes;
    dp.num_bufs = 3;
    if (table_buf) {
        dp.bufs[3] = table_buf;
        dp.num_bufs = 4;
    }
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = 1;
    dp.groups_z = 1;

    return dispatch_compute(&dp);
}

// gpu_matvec_offset_dp4a: dp4a integer dot product path for MoE expert projections.
// Input is pre-quantized Q8_1 buffer. Weights at byte offset within packed expert tensor.
int gpu_matvec_offset_dp4a(GpuBuf out_buf, int out_offset_bytes,
                           GpuBuf weights_buf, int weights_offset_bytes,
                           GpuBuf q8_1_buf, int rows, int cols, int qtype) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_Q4_0: pipe = PIPE_MATVEC_Q4_0_DP4A; rows_per_wg = 2; break;
        case QTYPE_Q5_0: pipe = PIPE_MATVEC_Q5_0_DP4A; break;
        case QTYPE_Q8_0: pipe = PIPE_MATVEC_Q8_0_DP4A; break;
        case QTYPE_Q4_K: pipe = PIPE_MATVEC_Q4_K_DP4A; break;
        case QTYPE_Q6_K: pipe = PIPE_MATVEC_Q6_K_DP4A; rows_per_wg = 2; break;
        case QTYPE_Q3_K: pipe = PIPE_MATVEC_Q3_K_DP4A; rows_per_wg = 2; break;
        case QTYPE_Q5_K: pipe = PIPE_MATVEC_Q5_K_DP4A; rows_per_wg = 2; break;
        case 39: pipe = PIPE_MATVEC_MXFP4_DP4A; break;
        default:
            return gpu_matvec_offset(out_buf, out_offset_bytes, weights_buf,
                                     weights_offset_bytes, q8_1_buf, rows, cols, qtype);
    }

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = q8_1_buf;
    dp.buf_offsets[0] = out_offset_bytes;
    dp.buf_offsets[1] = weights_offset_bytes;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// gpu_moe_matvec_dp4a: batched dp4a MoE matvec for all active experts.
// Expert indices come from GPU buffer (no CPU download needed).
// Output is interleaved: out[slot * rows + row] for each expert slot.
int gpu_moe_matvec_dp4a(GpuBuf out_buf, GpuBuf weights_buf,
                        GpuBuf q8_1_buf, GpuBuf indices_buf,
                        int rows, int cols, int qtype,
                        int expert_stride, int base_offset,
                        int shared_input, int n_used) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_Q4_0: pipe = PIPE_MATVEC_Q4_0_DP4A_MOE; break;
        case QTYPE_Q5_0: pipe = PIPE_MATVEC_Q5_0_DP4A_MOE; break;
        case QTYPE_Q8_0: pipe = PIPE_MATVEC_Q8_0_DP4A_MOE; break;
        case QTYPE_Q4_K: pipe = PIPE_MATVEC_Q4_K_DP4A_MOE; rows_per_wg = 2; break;
        case QTYPE_Q6_K: pipe = PIPE_MATVEC_Q6_K_DP4A_MOE; rows_per_wg = 2; break;
        case QTYPE_Q3_K: pipe = PIPE_MATVEC_Q3_K_DP4A_MOE; rows_per_wg = 2; break;
        case QTYPE_Q5_K: pipe = PIPE_MATVEC_Q5_K_DP4A_MOE; rows_per_wg = 2; break;
        case 39: pipe = PIPE_MATVEC_MXFP4_DP4A_MOE; rows_per_wg = 4; break;
        case QTYPE_IQ3_XXS: pipe = PIPE_MATVEC_IQ3_XXS_DP4A_MOE; break;
        case QTYPE_IQ4_XS:  pipe = PIPE_MATVEC_IQ4_XS_DP4A_MOE; break;
        case 20:             pipe = PIPE_MATVEC_IQ4_NL_DP4A_MOE; break;
        case QTYPE_IQ3_S:   pipe = PIPE_MATVEC_IQ3_S_DP4A_MOE; break;
        default: return GPU_ERR_DISPATCH;
    }

    struct { int rows; int cols; int expert_stride; int base_offset; int shared_input; } pc = {
        rows, cols, expert_stride, base_offset, shared_input
    };
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = q8_1_buf;
    dp.bufs[3] = indices_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = n_used;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// IQ-quant MoE matvec: uses float input + lookup table (5 bindings).
// Table buffer is auto-resolved from the qtype (same tables as regular matvec).
int gpu_moe_matvec_iq(GpuBuf out_buf, GpuBuf weights_buf,
                      GpuBuf x_buf, GpuBuf table_buf, GpuBuf indices_buf,
                      int rows, int cols, int qtype,
                      int expert_stride, int base_offset,
                      int shared_input, int n_used) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_IQ3_XXS:
            pipe = PIPE_MATVEC_IQ3_XXS_MOE;
            break;
        default: return GPU_ERR_DISPATCH;
    }

    struct { int rows; int cols; int expert_stride; int base_offset; int shared_input; } pc = {
        rows, cols, expert_stride, base_offset, shared_input
    };
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.bufs[3] = indices_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = n_used;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_moe_accumulate(GpuBuf out_buf, GpuBuf exp_outs_buf, GpuBuf weights_buf,
                       GpuBuf bias_buf, GpuBuf indices_buf,
                       int dim, int n_used, int has_bias) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int dim; int n_used; int has_bias; } pc = {dim, n_used, has_bias};
    DispatchParams dp = {0};
    dp.pipe = PIPE_MOE_ACCUMULATE;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = exp_outs_buf;
    dp.bufs[2] = weights_buf;
    dp.bufs[3] = bias_buf ? bias_buf : out_buf;
    dp.bufs[4] = indices_buf;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (dim + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu_oai_bias_moe(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                            GpuBuf gate_bias_buf, GpuBuf up_bias_buf, GpuBuf indices_buf,
                            int total_n, float alpha, float limit, int exp_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int total_n; float alpha; float limit; int exp_dim; } pc = {
        total_n, alpha, limit, exp_dim
    };
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU_OAI_BIAS_MOE;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.bufs[3] = gate_bias_buf;
    dp.bufs[4] = up_bias_buf;
    dp.bufs[5] = indices_buf;
    dp.num_bufs = 6;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total_n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_moe_bias_add(GpuBuf data_buf, GpuBuf bias_buf, GpuBuf indices_buf,
                     int exp_dim, int n_used) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int exp_dim; int n_used; } pc = {exp_dim, n_used};
    int total = n_used * exp_dim;
    DispatchParams dp = {0};
    dp.pipe = PIPE_MOE_BIAS_ADD;
    dp.bufs[0] = data_buf;
    dp.bufs[1] = bias_buf;
    dp.bufs[2] = indices_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

static int dp4a_safe_type(int qtype, int cols);

// ---------------------------------------------------------------------------
// Fused MoE FFN: all MoE steps in a single CGo call
// ---------------------------------------------------------------------------
int gpu_forward_moe_ffn(const GpuMoEConf* mc) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int dim = mc->dim;
    int exp_dim = mc->exp_dim;
    int n_used = mc->n_used;
    int total_exp_dim = n_used * exp_dim;

    // 1. Router matvec
    gpu_barrier();
    int rc = gpu_matvec(mc->moe_logits, mc->router_w, mc->ffn_norm,
                        mc->router_rows, mc->router_cols, mc->router_type);
    if (rc != GPU_OK) return rc;

    // 2. Router bias (if present)
    if (mc->router_bias) {
        gpu_barrier();
        rc = gpu_add(mc->moe_logits, mc->moe_logits, mc->router_bias, mc->n_experts);
        if (rc != GPU_OK) return rc;
    }

    // 3. TopK (+ optional Q8_1 quantize — only if gate/up types benefit from dp4a)
    gpu_barrier();
    rc = gpu_moe_topk(mc->moe_logits, mc->moe_topk_idx, mc->moe_topk_w,
                      mc->n_experts, n_used, mc->gating_func,
                      mc->weights_norm, mc->weights_scale);
    if (rc != GPU_OK) return rc;
    int gate_dp4a = dp4a_safe_type(mc->gate_type, dim);
    int up_dp4a   = dp4a_safe_type(mc->up_type, dim);
    if (gate_dp4a || up_dp4a) {
        rc = gpu_quantize_q8_1(mc->q8_input, mc->ffn_norm, dim);
        if (rc != GPU_OK) return rc;
    }

    // 4. Gate + Up MoE projections — use dp4a or native per tensor type
    gpu_barrier();
    if (gate_dp4a) {
        rc = gpu_moe_matvec_dp4a(mc->gate_scratch, mc->gate_w, mc->q8_input, mc->moe_topk_idx,
                                  exp_dim, dim, mc->gate_type,
                                  mc->gate_stride, mc->gate_base, 1, n_used);
    } else {
        rc = gpu_moe_matvec_native(mc->gate_scratch, mc->gate_w, mc->ffn_norm, mc->moe_topk_idx,
                                    exp_dim, dim, mc->gate_type,
                                    mc->gate_stride, mc->gate_base, 1, n_used);
    }
    if (rc != GPU_OK) return rc;
    if (up_dp4a) {
        rc = gpu_moe_matvec_dp4a(mc->up_scratch, mc->up_w, mc->q8_input, mc->moe_topk_idx,
                                  exp_dim, dim, mc->up_type,
                                  mc->up_stride, mc->up_base, 1, n_used);
    } else {
        rc = gpu_moe_matvec_native(mc->up_scratch, mc->up_w, mc->ffn_norm, mc->moe_topk_idx,
                                    exp_dim, dim, mc->up_type,
                                    mc->up_stride, mc->up_base, 1, n_used);
    }
    if (rc != GPU_OK) return rc;

    // 5. Activation (SwiGLU or SwiGLU_OAI, with optional bias)
    GpuBuf hidden_buf;
    gpu_barrier();
    int has_gate_bias = mc->gate_bias != 0;
    int has_up_bias = mc->up_bias != 0;

    if (mc->is_oai && has_gate_bias && has_up_bias) {
        hidden_buf = mc->gate_scratch;
        rc = gpu_swiglu_oai_bias_moe(mc->gate_scratch, mc->gate_scratch, mc->up_scratch,
                                     mc->gate_bias, mc->up_bias, mc->moe_topk_idx,
                                     total_exp_dim, mc->alpha, mc->limit, exp_dim);
        if (rc != GPU_OK) return rc;
    } else if (mc->is_oai) {
        hidden_buf = mc->gate_scratch;
        rc = gpu_swiglu_oai(mc->gate_scratch, mc->gate_scratch, mc->up_scratch,
                            total_exp_dim, mc->alpha, mc->limit);
        if (rc != GPU_OK) return rc;
    } else {
        hidden_buf = mc->gate_scratch; // reuse gate_scratch for hidden output
        if (has_gate_bias) {
            rc = gpu_moe_bias_add(mc->gate_scratch, mc->gate_bias, mc->moe_topk_idx, exp_dim, n_used);
            if (rc != GPU_OK) return rc;
        }
        if (has_up_bias) {
            rc = gpu_moe_bias_add(mc->up_scratch, mc->up_bias, mc->moe_topk_idx, exp_dim, n_used);
            if (rc != GPU_OK) return rc;
        }
        if (has_gate_bias || has_up_bias) {
            gpu_barrier();
        }
        rc = gpu_swiglu(mc->gate_scratch, mc->gate_scratch, mc->up_scratch, total_exp_dim);
        if (rc != GPU_OK) return rc;
    }

    // 6–7. Down projections — dp4a or native depending on type
    int down_dp4a = dp4a_safe_type(mc->down_type, exp_dim);
    gpu_barrier();
    if (down_dp4a) {
        rc = gpu_quantize_q8_1(mc->q8_down_packed, hidden_buf, total_exp_dim);
        if (rc != GPU_OK) return rc;
        gpu_barrier();
        rc = gpu_moe_matvec_dp4a(mc->out_scratch, mc->down_w, mc->q8_down_packed, mc->moe_topk_idx,
                                  dim, exp_dim, mc->down_type,
                                  mc->down_stride, 0, 0, n_used);
    } else {
        rc = gpu_moe_matvec_native(mc->out_scratch, mc->down_w, hidden_buf, mc->moe_topk_idx,
                                    dim, exp_dim, mc->down_type,
                                    mc->down_stride, 0, 0, n_used);
    }
    if (rc != GPU_OK) return rc;

    // 8. Weighted accumulation
    gpu_barrier();
    int has_down_bias = mc->down_bias != 0;
    rc = gpu_moe_accumulate(mc->ffn_out, mc->out_scratch, mc->moe_topk_w,
                            mc->down_bias, mc->moe_topk_idx,
                            dim, n_used, has_down_bias);
    if (rc != GPU_OK) return rc;

    // 9. Shared expert (if present) — use dp4a when possible.
    //    mc->q8_input already has Q8_1(ffn_norm) from step 3 for gate/up input.
    //    mc->q8_down_packed is free (main experts done) for shared down quantization.
    if (mc->has_shared && mc->sh_gate_w) {
        gpu_barrier();
        int sh_gate_dp4a = mc->q8_input && dp4a_safe_type(mc->sh_gate_type, mc->sh_gate_cols);
        int sh_up_dp4a   = mc->q8_input && dp4a_safe_type(mc->sh_up_type, mc->sh_up_cols);
        if ((sh_gate_dp4a || sh_up_dp4a) && !(gate_dp4a || up_dp4a)) {
            rc = gpu_quantize_q8_1(mc->q8_input, mc->ffn_norm, dim);
            if (rc != GPU_OK) return rc;
            gpu_barrier();
        }
        if (sh_gate_dp4a)
            rc = gpu_matvec_dp4a(mc->sh_gate_scratch, mc->sh_gate_w, mc->q8_input,
                                 mc->sh_gate_rows, mc->sh_gate_cols, mc->sh_gate_type);
        else
            rc = gpu_matvec(mc->sh_gate_scratch, mc->sh_gate_w, mc->ffn_norm,
                            mc->sh_gate_rows, mc->sh_gate_cols, mc->sh_gate_type);
        if (rc != GPU_OK) return rc;
        if (sh_up_dp4a)
            rc = gpu_matvec_dp4a(mc->sh_up_scratch, mc->sh_up_w, mc->q8_input,
                                 mc->sh_up_rows, mc->sh_up_cols, mc->sh_up_type);
        else
            rc = gpu_matvec(mc->sh_up_scratch, mc->sh_up_w, mc->ffn_norm,
                            mc->sh_up_rows, mc->sh_up_cols, mc->sh_up_type);
        if (rc != GPU_OK) return rc;
        gpu_barrier();
        rc = gpu_swiglu(mc->sh_hidden_scratch, mc->sh_gate_scratch, mc->sh_up_scratch,
                        mc->sh_gate_rows);
        if (rc != GPU_OK) return rc;
        gpu_barrier();
        int sh_down_dp4a = mc->q8_down_packed && dp4a_safe_type(mc->sh_down_type, mc->sh_down_cols);
        if (sh_down_dp4a) {
            rc = gpu_quantize_q8_1(mc->q8_down_packed, mc->sh_hidden_scratch, mc->sh_gate_rows);
            if (rc != GPU_OK) return rc;
            gpu_barrier();
            rc = gpu_matvec_dp4a(mc->sh_out_scratch, mc->sh_down_w, mc->q8_down_packed,
                                 mc->sh_down_rows, mc->sh_down_cols, mc->sh_down_type);
        } else {
            rc = gpu_matvec(mc->sh_out_scratch, mc->sh_down_w, mc->sh_hidden_scratch,
                            mc->sh_down_rows, mc->sh_down_cols, mc->sh_down_type);
        }
        if (rc != GPU_OK) return rc;
        if (mc->sh_router_w) {
            gpu_barrier();
            rc = gpu_shared_expert_gate(mc->sh_out_scratch, mc->sh_router_w,
                                        mc->ffn_norm, dim);
            if (rc != GPU_OK) return rc;
        }
        gpu_barrier();
        rc = gpu_add(mc->ffn_out, mc->ffn_out, mc->sh_out_scratch, dim);
        if (rc != GPU_OK) return rc;
    }

    return GPU_OK;
}

// ---------------------------------------------------------------------------
// Native (float-input) batched MoE matvec for ALL quant types.
// Dispatches for n_used experts via gl_WorkGroupID.y indexing.
// ---------------------------------------------------------------------------
int gpu_moe_matvec_native(GpuBuf out_buf, GpuBuf weights_buf,
                          GpuBuf x_buf, GpuBuf indices_buf,
                          int rows, int cols, int qtype,
                          int expert_stride, int base_offset,
                          int shared_input, int n_used) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    GpuBuf table_buf = 0;
    int rows_per_wg = 4;

    switch (qtype) {
        case QTYPE_F32:     pipe = PIPE_MATVEC_F32_MOE; break;
        case QTYPE_Q4_0:    pipe = PIPE_MATVEC_Q4_0_MOE; break;
        case QTYPE_Q8_0:    pipe = PIPE_MATVEC_Q8_0_MOE; break;
        case QTYPE_Q3_K:    pipe = PIPE_MATVEC_Q3_K_MOE; rows_per_wg = 2; break;
        case QTYPE_Q4_K:    pipe = PIPE_MATVEC_Q4_K_MOE; rows_per_wg = 2; break;
        case QTYPE_Q5_K:    pipe = PIPE_MATVEC_Q5_K_MOE; rows_per_wg = 2; break;
        case QTYPE_Q5_0:    pipe = PIPE_MATVEC_Q5_0_MOE; break;
        case QTYPE_Q6_K:    pipe = PIPE_MATVEC_Q6_K_MOE; rows_per_wg = 2; break;
        case QTYPE_Q2_K:    pipe = PIPE_MATVEC_Q2_K_MOE; rows_per_wg = 2; break;
        case QTYPE_TQ1_0:   pipe = PIPE_MATVEC_TQ1_0_MOE; rows_per_wg = 2; break;
        case QTYPE_IQ1_S:   pipe = PIPE_MATVEC_IQ1_S_MOE;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; rows_per_wg = 2; break;
        case QTYPE_IQ1_M:   pipe = PIPE_MATVEC_IQ1_M_MOE;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; rows_per_wg = 2; break;
        case QTYPE_IQ2_XXS: pipe = PIPE_MATVEC_IQ2_XXS_MOE; table_buf = g_iq_tables[IQ_TABLE_IQ2XXS]; rows_per_wg = 2; break;
        case QTYPE_IQ2_S:   pipe = PIPE_MATVEC_IQ2_S_MOE;   table_buf = g_iq_tables[IQ_TABLE_IQ2S]; break;
        case QTYPE_IQ3_XXS: pipe = PIPE_MATVEC_IQ3_XXS_MOE; break;
        case QTYPE_IQ3_S:   pipe = PIPE_MATVEC_IQ3_S_MOE;   table_buf = g_iq_tables[IQ_TABLE_IQ3S]; break;
        case QTYPE_IQ4_XS:  pipe = PIPE_MATVEC_IQ4_XS_MOE;  rows_per_wg = 2; break;
        case 20:             pipe = PIPE_MATVEC_IQ4_NL_MOE;  rows_per_wg = 2; break;
        case 39:             pipe = PIPE_MATVEC_MXFP4_MOE;   rows_per_wg = 1; break;
        default: return GPU_ERR_DISPATCH;
    }

    struct { int rows; int cols; int expert_stride; int base_offset; int shared_input; } pc = {
        rows, cols, expert_stride, base_offset, shared_input
    };
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    if (table_buf) {
        dp.bufs[3] = table_buf;
        dp.bufs[4] = indices_buf;
        dp.num_bufs = 5;
    } else {
        dp.bufs[3] = indices_buf;
        dp.num_bufs = 4;
    }
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = n_used;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// ---------------------------------------------------------------------------
// Fused native MoE FFN: all MoE steps in a single CGo call using native matvec.
// No Q8_1 quantization needed — uses float input directly.
// ---------------------------------------------------------------------------
int gpu_forward_moe_ffn_native(const GpuMoEConf* mc) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int dim = mc->dim;
    int exp_dim = mc->exp_dim;
    int n_used = mc->n_used;
    int total_exp_dim = n_used * exp_dim;

    // 1. Router matvec
    gpu_barrier();
    int rc = gpu_matvec(mc->moe_logits, mc->router_w, mc->ffn_norm,
                        mc->router_rows, mc->router_cols, mc->router_type);
    if (rc != GPU_OK) return rc;

    // 2. Router bias (if present)
    if (mc->router_bias) {
        gpu_barrier();
        rc = gpu_add(mc->moe_logits, mc->moe_logits, mc->router_bias, mc->n_experts);
        if (rc != GPU_OK) return rc;
    }

    // 3. TopK (handles weights_norm and weights_scale internally)
    gpu_barrier();
    rc = gpu_moe_topk(mc->moe_logits, mc->moe_topk_idx, mc->moe_topk_w,
                      mc->n_experts, n_used, mc->gating_func,
                      mc->weights_norm, mc->weights_scale);
    if (rc != GPU_OK) return rc;

    // 4. Gate + Up MoE projections using native matvec (float input = ffn_norm)
    gpu_barrier();
    rc = gpu_moe_matvec_native(mc->gate_scratch, mc->gate_w, mc->ffn_norm, mc->moe_topk_idx,
                               exp_dim, dim, mc->gate_type,
                               mc->gate_stride, mc->gate_base, 1, n_used);
    if (rc != GPU_OK) return rc;
    rc = gpu_moe_matvec_native(mc->up_scratch, mc->up_w, mc->ffn_norm, mc->moe_topk_idx,
                               exp_dim, dim, mc->up_type,
                               mc->up_stride, mc->up_base, 1, n_used);
    if (rc != GPU_OK) return rc;

    // 5. Activation (SwiGLU or SwiGLU_OAI, with optional bias)
    GpuBuf hidden_buf;
    gpu_barrier();
    int has_gate_bias = mc->gate_bias != 0;
    int has_up_bias = mc->up_bias != 0;

    if (mc->is_oai && has_gate_bias && has_up_bias) {
        hidden_buf = mc->gate_scratch;
        rc = gpu_swiglu_oai_bias_moe(mc->gate_scratch, mc->gate_scratch, mc->up_scratch,
                                     mc->gate_bias, mc->up_bias, mc->moe_topk_idx,
                                     total_exp_dim, mc->alpha, mc->limit, exp_dim);
        if (rc != GPU_OK) return rc;
    } else if (mc->is_oai) {
        hidden_buf = mc->gate_scratch;
        rc = gpu_swiglu_oai(mc->gate_scratch, mc->gate_scratch, mc->up_scratch,
                            total_exp_dim, mc->alpha, mc->limit);
        if (rc != GPU_OK) return rc;
    } else {
        hidden_buf = mc->gate_scratch;
        if (has_gate_bias) {
            rc = gpu_moe_bias_add(mc->gate_scratch, mc->gate_bias, mc->moe_topk_idx, exp_dim, n_used);
            if (rc != GPU_OK) return rc;
        }
        if (has_up_bias) {
            rc = gpu_moe_bias_add(mc->up_scratch, mc->up_bias, mc->moe_topk_idx, exp_dim, n_used);
            if (rc != GPU_OK) return rc;
        }
        if (has_gate_bias || has_up_bias) {
            gpu_barrier();
        }
        rc = gpu_swiglu(mc->gate_scratch, mc->gate_scratch, mc->up_scratch, total_exp_dim);
        if (rc != GPU_OK) return rc;
    }

    // 6. Down projections using native matvec (float input = hidden_buf)
    //    For down proj, each expert's hidden is at slot * exp_dim in hidden_buf,
    //    so shared_input=0 (per-slot input).
    gpu_barrier();
    rc = gpu_moe_matvec_native(mc->out_scratch, mc->down_w, hidden_buf, mc->moe_topk_idx,
                               dim, exp_dim, mc->down_type,
                               mc->down_stride, 0, 0, n_used);
    if (rc != GPU_OK) return rc;

    // 7. Weighted accumulation
    gpu_barrier();
    int has_down_bias = mc->down_bias != 0;
    rc = gpu_moe_accumulate(mc->ffn_out, mc->out_scratch, mc->moe_topk_w,
                            mc->down_bias, mc->moe_topk_idx,
                            dim, n_used, has_down_bias);
    if (rc != GPU_OK) return rc;

    // 8. Shared expert (if present)
    if (mc->has_shared && mc->sh_gate_w) {
        gpu_barrier();
        rc = gpu_matvec(mc->sh_gate_scratch, mc->sh_gate_w, mc->ffn_norm,
                        mc->sh_gate_rows, mc->sh_gate_cols, mc->sh_gate_type);
        if (rc != GPU_OK) return rc;
        rc = gpu_matvec(mc->sh_up_scratch, mc->sh_up_w, mc->ffn_norm,
                        mc->sh_up_rows, mc->sh_up_cols, mc->sh_up_type);
        if (rc != GPU_OK) return rc;
        gpu_barrier();
        rc = gpu_swiglu(mc->sh_hidden_scratch, mc->sh_gate_scratch, mc->sh_up_scratch,
                        mc->sh_gate_rows);
        if (rc != GPU_OK) return rc;
        gpu_barrier();
        rc = gpu_matvec(mc->sh_out_scratch, mc->sh_down_w, mc->sh_hidden_scratch,
                        mc->sh_down_rows, mc->sh_down_cols, mc->sh_down_type);
        if (rc != GPU_OK) return rc;
        if (mc->sh_router_w) {
            gpu_barrier();
            rc = gpu_shared_expert_gate(mc->sh_out_scratch, mc->sh_router_w,
                                        mc->ffn_norm, dim);
            if (rc != GPU_OK) return rc;
        }
        gpu_barrier();
        rc = gpu_add(mc->ffn_out, mc->ffn_out, mc->sh_out_scratch, dim);
        if (rc != GPU_OK) return rc;
    }

    return GPU_OK;
}

int gpu_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float eps; } pc = {n, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_RMSNORM;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = x_buf;
    dp.bufs[2] = weight_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_layernorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, GpuBuf bias_buf, int n, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float eps; } pc = {n, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_LAYERNORM;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = x_buf;
    dp.bufs[2] = weight_buf;
    dp.bufs[3] = bias_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_rmsnorm_heads(GpuBuf data_buf, GpuBuf weight_buf, int num_heads, int head_dim, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int head_dim; float eps; } pc = {head_dim, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_RMSNORM_HEADS;
    dp.bufs[0] = data_buf;
    dp.bufs[1] = weight_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_softmax(GpuBuf buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SOFTMAX;
    dp.bufs[0] = buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_rope(GpuBuf q_buf, GpuBuf k_buf, GpuBuf cos_table, GpuBuf sin_table,
             int num_heads, int num_kv_heads, int head_dim, int rope_dim,
             int pos, int neox) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int num_kv_heads; int head_dim; int rope_dim; int pos; int neox; } pc =
        {num_heads, num_kv_heads, head_dim, rope_dim, pos, neox};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ROPE;
    dp.bufs[0] = q_buf;
    dp.bufs[1] = k_buf;
    dp.bufs[2] = cos_table;
    dp.bufs[3] = sin_table;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (num_heads > num_kv_heads ? num_heads : num_kv_heads);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu_at(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                  int out_off, int gate_off, int up_off, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.buf_offsets[0] = out_off;
    dp.buf_offsets[1] = gate_off;
    dp.buf_offsets[2] = up_off;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu_oai_at(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                      int out_off, int gate_off, int up_off,
                      int n, float alpha, float limit) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float alpha; float limit; } pc = {n, alpha, limit};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU_OAI;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.buf_offsets[0] = out_off;
    dp.buf_offsets[1] = gate_off;
    dp.buf_offsets[2] = up_off;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_quantize_q8_1_at(GpuBuf q8_1_buf, int q8_off, GpuBuf f32_buf, int f32_off, int n_elements) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n_elements; } pc = {n_elements};
    DispatchParams dp = {0};
    dp.pipe = PIPE_QUANTIZE_Q8_1;
    dp.bufs[0] = f32_buf;
    dp.bufs[1] = q8_1_buf;
    dp.buf_offsets[0] = f32_off;
    dp.buf_offsets[1] = q8_off;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    int n_blocks = (n_elements + 31) / 32;
    dp.groups_x = (n_blocks + 3) / 4;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu_oai(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n, float alpha, float limit) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float alpha; float limit; } pc = {n, alpha, limit};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU_OAI;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add_offset(GpuBuf out_buf, GpuBuf bias_buf, int n, int offset) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; int offset; } pc = {n, offset};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ADD_OFFSET;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = bias_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_scale_add(GpuBuf out_buf, GpuBuf in_buf, int n, float scale) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float scale; } pc = {n, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SCALE_ADD;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = in_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_swiglu_oai_bias(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf,
                        GpuBuf gate_bias_buf, GpuBuf up_bias_buf,
                        int n, float alpha, float limit,
                        int gate_bias_offset, int up_bias_offset) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float alpha; float limit; int gate_bias_offset; int up_bias_offset; } pc = {
        n, alpha, limit, gate_bias_offset, up_bias_offset
    };
    DispatchParams dp = {0};
    dp.pipe = PIPE_SWIGLU_OAI_BIAS;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.bufs[3] = gate_bias_buf;
    dp.bufs[4] = up_bias_buf;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_moe_topk(GpuBuf logits_buf, GpuBuf out_indices_buf, GpuBuf out_weights_buf,
                 int n_experts, int k, int gating_func,
                 int weights_norm, float weights_scale) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n_experts; int k; int gating_func; int weights_norm; float weights_scale; } pc = {
        n_experts, k, gating_func, weights_norm, weights_scale
    };
    DispatchParams dp = {0};
    dp.pipe = PIPE_MOE_TOPK;
    dp.bufs[0] = logits_buf;
    dp.bufs[1] = out_indices_buf;
    dp.bufs[2] = out_weights_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_geglu(GpuBuf out_buf, GpuBuf gate_buf, GpuBuf up_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEGLU;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = gate_buf;
    dp.bufs[2] = up_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gelu(GpuBuf buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GELU;
    dp.bufs[0] = buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add(GpuBuf out_buf, GpuBuf a_buf, GpuBuf b_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ADD;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = a_buf;
    dp.bufs[2] = b_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add_bias(GpuBuf buf, GpuBuf bias_buf, int n) {
    return gpu_add(buf, buf, bias_buf, n);
}

int gpu_scale(GpuBuf buf, float s, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float s; } pc = {n, s};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SCALE;
    dp.bufs[0] = buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_add_rmsnorm(GpuBuf norm_out, GpuBuf sum_out,
                    GpuBuf a_buf, GpuBuf b_buf, GpuBuf weight_buf,
                    int n, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; float eps; } pc = {n, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ADD_RMSNORM;
    dp.bufs[0] = norm_out;
    dp.bufs[1] = sum_out;
    dp.bufs[2] = a_buf;
    dp.bufs[3] = b_buf;
    dp.bufs[4] = weight_buf;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_quantize_q8_1(GpuBuf q8_1_buf, GpuBuf f32_buf, int n_elements) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n_elements; } pc = {n_elements};
    DispatchParams dp = {0};
    dp.pipe = PIPE_QUANTIZE_Q8_1;
    dp.bufs[0] = f32_buf;
    dp.bufs[1] = q8_1_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    // 32 threads per WG, 4 blocks per WG
    int n_blocks = (n_elements + 31) / 32;
    dp.groups_x = (n_blocks + 3) / 4;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_matvec_dp4a(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf q8_1_buf,
                    int rows, int cols, int qtype) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    PipelineID pipe;
    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_Q4_0: pipe = PIPE_MATVEC_Q4_0_DP4A; rows_per_wg = 2; break;
        case QTYPE_Q5_0: pipe = PIPE_MATVEC_Q5_0_DP4A; break;
        case QTYPE_Q8_0: pipe = PIPE_MATVEC_Q8_0_DP4A; break;
        case QTYPE_Q4_K: pipe = PIPE_MATVEC_Q4_K_DP4A; break;
        case QTYPE_Q6_K: pipe = PIPE_MATVEC_Q6_K_DP4A; rows_per_wg = 2; break;
        case QTYPE_Q3_K: pipe = PIPE_MATVEC_Q3_K_DP4A; rows_per_wg = 2; break;
        case QTYPE_Q5_K: pipe = PIPE_MATVEC_Q5_K_DP4A; rows_per_wg = 2; break;
        case 39: pipe = PIPE_MATVEC_MXFP4_DP4A; break;
        case QTYPE_IQ3_XXS: pipe = PIPE_MATVEC_IQ3_XXS_DP4A; break;
        case QTYPE_IQ4_XS:  pipe = PIPE_MATVEC_IQ4_XS_DP4A; rows_per_wg = 2; break;
        case 20:             pipe = PIPE_MATVEC_IQ4_NL_DP4A; rows_per_wg = 2; break;
        case QTYPE_IQ3_S:   pipe = PIPE_MATVEC_IQ3_S_DP4A; break;
        default: return gpu_matvec(out_buf, weights_buf, q8_1_buf, rows, cols, qtype);
    }

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = q8_1_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// Fused Q4_K dp4a matvec with on-the-fly Q8_1 quantization in shared memory.
// Caller must ensure cols <= 8192 (shared memory holds max 256 Q8 blocks).
int gpu_matvec_dp4a_fused(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_f32_buf,
                          int rows, int cols) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = PIPE_MATVEC_Q4_K_DP4A_FUSED;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_f32_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + 3) / 4;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_copy_f32(GpuBuf dst, GpuBuf src, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    BufferAlloc* s = get_buf(src);
    BufferAlloc* d = get_buf(dst);
    if (!s || !d) return GPU_ERR_DISPATCH;

    begin_cmd();
    VkBufferCopy region = {0, 0, (uint64_t)n * 4};
    vkCmdCopyBuffer_(g.cmd_buf, s->buffer, d->buffer, 1, &region);
    submit_and_wait();
    return GPU_OK;
}

/* Forward declarations for batched Q4_K paths */
static GpuBuf ensure_q8_scratch(uint64_t needed);
static GpuBuf ensure_dequant_scratch(uint64_t needed);
int gpu_gemm_q4k_dp4a(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf q8_buf,
                      int rows, int cols, int npos);
int gpu_gemm_q4k_tiled(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                        int rows, int cols, int npos);
int gpu_gemm_q4k_fused(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                        int rows, int cols, int npos);
int gpu_gemm_q4k_smem(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf q8_buf,
                       int rows, int cols, int npos);
int gpu_gemm_q4k_coopmat(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                          int rows, int cols, int npos);
int gpu_gemm_q5k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos);
int gpu_gemm_q6k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos);
int gpu_gemm_q6k_f32(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos);
int gpu_gemm_q4k_dequant_coopmat(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                                  int rows, int cols, int npos);

int gpu_batch_matvec(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos, int qtype) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    if (npos <= 0) return GPU_OK;
    if (npos == 1) return gpu_matvec(out_buf, weights_buf, x_buf, rows, cols, qtype);

    /* Use BF16 coopmat GEMM when available and batched */
    if (qtype == QTYPE_BF16 && npos > 1 && g.has_coopmat) {
        return gpu_gemm_bf16(out_buf, weights_buf, x_buf, rows, cols, npos);
    }
    /* Use Q4_K coopmat GEMM when available and batched */
    if (qtype == QTYPE_Q4_K && npos > 1 && g.has_coopmat) {
        return gpu_gemm_q4k(out_buf, weights_buf, x_buf, rows, cols, npos);
    }
    /* Fallback: dequant + scalar tiled GEMM for Q4_K without coopmat */
    if (qtype == QTYPE_Q4_K && npos > 1) {
        return gpu_gemm_q4k_tiled(out_buf, weights_buf, x_buf, rows, cols, npos);
    }
    /* Use Q5_K coopmat GEMM when available and batched */
    if (qtype == QTYPE_Q5_K && npos > 1 && g.has_coopmat) {
        return gpu_gemm_q5k(out_buf, weights_buf, x_buf, rows, cols, npos);
    }
    /* Use Q6_K coopmat GEMM when available, otherwise f32 tiled */
    if (qtype == QTYPE_Q6_K && npos > 1 && g.has_coopmat) {
        return gpu_gemm_q6k(out_buf, weights_buf, x_buf, rows, cols, npos);
    }
    if (qtype == QTYPE_Q6_K && npos > 1) {
        return gpu_gemm_q6k_f32(out_buf, weights_buf, x_buf, rows, cols, npos);
    }

    PipelineID pipe;
    GpuBuf table_buf = 0;
    switch (qtype) {
        case QTYPE_F32:     pipe = PIPE_MATVEC_F32; break;
        case QTYPE_F16:     pipe = PIPE_MATVEC_F16; break;
        case QTYPE_Q4_0:    pipe = PIPE_MATVEC_Q4_0; break;
        case QTYPE_Q8_0:    pipe = PIPE_MATVEC_Q8_0; break;
        case QTYPE_Q3_K:    pipe = PIPE_MATVEC_Q3_K; break;
        case QTYPE_Q4_K:    pipe = PIPE_MATVEC_Q4_K; break;
        case QTYPE_Q5_K:    pipe = PIPE_MATVEC_Q5_K; break;
        case QTYPE_Q5_0:    pipe = PIPE_MATVEC_Q5_0; break;
        case QTYPE_Q5_1:    pipe = PIPE_MATVEC_Q5_1; break;
        case QTYPE_Q6_K:    pipe = PIPE_MATVEC_Q6_K; break;
        case QTYPE_Q2_K:    pipe = PIPE_MATVEC_Q2_K; break;
        case QTYPE_TQ1_0:   pipe = PIPE_MATVEC_TQ1_0; break;
        case QTYPE_IQ1_S:   pipe = PIPE_MATVEC_IQ1_S;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; break;
        case QTYPE_IQ1_M:   pipe = PIPE_MATVEC_IQ1_M;   table_buf = g_iq_tables[IQ_TABLE_IQ1S]; break;
        case QTYPE_IQ2_XXS: pipe = PIPE_MATVEC_IQ2_XXS; table_buf = g_iq_tables[IQ_TABLE_IQ2XXS]; break;
        case QTYPE_IQ2_S:   pipe = PIPE_MATVEC_IQ2_S;   table_buf = g_iq_tables[IQ_TABLE_IQ2S]; break;
        case QTYPE_IQ3_XXS: pipe = PIPE_MATVEC_IQ3_XXS; break;
        case QTYPE_IQ3_S:   pipe = PIPE_MATVEC_IQ3_S;   table_buf = g_iq_tables[IQ_TABLE_IQ3S]; break;
        case QTYPE_IQ4_XS:  pipe = PIPE_MATVEC_IQ4_XS;  break;
        case 20:            pipe = PIPE_MATVEC_IQ4_NL;  break;
        case 39:            pipe = PIPE_MATVEC_MXFP4;   break;
        case QTYPE_BF16:    pipe = PIPE_MATVEC_BF16;    break;
        default: return GPU_ERR_DISPATCH;
    }

    int rows_per_wg = 4;
    switch (qtype) {
        case QTYPE_Q4_0: case QTYPE_Q8_0:
        case QTYPE_Q4_K: case QTYPE_Q6_K: case QTYPE_Q3_K: case QTYPE_Q5_K:
        case QTYPE_Q2_K: case QTYPE_TQ1_0:
        case QTYPE_IQ1_S: case QTYPE_IQ1_M:
        case QTYPE_IQ2_XXS:
        case QTYPE_IQ4_XS:
        case 20:
            rows_per_wg = 2;
            break;
        case 39:
            rows_per_wg = 1;
            break;
        case QTYPE_BF16:
            rows_per_wg = 8;
            break;
    }

    struct { int rows; int cols; } pc = {rows, cols};
    DispatchParams dp = {0};
    dp.pipe = pipe;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    if (table_buf) {
        dp.bufs[3] = table_buf;
        dp.num_bufs = 4;
    }
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + rows_per_wg - 1) / rows_per_wg;
    dp.groups_y = npos;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_copy_region(GpuBuf dst, uint64_t dst_offset, GpuBuf src, uint64_t src_offset, uint64_t size) {
    BufferAlloc* s = get_buf(src);
    BufferAlloc* d = get_buf(dst);
    if (!s || !d) return GPU_ERR_DISPATCH;

    if (g.recording) {
        /* Pre-barrier: ensure preceding compute writes are visible to transfer reads */
        if (g.dispatch_count > 0) {
            VkMemoryBarrier mb_pre = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb_pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb_pre.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier_(g.cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &mb_pre, 0, NULL, 0, NULL);
        }
        VkBufferCopy region = {src_offset, dst_offset, size};
        vkCmdCopyBuffer_(g.cmd_buf, s->buffer, d->buffer, 1, &region);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier_(g.cmd_buf,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, NULL, 0, NULL);
        g.dispatch_count++;
    } else {
        begin_cmd();
        VkBufferCopy region = {src_offset, dst_offset, size};
        vkCmdCopyBuffer_(g.cmd_buf, s->buffer, d->buffer, 1, &region);
        submit_and_wait();
    }
    return GPU_OK;
}

static int gpu_batch_add_bias_expand(GpuBuf dst_buf, GpuBuf bias_buf, GpuBuf scratch_buf,
                                     int elems_per_pos, int npos) {
    uint64_t bytes_per_pos = (uint64_t) elems_per_pos * 4u;
    for (int p = 0; p < npos; ++p) {
        int rc = gpu_copy_region(scratch_buf, (uint64_t) p * bytes_per_pos,
                                 bias_buf, 0, bytes_per_pos);
        if (rc != GPU_OK) return rc;
    }
    return gpu_add(dst_buf, dst_buf, scratch_buf, elems_per_pos * npos);
}

// ---------------------------------------------------------------------------
// Diffusion-specific operations
// ---------------------------------------------------------------------------

int gpu_broadcast_mul(GpuBuf data_buf, GpuBuf scale_buf, int total_n, int dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int total_n; int dim; } pc = {total_n, dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_BROADCAST_MUL;
    dp.bufs[0] = data_buf;
    dp.bufs[1] = scale_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total_n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_tanh_gate_residual(GpuBuf out_buf, GpuBuf residual_buf, GpuBuf data_buf,
                           GpuBuf gate_buf, int total_n, int dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int total_n; int dim; } pc = {total_n, dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_TANH_GATE_RESIDUAL;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = residual_buf;
    dp.bufs[2] = data_buf;
    dp.bufs[3] = gate_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total_n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_rope_3d(GpuBuf vec_buf, GpuBuf pe_buf, int n_pos, int n_heads,
                int head_dim, int pe_offset, int pe_stride) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int half_dim = head_dim / 2;
    int total = n_pos * n_heads * half_dim;
    struct { int n_pos; int n_heads; int head_dim; int pe_offset; int pe_stride; } pc =
        {n_pos, n_heads, head_dim, pe_offset, pe_stride};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ROPE_3D;
    dp.bufs[0] = vec_buf;
    dp.bufs[1] = pe_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_attention_full_f32(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_buf, GpuBuf v_buf,
                           int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                           int seq_len, float scale) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    // Attention Br=8: groups_x = num_heads, groups_y = ceil(seq_len/8)
    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, seq_len, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_FULL_F16;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_buf;
    dp.bufs[3] = v_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = (seq_len + 7) / 8;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_rmsnorm_heads_batch(GpuBuf data_buf, GpuBuf weight_buf,
                            int num_heads, int head_dim, int npos, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    // Reuses existing PIPE_RMSNORM_HEADS shader (already supports pos_id via groups_y)
    struct { int head_dim; float eps; } pc = {head_dim, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_RMSNORM_HEADS;
    dp.bufs[0] = data_buf;
    dp.bufs[1] = weight_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = npos;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// ---------------------------------------------------------------------------
// VAE-specific operations
// ---------------------------------------------------------------------------

int gpu_conv2d_f32(GpuBuf out_buf, GpuBuf in_buf, GpuBuf weight_buf, GpuBuf bias_buf,
                   int inCh, int H, int W, int kH, int kW, int padH, int padW,
                   int stride, int outH, int outW, int outCh) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int inCh; int H; int W; int kH; int kW; int padH; int padW; int stride; int outH; int outW; } pc =
        {inCh, H, W, kH, kW, padH, padW, stride, outH, outW};
    DispatchParams dp = {0};
    dp.pipe = PIPE_CONV2D_F32;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = in_buf;
    dp.bufs[2] = weight_buf;
    dp.bufs[3] = bias_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (outH * outW + 255) / 256;
    dp.groups_y = outCh;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_group_norm(GpuBuf out_buf, GpuBuf in_buf, GpuBuf weight_buf, GpuBuf bias_buf,
                   int C, int spatialSize, int numGroups, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int C; int spatialSize; int numGroups; float eps; } pc = {C, spatialSize, numGroups, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GROUP_NORM;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = in_buf;
    dp.bufs[2] = weight_buf;
    dp.bufs[3] = bias_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = numGroups;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_silu(GpuBuf data_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SILU;
    dp.bufs[0] = data_buf;
    dp.num_bufs = 1;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_upsample_nearest(GpuBuf out_buf, GpuBuf in_buf, int C, int H, int W) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int total = C * H * 2 * W * 2;
    struct { int C; int H; int W; } pc = {C, H, W};
    DispatchParams dp = {0};
    dp.pipe = PIPE_UPSAMPLE_NEAREST;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = in_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_spatial_attention(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_buf, GpuBuf v_buf,
                          int C, int spatial, float scale) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int C; int spatial; float scale; } pc = {C, spatial, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SPATIAL_ATTENTION;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_buf;
    dp.bufs[3] = v_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = spatial;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q4_k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                   int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q4_K;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + 1) / 2;  /* NUM_ROWS = 2 */
    dp.groups_y = (npos + 3) / 4;  /* NT = 4 tokens per workgroup */
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_split_qkv(GpuBuf q_buf, GpuBuf k_buf, GpuBuf v_buf, GpuBuf qkv_buf,
                   int qDim, int kvDim, int seqLen) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int qkvDim = qDim + 2 * kvDim;
    int total = seqLen * qkvDim;
    struct { int qDim; int kvDim; int seqLen; } pc = {qDim, kvDim, seqLen};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SPLIT_QKV;
    dp.bufs[0] = q_buf;
    dp.bufs[1] = k_buf;
    dp.bufs[2] = v_buf;
    dp.bufs[3] = qkv_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (total + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// ---------------------------------------------------------------------------
// Scratch buffer for dequantized FP16 weights (lazy-allocated, reused)
// ---------------------------------------------------------------------------
static GpuBuf g_dequant_scratch = 0;
static uint64_t g_dequant_scratch_size = 0;

static GpuBuf ensure_dequant_scratch(uint64_t needed) {
    if (g_dequant_scratch && g_dequant_scratch_size >= needed)
        return g_dequant_scratch;
    if (g_dequant_scratch) {
        gpu_free(g_dequant_scratch);
        g_dequant_scratch = 0;
        g_dequant_scratch_size = 0;
    }
    g_dequant_scratch = gpu_alloc(needed, 0);
    if (g_dequant_scratch) g_dequant_scratch_size = needed;
    return g_dequant_scratch;
}

// ---------------------------------------------------------------------------
// Scratch buffer for Q8_1 quantized input (lazy-allocated, reused)
// ---------------------------------------------------------------------------
static GpuBuf g_q8_scratch = 0;
static uint64_t g_q8_scratch_size = 0;

static GpuBuf ensure_q8_scratch(uint64_t needed) {
    if (g_q8_scratch && g_q8_scratch_size >= needed)
        return g_q8_scratch;
    if (g_q8_scratch) {
        gpu_free(g_q8_scratch);
        g_q8_scratch = 0;
        g_q8_scratch_size = 0;
    }
    g_q8_scratch = gpu_alloc(needed, 0);
    if (g_q8_scratch) g_q8_scratch_size = needed;
    return g_q8_scratch;
}

int gpu_dequant_q4k_f16(GpuBuf out_f16, GpuBuf weights_q4k, int rows, int cols) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int blocks_per_row = cols / 256;
    int num_blocks = rows * blocks_per_row;

    struct { int num_blocks; int blocks_per_row; } pc = {num_blocks, blocks_per_row};
    DispatchParams dp = {0};
    dp.pipe = PIPE_DEQUANT_Q4K_F16;
    dp.bufs[0] = weights_q4k;
    dp.bufs[1] = out_f16;
    dp.num_bufs = 2;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_blocks;  /* one WG per Q4_K block */
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_f16(GpuBuf out_buf, GpuBuf weight_f16, GpuBuf x_buf,
                 int M, int N, int K, int out_stride) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int M; int N; int K; int out_stride; } pc = {M, N, K, out_stride};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_F16;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weight_f16;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (M + 63) / 64;   /* TM = 64 */
    dp.groups_y = (N + 63) / 64;   /* TN = 64 */
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q4k_dp4a(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf q8_buf,
                      int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q4K_DP4A;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = q8_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + 3) / 4;   /* 4 rows per WG (one per subgroup) */
    dp.groups_y = (npos + 31) / 32; /* NT = 32 tokens per WG */
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

/* Chunked two-pass Q4_K GEMM: dequant rows to FP16 in L2-sized chunks, then tiled GEMM.
 * Q4_K block = 144 bytes, 256 elements per block.
 * Processing N_CHUNK rows at a time keeps FP16 data in L2 cache. */
#define Q4K_BLOCK_BYTES 144
#define TILED_N_CHUNK   2048

int gpu_gemm_q4k_tiled(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                        int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int blocks_per_row = cols / 256;
    int n_chunk = TILED_N_CHUNK;
    if (n_chunk > rows) n_chunk = rows;

    /* FP16 scratch for one chunk: n_chunk * cols * 2 bytes */
    uint64_t scratch_size = (uint64_t)n_chunk * cols * 2;
    GpuBuf scratch = ensure_dequant_scratch(scratch_size);
    if (!scratch) return GPU_ERR_OOM;

    for (int r0 = 0; r0 < rows; r0 += n_chunk) {
        int n_this = rows - r0;
        if (n_this > n_chunk) n_this = n_chunk;

        /* Step 1: Dequant this chunk of Q4_K rows → FP16 scratch */
        int num_blocks = n_this * blocks_per_row;
        struct { int num_blocks; int blocks_per_row; } dq_pc = {num_blocks, blocks_per_row};
        DispatchParams dq = {0};
        dq.pipe = PIPE_DEQUANT_Q4K_F16;
        dq.bufs[0] = weights_buf;
        dq.bufs[1] = scratch;
        dq.num_bufs = 2;
        dq.buf_offsets[0] = (uint64_t)r0 * blocks_per_row * Q4K_BLOCK_BYTES;
        dq.push_data = &dq_pc;
        dq.push_size = sizeof(dq_pc);
        dq.groups_x = num_blocks;
        dq.groups_y = 1;
        dq.groups_z = 1;
        int rc = dispatch_compute(&dq);
        if (rc != GPU_OK) return rc;

        /* Barrier: dequant writes must complete before GEMM reads */
        gpu_barrier();

        /* Step 2: Tiled GEMM: C[m, r0..r0+n_this] = A[m,k] * B_f16[n,k] */
        struct { int M; int N; int K; int out_stride; } gm_pc = {npos, n_this, cols, rows};
        DispatchParams gm = {0};
        gm.pipe = PIPE_GEMM_F16;
        gm.bufs[0] = out_buf;
        gm.bufs[1] = scratch;
        gm.bufs[2] = x_buf;
        gm.num_bufs = 3;
        gm.buf_offsets[0] = (uint64_t)r0 * sizeof(float);  /* output column offset */
        gm.push_data = &gm_pc;
        gm.push_size = sizeof(gm_pc);
        gm.groups_x = (npos + 63) / 64;    /* TM = 64 */
        gm.groups_y = (n_this + 63) / 64;  /* TN = 64 */
        gm.groups_z = 1;
        rc = dispatch_compute(&gm);
        if (rc != GPU_OK) return rc;

        /* Barrier before next chunk's dequant overwrites scratch */
        gpu_barrier();
    }

    return GPU_OK;
}

/* Fused Q4_K tiled GEMM: dequants Q4_K blocks on-the-fly in shared memory.

/* Two-pass Q4_K GEMM with cooperative matrix tensor cores:
 * 1. Dequant all Q4_K rows to FP16 scratch buffer
 * 2. Clean f16×f32 GEMM using cooperative matrix (tensor cores)
 * This avoids the scattered byte reads that bottleneck gpu_gemm_q4k(). */
int gpu_gemm_q4k_dequant_coopmat(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                                  int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int blocks_per_row = cols / 256;
    int num_blocks = rows * blocks_per_row;

    /* FP16 scratch for ALL rows: rows * cols * 2 bytes */
    uint64_t scratch_size = (uint64_t)rows * cols * 2;
    GpuBuf scratch = ensure_dequant_scratch(scratch_size);
    if (!scratch) return GPU_ERR_OOM;

    /* Barrier: ensure previous GEMM finished reading scratch before we overwrite.
     * This is a leading barrier — it only blocks the dequant from starting,
     * not subsequent unrelated operations after the GEMM. */
    gpu_barrier();

    /* Step 1: Dequant all Q4_K rows → FP16 scratch */
    struct { int num_blocks; int blocks_per_row; } dq_pc = {num_blocks, blocks_per_row};
    DispatchParams dq = {0};
    dq.pipe = PIPE_DEQUANT_Q4K_F16;
    dq.bufs[0] = weights_buf;
    dq.bufs[1] = scratch;
    dq.num_bufs = 2;
    dq.push_data = &dq_pc;
    dq.push_size = sizeof(dq_pc);
    dq.groups_x = num_blocks;
    dq.groups_y = 1;
    dq.groups_z = 1;
    int rc = dispatch_compute(&dq);
    if (rc != GPU_OK) return rc;

    /* Barrier: dequant writes must complete before GEMM reads */
    gpu_barrier();

    /* Step 2: Coopmat GEMM: out[npos, rows] = input[npos, cols] × weights_f16[rows, cols]^T */
    struct { int rows; int cols; int npos; } gm_pc = {rows, cols, npos};
    DispatchParams gm = {0};
    gm.pipe = PIPE_GEMM_F16_COOPMAT;
    gm.bufs[0] = out_buf;
    gm.bufs[1] = scratch;
    gm.bufs[2] = x_buf;
    gm.num_bufs = 3;
    gm.push_data = &gm_pc;
    gm.push_size = sizeof(gm_pc);
    gm.groups_x = ((rows + 63) / 64) * ((npos + 63) / 64);
    gm.groups_y = 1;
    gm.groups_z = 1;
    rc = dispatch_compute(&gm);
    if (rc != GPU_OK) return rc;

    return GPU_OK;
}

/* Fused Q4_K tiled GEMM: dequants Q4_K blocks on-the-fly in shared memory.
 * Single-pass, no scratch buffer.  TM=32, TN=32, 128 threads per WG. */
int gpu_gemm_q4k_fused(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                        int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int M; int N; int K; } pc = {npos, rows, cols};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q4K_FUSED;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (npos + 31) / 32;  /* TM = 32 */
    dp.groups_y = (rows + 31) / 32;  /* TN = 32 */
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

/* Shared-memory preloaded Q4_K × Q8_1 dp4a GEMM.
 * Cooperative Q8_1 loading into shared memory eliminates scattered per-token reads. */
int gpu_gemm_q4k_smem(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf q8_buf,
                       int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q4K_SMEM;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = q8_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + 3) / 4;   /* 4 rows per WG */
    dp.groups_y = (npos + 15) / 16; /* NT = 16 tokens per WG */
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q4k_coopmat(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                          int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q4K_COOPMAT;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;           /* FP32 input (no Q8_1 quantization) */
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (rows + 31) / 32; /* WG_M = 32 rows per WG */
    dp.groups_y = (npos + 31) / 32; /* WG_N = 32 tokens per WG */
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_bf16(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                  int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_BF16;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = ((rows + 63) / 64) * ((npos + 63) / 64);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q4k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q4K_COOPMAT;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = ((rows + 63) / 64) * ((npos + 63) / 64);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q5k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q5K_COOPMAT;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = ((rows + 63) / 64) * ((npos + 63) / 64);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q6k(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                 int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q6K_COOPMAT;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = ((rows + 63) / 64) * ((npos + 63) / 64);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_gemm_q6k_f32(GpuBuf out_buf, GpuBuf weights_buf, GpuBuf x_buf,
                     int rows, int cols, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int rows; int cols; int npos; } pc = {rows, cols, npos};
    DispatchParams dp = {0};
    dp.pipe = PIPE_GEMM_Q6K_F32;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = weights_buf;
    dp.bufs[2] = x_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    /* BM=64, BN=32 */
    dp.groups_x = ((rows + 63) / 64) * ((npos + 31) / 32);
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_dequantize(GpuBuf out_f32_buf, GpuBuf quant_buf, int n, int qtype) {
    // TODO: implement dequantize shaders
    return GPU_ERR_SHADER;
}

int gpu_attention(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                  int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                  int seq_len, float scale, int start_pos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int window_len = seq_len - start_pos;
    uint64_t kv_byte_offset = (uint64_t)start_pos * kv_dim * 4;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, window_len, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_cache_buf;
    dp.bufs[3] = v_cache_buf;
    dp.buf_offsets[2] = kv_byte_offset;
    dp.buf_offsets[3] = kv_byte_offset;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_attention_sinks(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                        GpuBuf sinks_buf, int num_heads, int num_kv_heads, int head_dim,
                        int kv_dim, int seq_len, float scale, int start_pos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int window_len = seq_len - start_pos;
    uint64_t kv_byte_offset = (uint64_t)start_pos * kv_dim * 4;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, window_len, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_SINKS;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_cache_buf;
    dp.bufs[3] = v_cache_buf;
    dp.bufs[4] = sinks_buf;
    dp.buf_offsets[2] = kv_byte_offset;
    dp.buf_offsets[3] = kv_byte_offset;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_kv_store(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                 GpuBuf k_buf, GpuBuf v_buf,
                 int pos, int kv_dim) {
    BufferAlloc* kc = get_buf(k_cache_buf);
    BufferAlloc* vc = get_buf(v_cache_buf);
    BufferAlloc* kb = get_buf(k_buf);
    BufferAlloc* vb = get_buf(v_buf);
    if (!kc || !vc || !kb || !vb) return GPU_ERR_DISPATCH;

    if (g.recording) {
        // Combined barrier: ensures all prior compute writes (incl. RoPE) are
        // visible to both the transfer copies AND subsequent compute shaders.
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier_(g.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, NULL, 0, NULL);

        VkBufferCopy kr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, kb->buffer, kc->buffer, 1, &kr);
        VkBufferCopy vr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, vb->buffer, vc->buffer, 1, &vr);

        // Barrier: transfer writes visible to compute reads
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier_(g.cmd_buf,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, NULL, 0, NULL);
        g.dispatch_count++;
    } else {
        begin_cmd();
        VkBufferCopy kr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, kb->buffer, kc->buffer, 1, &kr);
        VkBufferCopy vr = {0, (uint64_t)pos * kv_dim * 4, (uint64_t)kv_dim * 4};
        vkCmdCopyBuffer_(g.cmd_buf, vb->buffer, vc->buffer, 1, &vr);
        submit_and_wait();
    }
    return GPU_OK;
}

// ---------------------------------------------------------------------------
// FP16 KV cache operations
// ---------------------------------------------------------------------------

int gpu_kv_store_f16(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                     GpuBuf k_buf, GpuBuf v_buf,
                     int pos, int kv_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int pos; int kv_dim; } pc = {pos, kv_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_KV_STORE_F16;
    dp.bufs[0] = k_cache_buf;
    dp.bufs[1] = v_cache_buf;
    dp.bufs[2] = k_buf;
    dp.bufs[3] = v_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (kv_dim / 2 + 255) / 256;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_batch_kv_store_f16(GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                           GpuBuf k_buf, GpuBuf v_buf,
                           int start_pos, int kv_dim, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int start_pos; int kv_dim; } pc = {start_pos, kv_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_KV_STORE_BATCH_F16;
    dp.bufs[0] = k_cache_buf;
    dp.bufs[1] = v_cache_buf;
    dp.bufs[2] = k_buf;
    dp.bufs[3] = v_buf;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = (kv_dim / 2 + 255) / 256;
    dp.groups_y = npos;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_attention_f16(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                      int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                      int seq_len, float scale, float softcap, int start_pos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int window_len = seq_len - start_pos;
    // FP16 KV: each position = kv_dim/2 uint32s = kv_dim*2 bytes
    uint64_t kv_byte_offset = (uint64_t)start_pos * kv_dim * 2;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; float softcap; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, window_len, scale, softcap};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_TILED;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_cache_buf;
    dp.bufs[3] = v_cache_buf;
    dp.buf_offsets[2] = kv_byte_offset;
    dp.buf_offsets[3] = kv_byte_offset;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_attention_f16_sinks(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                            GpuBuf sinks_buf, int num_heads, int num_kv_heads, int head_dim,
                            int kv_dim, int seq_len, float scale, float softcap, int start_pos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int window_len = seq_len - start_pos;
    uint64_t kv_byte_offset = (uint64_t)start_pos * kv_dim * 2;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; float softcap; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, window_len, scale, softcap};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_TILED_SINKS;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_cache_buf;
    dp.bufs[3] = v_cache_buf;
    dp.bufs[4] = sinks_buf;
    dp.buf_offsets[2] = kv_byte_offset;
    dp.buf_offsets[3] = kv_byte_offset;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_batch_attention_f16(GpuBuf out, GpuBuf q, GpuBuf k_cache, GpuBuf v_cache,
                            int num_heads, int num_kv_heads, int head_dim,
                            int kv_dim, int start_seq_len, float scale, float softcap, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    struct { int nh; int nkv; int hd; int kvd; int sl; float sc; float scap; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, start_seq_len, scale, softcap};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_TILED;
    dp.bufs[0] = out; dp.bufs[1] = q;
    dp.bufs[2] = k_cache; dp.bufs[3] = v_cache;
    dp.num_bufs = 4;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = npos; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_batch_attention_f16_sinks(GpuBuf out, GpuBuf q, GpuBuf k_cache, GpuBuf v_cache,
                                   GpuBuf sinks, int num_heads, int num_kv_heads, int head_dim,
                                   int kv_dim, int start_seq_len, float scale, float softcap, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    struct { int nh; int nkv; int hd; int kvd; int sl; float sc; float scap; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, start_seq_len, scale, softcap};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_TILED_SINKS;
    dp.bufs[0] = out; dp.bufs[1] = q;
    dp.bufs[2] = k_cache; dp.bufs[3] = v_cache;
    dp.bufs[4] = sinks;
    dp.num_bufs = 5;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = npos; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_attention_tiled_f32(GpuBuf out_buf, GpuBuf q_buf, GpuBuf k_cache_buf, GpuBuf v_cache_buf,
                            int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                            int seq_len, float scale, float softcap, int start_pos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int window_len = seq_len - start_pos;
    // FP32 KV: each position = kv_dim floats = kv_dim*4 bytes
    uint64_t kv_byte_offset = (uint64_t)start_pos * kv_dim * 4;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim; int seq_len; float scale; float softcap; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, window_len, scale, softcap};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_TILED_F32;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_cache_buf;
    dp.bufs[3] = v_cache_buf;
    dp.buf_offsets[2] = kv_byte_offset;
    dp.buf_offsets[3] = kv_byte_offset;
    dp.num_bufs = 4;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_batch_attention_tiled_f32(GpuBuf out, GpuBuf q, GpuBuf k_cache, GpuBuf v_cache,
                                  int num_heads, int num_kv_heads, int head_dim,
                                  int kv_dim, int start_seq_len, float scale, float softcap, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    struct { int nh; int nkv; int hd; int kvd; int sl; float sc; float scap; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, start_seq_len, scale, softcap};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION_TILED_F32;
    dp.bufs[0] = out; dp.bufs[1] = q;
    dp.bufs[2] = k_cache; dp.bufs[3] = v_cache;
    dp.num_bufs = 4;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = npos; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// ---------------------------------------------------------------------------
// PagedAttention operations
// ---------------------------------------------------------------------------

int gpu_paged_kv_store(GpuBuf k_pool, GpuBuf v_pool,
                       GpuBuf k_buf, GpuBuf v_buf,
                       int effective_pos, int kv_dim) {
    // Reuse the same copy logic as gpu_kv_store but with pool buffers
    // and an effective_pos = physical_block * block_size + slot_in_block
    return gpu_kv_store(k_pool, v_pool, k_buf, v_buf, effective_pos, kv_dim);
}

int gpu_paged_attention(GpuBuf out_buf, GpuBuf q_buf,
                        GpuBuf k_pool_buf, GpuBuf v_pool_buf,
                        GpuBuf block_table_buf,
                        int num_heads, int num_kv_heads, int head_dim, int kv_dim,
                        int seq_len, float scale, int block_size) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int num_kv_heads; int head_dim; int kv_dim;
             int seq_len; float scale; int block_size; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, seq_len, scale, block_size};
    DispatchParams dp = {0};
    dp.pipe = PIPE_PAGED_ATTENTION;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = q_buf;
    dp.bufs[2] = k_pool_buf;
    dp.bufs[3] = v_pool_buf;
    dp.bufs[4] = block_table_buf;
    dp.num_bufs = 5;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = num_heads;
    dp.groups_y = 1;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// ---------------------------------------------------------------------------
// SSM (Gated Delta Net) operations
// ---------------------------------------------------------------------------

int gpu_ssm_conv1d_silu(GpuBuf qkv, GpuBuf conv_state, GpuBuf conv_w,
                        int channels, int conv_k) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int channels; int conv_k; } pc = {channels, conv_k};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_CONV1D_SILU;
    dp.bufs[0] = qkv; dp.bufs[1] = conv_state; dp.bufs[2] = conv_w;
    dp.num_bufs = 3;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = (channels + 255) / 256; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_preprocess(GpuBuf alpha, GpuBuf beta, GpuBuf ssma, GpuBuf dt_bias,
                       GpuBuf qkv, int num_heads, int head_k_dim, int key_dim,
                       float rms_eps, int has_dt_bias) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int head_k_dim; int key_dim; float rms_eps; int has_dt_bias; }
        pc = {num_heads, head_k_dim, key_dim, rms_eps, has_dt_bias};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_PREPROCESS;
    dp.bufs[0] = alpha; dp.bufs[1] = beta; dp.bufs[2] = ssma;
    dp.bufs[3] = dt_bias ? dt_bias : alpha;
    dp.bufs[4] = qkv;
    dp.num_bufs = 5;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_delta_rule(GpuBuf state, GpuBuf qkv, GpuBuf alpha, GpuBuf beta,
                       GpuBuf y, int num_heads, int head_k_dim, int head_v_dim, int key_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int head_k_dim; int head_v_dim; int key_dim; }
        pc = {num_heads, head_k_dim, head_v_dim, key_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_DELTA_RULE;
    dp.bufs[0] = state; dp.bufs[1] = qkv; dp.bufs[2] = alpha; dp.bufs[3] = beta; dp.bufs[4] = y;
    dp.num_bufs = 5;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_decay(GpuBuf state, GpuBuf alpha, int num_heads, int head_k_dim, int head_v_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int head_k_dim; int head_v_dim; }
        pc = {num_heads, head_k_dim, head_v_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_DECAY;
    dp.bufs[0] = state; dp.bufs[1] = alpha;
    dp.num_bufs = 2;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_predict(GpuBuf state, GpuBuf qkv, GpuBuf pred,
                    int num_heads, int head_k_dim, int head_v_dim, int key_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int head_k_dim; int head_v_dim; int key_dim; }
        pc = {num_heads, head_k_dim, head_v_dim, key_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_PREDICT;
    dp.bufs[0] = state; dp.bufs[1] = qkv; dp.bufs[2] = pred;
    dp.num_bufs = 3;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_update(GpuBuf state, GpuBuf qkv, GpuBuf beta, GpuBuf pred,
                   int num_heads, int head_k_dim, int head_v_dim, int key_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int head_k_dim; int head_v_dim; int key_dim; }
        pc = {num_heads, head_k_dim, head_v_dim, key_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_UPDATE;
    dp.bufs[0] = state; dp.bufs[1] = qkv; dp.bufs[2] = beta; dp.bufs[3] = pred;
    dp.num_bufs = 4;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_output(GpuBuf state, GpuBuf qkv, GpuBuf y,
                   int num_heads, int head_k_dim, int head_v_dim, int key_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int num_heads; int head_k_dim; int head_v_dim; int key_dim; }
        pc = {num_heads, head_k_dim, head_v_dim, key_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_OUTPUT;
    dp.bufs[0] = state; dp.bufs[1] = qkv; dp.bufs[2] = y;
    dp.num_bufs = 3;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_ssm_norm_gate(GpuBuf y, GpuBuf z, GpuBuf norm_w,
                      int num_heads, int head_v_dim, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int head_v_dim; float eps; } pc = {head_v_dim, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SSM_NORM_GATE;
    dp.bufs[0] = y; dp.bufs[1] = z; dp.bufs[2] = norm_w;
    dp.num_bufs = 3;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_deinterleave_qgate(GpuBuf qfull, GpuBuf q, GpuBuf qgate,
                           int num_heads, int head_dim) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    int total = num_heads * head_dim;
    struct { int num_heads; int head_dim; } pc = {num_heads, head_dim};
    DispatchParams dp = {0};
    dp.pipe = PIPE_DEINTERLEAVE_QGATE;
    dp.bufs[0] = qfull; dp.bufs[1] = q; dp.bufs[2] = qgate;
    dp.num_bufs = 3;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = (total + 255) / 256; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_sigmoid_gate(GpuBuf out_buf, GpuBuf gate_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SIGMOID_GATE;
    dp.bufs[0] = out_buf; dp.bufs[1] = gate_buf;
    dp.num_bufs = 2;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = (n + 255) / 256; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_shared_expert_gate(GpuBuf out_buf, GpuBuf gate_w_buf, GpuBuf input_buf, int n) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;

    struct { int n; } pc = {n};
    DispatchParams dp = {0};
    dp.pipe = PIPE_SHARED_EXPERT_GATE;
    dp.bufs[0] = out_buf; dp.bufs[1] = gate_w_buf; dp.bufs[2] = input_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = 1; dp.groups_y = 1; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

// ---------------------------------------------------------------------------
// Batch operations exposed for hybrid prefill
// ---------------------------------------------------------------------------

int gpu_batch_rope(GpuBuf q, GpuBuf k, GpuBuf cos_table, GpuBuf sin_table,
                   int num_heads, int num_kv_heads, int head_dim, int rope_dim,
                   int start_pos, int neox, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    struct { int nh; int nkv; int hd; int rd; int pos; int neox; } pc =
        {num_heads, num_kv_heads, head_dim, rope_dim, start_pos, neox};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ROPE;
    dp.bufs[0] = q; dp.bufs[1] = k; dp.bufs[2] = cos_table; dp.bufs[3] = sin_table;
    dp.num_bufs = 4;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = (num_heads > num_kv_heads ? num_heads : num_kv_heads);
    dp.groups_y = npos; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_batch_kv_store(GpuBuf k_cache, GpuBuf v_cache, GpuBuf k, GpuBuf v,
                       int start_pos, int kv_dim, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    BufferAlloc* kc = get_buf(k_cache);
    BufferAlloc* vc = get_buf(v_cache);
    BufferAlloc* kb = get_buf(k);
    BufferAlloc* vb = get_buf(v);
    if (!kc || !vc || !kb || !vb) return GPU_ERR_DISPATCH;

    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier_(g.cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);

    VkBufferCopy kr = {0, (uint64_t)start_pos * kv_dim * 4, (uint64_t)npos * kv_dim * 4};
    vkCmdCopyBuffer_(g.cmd_buf, kb->buffer, kc->buffer, 1, &kr);
    VkBufferCopy vr = {0, (uint64_t)start_pos * kv_dim * 4, (uint64_t)npos * kv_dim * 4};
    vkCmdCopyBuffer_(g.cmd_buf, vb->buffer, vc->buffer, 1, &vr);

    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier_(g.cmd_buf,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
    g.dispatch_count++;
    return GPU_OK;
}

int gpu_batch_attention(GpuBuf out, GpuBuf q, GpuBuf k_cache, GpuBuf v_cache,
                        int num_heads, int num_kv_heads, int head_dim,
                        int kv_dim, int start_seq_len, float scale, int npos) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    struct { int nh; int nkv; int hd; int kvd; int sl; float sc; } pc =
        {num_heads, num_kv_heads, head_dim, kv_dim, start_seq_len, scale};
    DispatchParams dp = {0};
    dp.pipe = PIPE_ATTENTION;
    dp.bufs[0] = out; dp.bufs[1] = q;
    dp.bufs[2] = k_cache; dp.bufs[3] = v_cache;
    dp.num_bufs = 4;
    dp.push_data = &pc; dp.push_size = sizeof(pc);
    dp.groups_x = num_heads; dp.groups_y = npos; dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_batch_add_bias2(GpuBuf dst, GpuBuf bias, GpuBuf scratch,
                        int elems_per_pos, int npos) {
    return gpu_batch_add_bias_expand(dst, bias, scratch, elems_per_pos, npos);
}

// dp4a_safe_type: returns 1 if the dp4a path should be used for this type+size.
// Matches llama.cpp's ggml_vk_should_use_mmvq heuristic for post-Turing NVIDIA:
//  - Q3_K, Q6_K: never (2-byte alignment issues)
//  - Q8_0: never (float dequant is cheap enough)
//  - k <= 4096: never (quantization overhead outweighs dp4a benefit at small dims)
//  - MXFP4, Q8_0 on post-Turing: never (only benefits pre-Turing)
static int dp4a_safe_type(int qtype, int cols) {
    if (qtype == QTYPE_Q3_K || qtype == QTYPE_Q6_K) return 0;
    if (qtype == QTYPE_Q8_0) return 0;
    if (cols <= 4096) return 0;
    switch (qtype) {
    case QTYPE_Q4_0: case QTYPE_Q5_0:
    case QTYPE_Q4_K: case QTYPE_Q5_K:
    case QTYPE_IQ3_XXS: case QTYPE_IQ4_XS:
    case 20: /* IQ4_NL */
    case QTYPE_IQ3_S:
        return 1;
    case 39: /* MXFP4 — llama.cpp disables on post-Turing NVIDIA */
        return 0;
    default:
        return 0;
    }
}

// Set DLGO_FUSED_Q4K=1 to enable experimental fused Q4_K shader.
// Currently slower than separate quantize+dp4a due to shared memory pressure.
static int g_fused_q4k_enabled = -1;
static int fused_q4k_available(void) {
    if (g_fused_q4k_enabled < 0) {
        const char* env = getenv("DLGO_FUSED_Q4K");
        g_fused_q4k_enabled = (env && env[0] == '1') ? 1 : 0;
    }
    return g_fused_q4k_enabled;
}

// Can this Q4_K dispatch use the fused shader? Requires cols <= 8192 for shared memory.
static int q4k_fused_ok(int dp4a_ok, int qtype, int cols) {
    return fused_q4k_available() && dp4a_ok && qtype == QTYPE_Q4_K && cols <= 8192;
}

// Returns true if this dp4a type needs a separate Q8_1 quantization step.
// Q4_K uses the fused shader when available and cols <= 8192.
static int needs_separate_q8(int qtype, int cols) {
    if (fused_q4k_available() && qtype == QTYPE_Q4_K && cols <= 8192) return 0;
    return dp4a_safe_type(qtype, cols);
}

// smart_matvec: uses dp4a when the tensor type supports it, else float.
// For Q4_K with dp4a and cols <= 8192, uses the fused shader that reads float_input directly.
static int smart_matvec(GpuBuf out, GpuBuf weights,
                        int dp4a_ok, GpuBuf q8_input, GpuBuf float_input,
                        int rows, int cols, int qtype) {
    if (q4k_fused_ok(dp4a_ok, qtype, cols))
        return gpu_matvec_dp4a_fused(out, weights, float_input, rows, cols);
    if (dp4a_ok && dp4a_safe_type(qtype, cols))
        return gpu_matvec_dp4a(out, weights, q8_input, rows, cols, qtype);
    return gpu_matvec(out, weights, float_input, rows, cols, qtype);
}

// ---------------------------------------------------------------------------
// Fused layer forward — all dispatches for one transformer layer in a single
// C call, eliminating per-operation CGo overhead.
// ---------------------------------------------------------------------------
int gpu_forward_layer(const GpuLayerConf* lc, int pos, int seq_len, float scale,
                      GpuBuf next_attn_norm) {
    int dim = lc->dim;
    int head_dim = lc->head_dim;
    int num_heads = lc->num_heads;
    int num_kv_heads = lc->num_kv_heads;
    int kv_dim = lc->kv_dim;

    // core_type 1 = SSM: Go side already filled attn_proj, skip to residual+FFN
    int dp4a_ok = lc->use_dp4a && lc->q8_1_scratch;

    if (lc->core_type == 0) {
        // Q/K/V MatVecs — per-tensor dp4a: quantize once, each matvec picks dp4a or float.
        // Q4_K uses fused shader (no separate quantize needed).
        gpu_barrier();
        int any_qkv_needs_q8 = dp4a_ok && (needs_separate_q8(lc->wq_type, lc->wq_cols) ||
                                             needs_separate_q8(lc->wk_type, lc->wk_cols) ||
                                             needs_separate_q8(lc->wv_type, lc->wv_cols));
        if (any_qkv_needs_q8) {
            gpu_quantize_q8_1(lc->q8_1_scratch, lc->x_norm, dim);
            gpu_barrier();
        }
        smart_matvec(lc->q, lc->wq, dp4a_ok, lc->q8_1_scratch, lc->x_norm, lc->wq_rows, lc->wq_cols, lc->wq_type);
        smart_matvec(lc->k, lc->wk, dp4a_ok, lc->q8_1_scratch, lc->x_norm, lc->wk_rows, lc->wk_cols, lc->wk_type);
        smart_matvec(lc->v, lc->wv, dp4a_ok, lc->q8_1_scratch, lc->x_norm, lc->wv_rows, lc->wv_cols, lc->wv_type);

        if (lc->bq || lc->bk || lc->bv || lc->q_norm_w) {
            gpu_barrier();
            if (lc->bq) gpu_add(lc->q, lc->q, lc->bq, num_heads * head_dim);
            if (lc->bk) gpu_add(lc->k, lc->k, lc->bk, kv_dim);
            if (lc->bv) gpu_add(lc->v, lc->v, lc->bv, kv_dim);
            if (lc->q_norm_w) {
                gpu_barrier();
                gpu_rmsnorm_heads(lc->q, lc->q_norm_w, num_heads, head_dim, lc->rms_eps);
                gpu_rmsnorm_heads(lc->k, lc->k_norm_w, num_kv_heads, head_dim, lc->rms_eps);
            }
        }

        gpu_barrier();
        gpu_rope(lc->q, lc->k, lc->rope_cos_table, lc->rope_sin_table, num_heads, num_kv_heads, head_dim, lc->rope_dim, pos, lc->rope_neox);
        gpu_barrier();
        gpu_kv_store_f16(lc->k_cache, lc->v_cache, lc->k, lc->v, pos, kv_dim);

        {
            gpu_barrier();
            int win_start = 0;
            if (lc->sliding_window > 0 && seq_len > lc->sliding_window)
                win_start = seq_len - lc->sliding_window;
            if (lc->attn_sinks) {
                gpu_attention_f16_sinks(lc->attn_out, lc->q, lc->k_cache, lc->v_cache,
                                        lc->attn_sinks, num_heads, num_kv_heads, head_dim, kv_dim, seq_len, scale, lc->attn_logit_softcap, win_start);
            } else {
                gpu_attention_f16(lc->attn_out, lc->q, lc->k_cache, lc->v_cache,
                                  num_heads, num_kv_heads, head_dim, kv_dim, seq_len, scale, lc->attn_logit_softcap, win_start);
            }
        }

        gpu_barrier();
        if (q4k_fused_ok(dp4a_ok, lc->wo_type, lc->wo_cols)) {
            gpu_matvec_dp4a_fused(lc->attn_proj, lc->wo, lc->attn_out, lc->wo_rows, lc->wo_cols);
        } else if (dp4a_ok && dp4a_safe_type(lc->wo_type, lc->wo_cols)) {
            gpu_quantize_q8_1(lc->q8_1_scratch, lc->attn_out, num_heads * head_dim);
            gpu_barrier();
            gpu_matvec_dp4a(lc->attn_proj, lc->wo, lc->q8_1_scratch, lc->wo_rows, lc->wo_cols, lc->wo_type);
        } else {
            gpu_matvec(lc->attn_proj, lc->wo, lc->attn_out, lc->wo_rows, lc->wo_cols, lc->wo_type);
        }
        if (lc->bo) {
            gpu_barrier();
            gpu_add(lc->attn_proj, lc->attn_proj, lc->bo, dim);
        }
    }

    if (lc->residual_type == 0) {
        gpu_barrier();
        if (lc->post_attn_norm_w) {
            gpu_rmsnorm(lc->attn_proj, lc->attn_proj, lc->post_attn_norm_w, dim, lc->rms_eps);
            gpu_barrier();
        }
        gpu_add_rmsnorm(lc->ffn_norm, lc->ffn_in, lc->x, lc->attn_proj, lc->ffn_norm_w, dim, lc->rms_eps);

        if (lc->ffn_type == 3) {
            if (lc->moe_conf) {
                int mrc;
                if (lc->moe_use_native) {
                    mrc = gpu_forward_moe_ffn_native(lc->moe_conf);
                } else {
                    mrc = gpu_forward_moe_ffn(lc->moe_conf);
                }
                if (mrc != GPU_OK) return mrc;

                gpu_barrier();
                if (next_attn_norm) {
                    gpu_add_rmsnorm(lc->x_norm, lc->x, lc->ffn_in, lc->ffn_out, next_attn_norm, dim, lc->rms_eps);
                } else {
                    gpu_add(lc->x, lc->ffn_in, lc->ffn_out, dim);
                }
                return GPU_OK;
            }
            return GPU_OK;
        }

        // FFN — per-tensor dp4a for gate/up input. Q4_K uses fused shader.
        gpu_barrier();
        int any_ffn_needs_q8 = dp4a_ok && (needs_separate_q8(lc->gate_type, lc->gate_cols) ||
                                            needs_separate_q8(lc->up_type, lc->up_cols));
        if (any_ffn_needs_q8) {
            gpu_quantize_q8_1(lc->q8_1_scratch, lc->ffn_norm, dim);
            gpu_barrier();
        }

        if (lc->ffn_type == 0) {
            smart_matvec(lc->gate, lc->ffn_gate_w, dp4a_ok, lc->q8_1_scratch, lc->ffn_norm, lc->gate_rows, lc->gate_cols, lc->gate_type);
            smart_matvec(lc->up, lc->ffn_up_w, dp4a_ok, lc->q8_1_scratch, lc->ffn_norm, lc->up_rows, lc->up_cols, lc->up_type);
            gpu_barrier();
            gpu_swiglu(lc->hidden, lc->gate, lc->up, lc->gate_rows);
        } else if (lc->ffn_type == 1) {
            smart_matvec(lc->gate, lc->ffn_gate_w, dp4a_ok, lc->q8_1_scratch, lc->ffn_norm, lc->gate_rows, lc->gate_cols, lc->gate_type);
            smart_matvec(lc->up, lc->ffn_up_w, dp4a_ok, lc->q8_1_scratch, lc->ffn_norm, lc->up_rows, lc->up_cols, lc->up_type);
            gpu_barrier();
            gpu_geglu(lc->hidden, lc->gate, lc->up, lc->gate_rows);
        } else {
            smart_matvec(lc->up, lc->ffn_up_w, dp4a_ok, lc->q8_1_scratch, lc->ffn_norm, lc->up_rows, lc->up_cols, lc->up_type);
            gpu_barrier();
            gpu_gelu(lc->up, lc->up_rows);
        }

        // FFN down projection
        gpu_barrier();
        GpuBuf hidden_buf = (lc->ffn_type == 2) ? lc->up : lc->hidden;
        int hidden_dim = (lc->ffn_type == 2) ? lc->up_rows : lc->gate_rows;
        if (q4k_fused_ok(dp4a_ok, lc->down_type, lc->down_cols)) {
            gpu_matvec_dp4a_fused(lc->ffn_out, lc->ffn_down_w, hidden_buf, lc->down_rows, lc->down_cols);
        } else if (dp4a_ok && dp4a_safe_type(lc->down_type, lc->down_cols)) {
            gpu_quantize_q8_1(lc->q8_1_scratch, hidden_buf, hidden_dim);
            gpu_barrier();
            gpu_matvec_dp4a(lc->ffn_out, lc->ffn_down_w, lc->q8_1_scratch, lc->down_rows, lc->down_cols, lc->down_type);
        } else {
            gpu_matvec(lc->ffn_out, lc->ffn_down_w, hidden_buf, lc->down_rows, lc->down_cols, lc->down_type);
        }

        gpu_barrier();
        if (lc->post_ffn_norm_w) {
            gpu_rmsnorm(lc->ffn_out, lc->ffn_out, lc->post_ffn_norm_w, dim, lc->rms_eps);
            gpu_barrier();
        }
        if (next_attn_norm) {
            gpu_add_rmsnorm(lc->x_norm, lc->x, lc->ffn_in, lc->ffn_out, next_attn_norm, dim, lc->rms_eps);
        } else {
            gpu_add(lc->x, lc->ffn_in, lc->ffn_out, dim);
        }
    } else {
        // Parallel residual: attn and FFN share the same input (x_norm)
        gpu_barrier();
        GpuBuf ffn_input = lc->x_norm;
        int any_par_needs_q8 = dp4a_ok && (needs_separate_q8(lc->gate_type, lc->gate_cols) ||
                                            needs_separate_q8(lc->up_type, lc->up_cols));
        if (any_par_needs_q8) {
            gpu_quantize_q8_1(lc->q8_1_scratch, ffn_input, dim);
            gpu_barrier();
        }

        if (lc->ffn_type == 0) {
            smart_matvec(lc->gate, lc->ffn_gate_w, dp4a_ok, lc->q8_1_scratch, ffn_input, lc->gate_rows, lc->gate_cols, lc->gate_type);
            smart_matvec(lc->up, lc->ffn_up_w, dp4a_ok, lc->q8_1_scratch, ffn_input, lc->up_rows, lc->up_cols, lc->up_type);
            gpu_barrier();
            gpu_swiglu(lc->hidden, lc->gate, lc->up, lc->gate_rows);
        } else if (lc->ffn_type == 1) {
            smart_matvec(lc->gate, lc->ffn_gate_w, dp4a_ok, lc->q8_1_scratch, ffn_input, lc->gate_rows, lc->gate_cols, lc->gate_type);
            smart_matvec(lc->up, lc->ffn_up_w, dp4a_ok, lc->q8_1_scratch, ffn_input, lc->up_rows, lc->up_cols, lc->up_type);
            gpu_barrier();
            gpu_geglu(lc->hidden, lc->gate, lc->up, lc->gate_rows);
        } else {
            smart_matvec(lc->up, lc->ffn_up_w, dp4a_ok, lc->q8_1_scratch, ffn_input, lc->up_rows, lc->up_cols, lc->up_type);
            gpu_barrier();
            gpu_gelu(lc->up, lc->up_rows);
        }

        // FFN down projection (parallel path)
        gpu_barrier();
        GpuBuf par_hidden = (lc->ffn_type == 2) ? lc->up : lc->hidden;
        int par_hidden_dim = (lc->ffn_type == 2) ? lc->up_rows : lc->gate_rows;
        if (q4k_fused_ok(dp4a_ok, lc->down_type, lc->down_cols)) {
            gpu_matvec_dp4a_fused(lc->ffn_out, lc->ffn_down_w, par_hidden, lc->down_rows, lc->down_cols);
        } else if (dp4a_ok && dp4a_safe_type(lc->down_type, lc->down_cols)) {
            gpu_quantize_q8_1(lc->q8_1_scratch, par_hidden, par_hidden_dim);
            gpu_barrier();
            gpu_matvec_dp4a(lc->ffn_out, lc->ffn_down_w, lc->q8_1_scratch, lc->down_rows, lc->down_cols, lc->down_type);
        } else {
            gpu_matvec(lc->ffn_out, lc->ffn_down_w, par_hidden, lc->down_rows, lc->down_cols, lc->down_type);
        }
        gpu_barrier();
        gpu_add(lc->x, lc->x, lc->attn_proj, dim);
        gpu_barrier();
        gpu_add(lc->x, lc->x, lc->ffn_out, dim);
        if (next_attn_norm) {
            gpu_barrier();
            gpu_rmsnorm(lc->x_norm, lc->x, next_attn_norm, dim, lc->rms_eps);
        }
    }

    return GPU_OK;
} // end gpu_forward_layer

int gpu_batch_rmsnorm(GpuBuf out_buf, GpuBuf x_buf, GpuBuf weight_buf, int n, int npos, float eps) {
    if (!g.initialized) return GPU_ERR_INIT_FAIL;
    if (!g.pipelines_ready && gpu_load_pipelines() != GPU_OK) return GPU_ERR_SHADER;
    if (npos <= 0) return GPU_OK;

    struct { int n; float eps; } pc = {n, eps};
    DispatchParams dp = {0};
    dp.pipe = PIPE_RMSNORM;
    dp.bufs[0] = out_buf;
    dp.bufs[1] = x_buf;
    dp.bufs[2] = weight_buf;
    dp.num_bufs = 3;
    dp.push_data = &pc;
    dp.push_size = sizeof(pc);
    dp.groups_x = 1;
    dp.groups_y = npos;
    dp.groups_z = 1;
    return dispatch_compute(&dp);
}

int gpu_forward_layer_batch(const GpuLayerConf* lc, int npos, int start_pos,
                            float scale, GpuBuf next_attn_norm) {
    if (npos <= 0) return GPU_OK;
    if (npos == 1) return gpu_forward_layer(lc, start_pos, start_pos + 1, scale, next_attn_norm);

    int dim = lc->dim;
    int head_dim = lc->head_dim;
    int num_heads = lc->num_heads;
    int num_kv_heads = lc->num_kv_heads;
    int kv_dim = lc->kv_dim;

    // core_type 1 = SSM/GatedQ: Go side already filled attn_proj, skip to residual+FFN
    if (lc->core_type == 0) {

    gpu_barrier();
    gpu_batch_matvec(lc->q, lc->wq, lc->x_norm, lc->wq_rows, lc->wq_cols, npos, lc->wq_type);
    gpu_batch_matvec(lc->k, lc->wk, lc->x_norm, lc->wk_rows, lc->wk_cols, npos, lc->wk_type);
    gpu_batch_matvec(lc->v, lc->wv, lc->x_norm, lc->wv_rows, lc->wv_cols, npos, lc->wv_type);

    if (lc->bq || lc->bk || lc->bv) {
        gpu_barrier();
        if (lc->bq) {
            int rc = gpu_batch_add_bias_expand(lc->q, lc->bq, lc->attn_out, num_heads * head_dim, npos);
            if (rc != GPU_OK) return rc;
        }
        if (lc->bk) {
            int rc = gpu_batch_add_bias_expand(lc->k, lc->bk, lc->attn_out, kv_dim, npos);
            if (rc != GPU_OK) return rc;
        }
        if (lc->bv) {
            int rc = gpu_batch_add_bias_expand(lc->v, lc->bv, lc->attn_out, kv_dim, npos);
            if (rc != GPU_OK) return rc;
        }
    }

    if (lc->q_norm_w) {
        gpu_barrier();
        {
            struct { int hd; float eps; } pc = {head_dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_RMSNORM_HEADS;
            dp.bufs[0] = lc->q; dp.bufs[1] = lc->q_norm_w; dp.num_bufs = 2;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = num_heads; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
        }
        {
            struct { int hd; float eps; } pc = {head_dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_RMSNORM_HEADS;
            dp.bufs[0] = lc->k; dp.bufs[1] = lc->k_norm_w; dp.num_bufs = 2;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = num_kv_heads; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
        }
    }

    gpu_barrier();
    {
        struct { int nh; int nkv; int hd; int rd; int pos; int neox; } pc =
            {num_heads, num_kv_heads, head_dim, lc->rope_dim, start_pos, lc->rope_neox};
        DispatchParams dp = {0};
        dp.pipe = PIPE_ROPE;
        dp.bufs[0] = lc->q; dp.bufs[1] = lc->k;
        dp.bufs[2] = lc->rope_cos_table; dp.bufs[3] = lc->rope_sin_table;
        dp.num_bufs = 4;
        dp.push_data = &pc; dp.push_size = sizeof(pc);
        dp.groups_x = (num_heads > num_kv_heads ? num_heads : num_kv_heads);
        dp.groups_y = npos; dp.groups_z = 1;
        dispatch_compute(&dp);
    }

    {
        gpu_barrier();
        gpu_batch_kv_store_f16(lc->k_cache, lc->v_cache, lc->k, lc->v, start_pos, kv_dim, npos);
    }

    {
        gpu_barrier();
        if (lc->attn_sinks) {
            gpu_batch_attention_f16_sinks(lc->attn_out, lc->q, lc->k_cache, lc->v_cache,
                                          lc->attn_sinks, num_heads, num_kv_heads, head_dim, kv_dim,
                                          start_pos + 1, scale, lc->attn_logit_softcap, npos);
        } else {
            gpu_batch_attention_f16(lc->attn_out, lc->q, lc->k_cache, lc->v_cache,
                                    num_heads, num_kv_heads, head_dim, kv_dim,
                                    start_pos + 1, scale, lc->attn_logit_softcap, npos);
        }
    }

    gpu_barrier();
    gpu_batch_matvec(lc->attn_proj, lc->wo, lc->attn_out, lc->wo_rows, lc->wo_cols, npos, lc->wo_type);
    if (lc->bo) {
        gpu_barrier();
        gpu_batch_add_bias_expand(lc->attn_proj, lc->bo, lc->attn_out, dim, npos);
    }

    } // end core_type == 0

    if (lc->residual_type == 0) {
        gpu_barrier();
        if (lc->post_attn_norm_w) {
            struct { int n; float eps; } pc = {dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_RMSNORM;
            dp.bufs[0] = lc->attn_proj; dp.bufs[1] = lc->attn_proj;
            dp.bufs[2] = lc->post_attn_norm_w; dp.num_bufs = 3;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = 1; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
            gpu_barrier();
        }
        {
            struct { int n; float eps; } pc = {dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_ADD_RMSNORM;
            dp.bufs[0] = lc->ffn_norm; dp.bufs[1] = lc->ffn_in;
            dp.bufs[2] = lc->x; dp.bufs[3] = lc->attn_proj;
            dp.bufs[4] = lc->ffn_norm_w; dp.num_bufs = 5;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = 1; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
        }

        if (lc->ffn_type == 3) {
            return GPU_OK;
        }

        gpu_barrier();
        if (lc->ffn_type == 0) {
            gpu_batch_matvec(lc->gate, lc->ffn_gate_w, lc->ffn_norm, lc->gate_rows, lc->gate_cols, npos, lc->gate_type);
            gpu_batch_matvec(lc->up, lc->ffn_up_w, lc->ffn_norm, lc->up_rows, lc->up_cols, npos, lc->up_type);
            gpu_barrier();
            gpu_swiglu(lc->hidden, lc->gate, lc->up, lc->gate_rows * npos);
            gpu_barrier();
            gpu_batch_matvec(lc->ffn_out, lc->ffn_down_w, lc->hidden, lc->down_rows, lc->down_cols, npos, lc->down_type);
        } else if (lc->ffn_type == 1) {
            gpu_batch_matvec(lc->gate, lc->ffn_gate_w, lc->ffn_norm, lc->gate_rows, lc->gate_cols, npos, lc->gate_type);
            gpu_batch_matvec(lc->up, lc->ffn_up_w, lc->ffn_norm, lc->up_rows, lc->up_cols, npos, lc->up_type);
            gpu_barrier();
            gpu_geglu(lc->hidden, lc->gate, lc->up, lc->gate_rows * npos);
            gpu_barrier();
            gpu_batch_matvec(lc->ffn_out, lc->ffn_down_w, lc->hidden, lc->down_rows, lc->down_cols, npos, lc->down_type);
        } else {
            gpu_batch_matvec(lc->up, lc->ffn_up_w, lc->ffn_norm, lc->up_rows, lc->up_cols, npos, lc->up_type);
            gpu_barrier();
            gpu_gelu(lc->up, lc->up_rows * npos);
            gpu_barrier();
            gpu_batch_matvec(lc->ffn_out, lc->ffn_down_w, lc->up, lc->down_rows, lc->down_cols, npos, lc->down_type);
        }

        gpu_barrier();
        if (lc->post_ffn_norm_w) {
            struct { int n; float eps; } pc = {dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_RMSNORM;
            dp.bufs[0] = lc->ffn_out; dp.bufs[1] = lc->ffn_out;
            dp.bufs[2] = lc->post_ffn_norm_w; dp.num_bufs = 3;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = 1; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
            gpu_barrier();
        }
        if (next_attn_norm) {
            struct { int n; float eps; } pc = {dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_ADD_RMSNORM;
            dp.bufs[0] = lc->x_norm; dp.bufs[1] = lc->x;
            dp.bufs[2] = lc->ffn_in; dp.bufs[3] = lc->ffn_out;
            dp.bufs[4] = next_attn_norm; dp.num_bufs = 5;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = 1; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
        } else {
            gpu_add(lc->x, lc->ffn_in, lc->ffn_out, dim * npos);
        }
    } else {
        GpuBuf ffn_input = lc->x_norm;
        gpu_barrier();
        if (lc->ffn_type == 0) {
            gpu_batch_matvec(lc->gate, lc->ffn_gate_w, ffn_input, lc->gate_rows, lc->gate_cols, npos, lc->gate_type);
            gpu_batch_matvec(lc->up, lc->ffn_up_w, ffn_input, lc->up_rows, lc->up_cols, npos, lc->up_type);
            gpu_barrier();
            gpu_swiglu(lc->hidden, lc->gate, lc->up, lc->gate_rows * npos);
            gpu_barrier();
            gpu_batch_matvec(lc->ffn_out, lc->ffn_down_w, lc->hidden, lc->down_rows, lc->down_cols, npos, lc->down_type);
        } else if (lc->ffn_type == 1) {
            gpu_batch_matvec(lc->gate, lc->ffn_gate_w, ffn_input, lc->gate_rows, lc->gate_cols, npos, lc->gate_type);
            gpu_batch_matvec(lc->up, lc->ffn_up_w, ffn_input, lc->up_rows, lc->up_cols, npos, lc->up_type);
            gpu_barrier();
            gpu_geglu(lc->hidden, lc->gate, lc->up, lc->gate_rows * npos);
            gpu_barrier();
            gpu_batch_matvec(lc->ffn_out, lc->ffn_down_w, lc->hidden, lc->down_rows, lc->down_cols, npos, lc->down_type);
        } else {
            gpu_batch_matvec(lc->up, lc->ffn_up_w, ffn_input, lc->up_rows, lc->up_cols, npos, lc->up_type);
            gpu_barrier();
            gpu_gelu(lc->up, lc->up_rows * npos);
            gpu_barrier();
            gpu_batch_matvec(lc->ffn_out, lc->ffn_down_w, lc->up, lc->down_rows, lc->down_cols, npos, lc->down_type);
        }
        gpu_barrier();
        gpu_add(lc->x, lc->x, lc->attn_proj, dim * npos);
        gpu_barrier();
        gpu_add(lc->x, lc->x, lc->ffn_out, dim * npos);
        if (next_attn_norm) {
            gpu_barrier();
            struct { int n; float eps; } pc = {dim, lc->rms_eps};
            DispatchParams dp = {0};
            dp.pipe = PIPE_RMSNORM;
            dp.bufs[0] = lc->x_norm; dp.bufs[1] = lc->x;
            dp.bufs[2] = next_attn_norm; dp.num_bufs = 3;
            dp.push_data = &pc; dp.push_size = sizeof(pc);
            dp.groups_x = 1; dp.groups_y = npos; dp.groups_z = 1;
            dispatch_compute(&dp);
        }
    }

    return GPU_OK;
} // end gpu_forward_layer_batch
