/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// KytyPS5 (Demon's Souls): tile lists a compute chain builds from the G-buffer through device addresses in a
// per-dispatch block (<VKKyty><TileChain>); the chain runs for the right eye too, on a copy naming the right's lists.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include <smmintrin.h>
#include <stdio.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <string.h>
#include <unordered_map>
#include <vector>

namespace wiz3d { namespace vk {

namespace {

const VkDeviceSize kTwinBytes = 32ull << 20;     // the lists are ~1.5 MB at 4K
const VkDeviceSize kRingBytes = 8ull << 20;      // ~50 blocks a frame: many frames in flight's worth
const size_t       kBlockBytes = 4096;           // the largest block of 266 dumped shaders is 2.2 KB
const unsigned     kStaleFrames = 8;             // a chain that stopped running: its addresses may have moved on

struct TileRange { VkDeviceAddress left; VkDeviceSize size, twinOff; unsigned frame; };

// A chain shader and the block offsets of the lists it fills, read from its SPIR-V (the CRC pins the offsets): the
// profile's <VKKyty><TileChain ComputeShader= Lists="a b"/> children (TilesLoadProfile).
struct ChainShader { uint32_t crc; uint32_t lists[2]; };
std::vector<ChainShader> s_chain;

const ChainShader* ChainFor(uint32_t crc)
{
    for (const ChainShader& c : s_chain) if (c.crc == crc) return &c;
    return nullptr;
}

} // namespace

void TilesLoadProfile()
{
    s_chain.clear();
    for (const ProfileEntry* e : ChildEntries("VKKyty"))
    {
        ChainShader c = {};
        if (EntryValue(*e, "@") != "TileChain" || sscanf_s(EntryValue(*e, "ComputeShader").c_str(), "0x%x", &c.crc) != 1) continue;
        sscanf_s(EntryValue(*e, "Lists").c_str(), "%u %u", &c.lists[0], &c.lists[1]);
        for (uint32_t& k : c.lists)
            if (k >= kBlockBytes)
            {
                Log("[tile] profile: chain %08X list offset %u is past the %zu-byte block: dropped\n", c.crc, k, kBlockBytes);
                k = 0;
            }
        s_chain.push_back(c);
    }
    Log("[tile] profile: %zu chain shader(s)\n", s_chain.size());
}

struct TileState
{
    std::mutex lock;
    std::map<VkDeviceAddress, VkBuffer> bda;         // every device-address buffer, by its address
    std::unordered_map<VkBuffer, VkDeviceAddress> addrOf;
    VkBuffer twin = VK_NULL_HANDLE, ring = VK_NULL_HANDLE;
    VkDeviceMemory twinMem = VK_NULL_HANDLE, ringMem = VK_NULL_HANDLE;
    VkDeviceAddress twinAddr = 0, ringAddr = 0;
    uint8_t* ringMap = nullptr;
    VkDeviceSize twinHead = 0, ringHead = 0;
    bool failed = false;
    std::vector<TileRange> ranges;
    std::unordered_map<VkPipeline, bool> noLists;    // a block's layout is the shader's: its first scan decides
    uint64_t objGen = 0;
    unsigned chainRuns = 0, repoints = 0, logged = 0;
};

void TilesInit(DeviceData* d)
{
    if (Traits().tileLists) d->tiles = new TileState;
}

void TilesDestroy(DeviceData* d)
{
    TileState* t = d->tiles;
    if (!t) return;
    if (t->twin) d->DestroyBuffer(d->device, t->twin, nullptr);
    if (t->twinMem) d->FreeMemory(d->device, t->twinMem, nullptr);
    if (t->ring) d->DestroyBuffer(d->device, t->ring, nullptr);
    if (t->ringMem) d->FreeMemory(d->device, t->ringMem, nullptr);
    delete t;
    d->tiles = nullptr;
}

void TileNoteBuffer(DeviceData* d, VkBuffer buffer)
{
    TileState* t = d->tiles;
    if (!t || !d->GetBufferDeviceAddress) return;
    VkBufferUsageFlags usage = 0;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->buffers.find(buffer);
        if (it != d->buffers.end()) usage = it->second.usage;
    }
    if (!(usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) return;
    VkBufferDeviceAddressInfo info = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    info.buffer = buffer;
    const VkDeviceAddress a = d->GetBufferDeviceAddress(d->device, &info);
    std::lock_guard<std::mutex> g(t->lock);
    t->bda[a] = buffer;
    t->addrOf[buffer] = a;
}

// The game's buffers move (its compaction, a level load): a list's address then belongs to something else.
void TileForgetBuffer(DeviceData* d, VkBuffer buffer, VkDeviceSize size)
{
    TileState* t = d->tiles;
    if (!t) return;
    std::lock_guard<std::mutex> g(t->lock);
    auto it = t->addrOf.find(buffer);
    if (it == t->addrOf.end()) return;
    const VkDeviceAddress a = it->second;
    t->addrOf.erase(it);
    auto bit = t->bda.find(a);
    if (bit != t->bda.end() && bit->second == buffer) t->bda.erase(bit);
    t->ranges.erase(std::remove_if(t->ranges.begin(), t->ranges.end(), [&](const TileRange& r) { return r.left >= a && r.left < a + size; }), t->ranges.end());
}

static VkBuffer AddressBuffer(DeviceData* d, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, VkDeviceMemory* mem, VkDeviceAddress* addr)
{
    VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer b = VK_NULL_HANDLE;
    if (d->CreateBuffer(d->device, &ci, nullptr, &b) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->device, b, &req);
    VkMemoryAllocateFlagsInfo flags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, want, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, mem) != VK_SUCCESS || d->BindBufferMemory(d->device, b, *mem, 0) != VK_SUCCESS)
    {
        if (*mem) d->FreeMemory(d->device, *mem, nullptr);
        *mem = VK_NULL_HANDLE;
        d->DestroyBuffer(d->device, b, nullptr);
        return VK_NULL_HANDLE;
    }
    VkBufferDeviceAddressInfo info = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
    info.buffer = b;
    *addr = d->GetBufferDeviceAddress(d->device, &info);
    return b;
}

static bool EnsureBuffers(DeviceData* d, TileState* t)
{
    if (t->twin && t->ring) return true;
    if (t->failed || !d->GetBufferDeviceAddress) return false;
    t->twin = AddressBuffer(d, kTwinBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &t->twinMem, &t->twinAddr);
    t->ring = AddressBuffer(d, kRingBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &t->ringMem, &t->ringAddr);
    void* map = nullptr;
    if (!t->twin || !t->ring || d->MapMemory(d->device, t->ringMem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS)
    {
        t->failed = true;
        Log("[tile] no right-eye lists: buffers failed (twin %p ring %p)\n", (void*)t->twin, (void*)t->ring);
        return false;
    }
    t->ringMap = (uint8_t*)map;
    Log("[tile] right-eye lists at %016llX, blocks at %016llX\n", (unsigned long long)t->twinAddr, (unsigned long long)t->ringAddr);
    return true;
}

// The mapped bytes at a device address, or null: the game's device-local buffers have none, its upload ring does.
static const uint8_t* HostAt(DeviceData* d, TileState* t, VkDeviceAddress a, VkDeviceSize* avail)
{
    auto it = t->bda.upper_bound(a);
    if (it == t->bda.begin()) return nullptr;
    --it;
    std::shared_lock<std::shared_mutex> g(d->lock);
    auto bit = d->buffers.find(it->second);
    if (bit == d->buffers.end() || bit->second.size < (1ull << 20)) return nullptr;
    const VkDeviceSize into = a - it->first;
    if (into >= bit->second.size) return nullptr;
    auto mit = d->memory.find(bit->second.memory);
    if (mit == d->memory.end() || !mit->second.mapped) return nullptr;
    const VkDeviceSize memOff = bit->second.offset + into;
    if (memOff < mit->second.mapOff || memOff >= mit->second.mapOff + mit->second.mapSize) return nullptr;
    *avail = (std::min)(bit->second.size - into, mit->second.mapOff + mit->second.mapSize - memOff);
    return (const uint8_t*)mit->second.mapped + (memOff - mit->second.mapOff);
}

// The ring is write-combined: ordinary reads crawl there (ReBAR), streaming loads do not.
static void ReadBlock(uint8_t* dst, const uint8_t* src, size_t n)
{
    if ((uintptr_t)src & 15) { memcpy(dst, src, n); return; }
    _mm_mfence();
    for (size_t i = 0; i < n; i += 16)
        _mm_storeu_si128((__m128i*)(dst + i), _mm_stream_load_si128((__m128i*)(src + i)));
}

static void RegisterList(DeviceData* d, TileState* t, const uint8_t* block, uint32_t k, size_t n)
{
    if (k + 12 > n) return;
    uint64_t a = 0;
    uint32_t dwords = 0;
    memcpy(&a, block + k, 8);
    memcpy(&dwords, block + k + 8, 4);
    const VkDeviceSize size = (VkDeviceSize)dwords * 4;
    if (!a || !size || size > kTwinBytes / 2) return;
    for (TileRange& r : t->ranges)
        if (a >= r.left && a + size <= r.left + r.size) { r.frame = d->frames; return; }
    if (!EnsureBuffers(d, t)) return;
    t->ranges.erase(std::remove_if(t->ranges.begin(), t->ranges.end(), [&](const TileRange& r) { return r.left >= a && r.left < a + size; }), t->ranges.end());
    VkDeviceSize off = (t->twinHead + 255) & ~(VkDeviceSize)255;
    if (off + size > kTwinBytes) { t->ranges.clear(); off = 0; }
    t->ranges.push_back({ a, size, off, d->frames });
    t->twinHead = off + size;
    t->noLists.clear();
    Log("[tile] list %016llX, %llu KB, right copy at +%llX\n", (unsigned long long)a, (unsigned long long)(size >> 10), (unsigned long long)off);
}

bool TileChainShader(DeviceData* d, uint32_t crc)
{
    return d->tiles && ChainFor(crc);
}

bool TileRightBlock(DeviceData* d, VkPipeline pipeline, uint32_t crc, uint32_t* pc, uint32_t dwords)
{
    TileState* t = d->tiles;
    if (!t) return false;
    std::lock_guard<std::mutex> g(t->lock);
    const ChainShader* chain = ChainFor(crc);
    if (!chain)
    {
        if (t->ranges.empty()) return false;
        const uint64_t gen = d->objGen.load(std::memory_order_acquire);
        if (t->objGen != gen) { t->noLists.clear(); t->objGen = gen; }
        if (t->noLists.count(pipeline)) return false;
    }
    // The block's address sits among the push data (dwords 2-3, or 8-9 in some shaders): the pair landing in a mapped
    // buffer.
    const uint8_t* host = nullptr;
    VkDeviceSize avail = 0;
    uint32_t at = 0;
    for (; at + 1 < dwords; ++at)
    {
        const uint64_t a = (uint64_t)pc[at] | ((uint64_t)pc[at + 1] << 32);
        if (a && !(a & 15) && (host = HostAt(d, t, a, &avail)) != nullptr) break;
    }
    if (!host) return false;
    alignas(16) uint8_t block[kBlockBytes];
    const size_t n = (size_t)(std::min)((VkDeviceSize)kBlockBytes, avail & ~(VkDeviceSize)15);
    if (n < 16) return false;
    ReadBlock(block, host, n);
    if (chain)
    {
        for (uint32_t k : chain->lists) if (k) RegisterList(d, t, block, k, n);
        ++t->chainRuns;
    }
    bool hit = false;
    for (size_t off = 0; off + 12 <= n; off += 4)
    {
        uint64_t v;
        memcpy(&v, block + off, 8);
        if (!v) continue;
        for (const TileRange& r : t->ranges)
        {
            if (v < r.left || v >= r.left + r.size || d->frames - r.frame > kStaleFrames) continue;
            uint32_t len;
            memcpy(&len, block + off + 8, 4);
            if (v + (uint64_t)len * 4 > r.left + r.size) break;
            v += t->twinAddr + r.twinOff - r.left;
            memcpy(block + off, &v, 8);
            hit = true;
            break;
        }
    }
    if (!hit)
    {
        if (!chain) t->noLists[pipeline] = true;
        return false;
    }
    VkDeviceSize head = (t->ringHead + 15) & ~(VkDeviceSize)15;
    if (head + n > kRingBytes) head = 0;
    memcpy(t->ringMap + head, block, n);
    t->ringHead = head + n;
    const uint64_t right = t->ringAddr + head;
    pc[at] = (uint32_t)right;
    pc[at + 1] = (uint32_t)(right >> 32);
    ++t->repoints;
    if (gInfo.CollectDebugInformation && d->frames - t->logged >= 600)
    {
        t->logged = d->frames;
        Log("[tile %u] lists=%zu chain runs=%u right blocks=%u (since the last row)\n", d->frames, t->ranges.size(), t->chainRuns, t->repoints);
        t->chainRuns = t->repoints = 0;
    }
    return true;
}

bool TileRightArgs(DeviceData* d, VkBuffer buf, VkDeviceSize off, VkBuffer* out, VkDeviceSize* outOff)
{
    TileState* t = d->tiles;
    if (!t) return false;
    std::lock_guard<std::mutex> g(t->lock);
    auto it = t->addrOf.find(buf);
    if (it == t->addrOf.end()) return false;
    const VkDeviceAddress a = it->second + off;
    for (const TileRange& r : t->ranges)
    {
        if (a < r.left || a + 12 > r.left + r.size || d->frames - r.frame > kStaleFrames) continue;
        *out = t->twin;
        *outOff = r.twinOff + (a - r.left);
        return true;
    }
    return false;
}

}} // namespace wiz3d::vk
