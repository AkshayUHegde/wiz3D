/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Statistics, as DX9's BaseStereoRenderer_Utils.cpp: the frame report's coverage tallies and rows (CollectDebugInformation)
// and the bound shader's CRC lookup.

#include "StdAfx.h"
#include "Context11Proxy-inl.h"

namespace wiz3d
{

// A driver Map over 0.1 ms, tallied by what it mapped for the waiting-maps row (CollectDebugInformation).
void Context11Proxy::NoteSlowMap(ID3D11Resource* real, D3D11_MAP type, LONGLONG t0, bool right)
{
    static LARGE_INTEGER freq = {};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t1;
    QueryPerformanceCounter(&t1);
    const LONGLONG dt = t1.QuadPart - t0;
    if (!real || dt * 10000 < freq.QuadPart) return;
    static const char* const kType[] = { "?", "READ", "WRITE", "READ_WRITE", "DISCARD", "NO_OVERWRITE" };
    static const char* const kUsage[] = { "default", "immutable", "dynamic", "staging" };
    const char* t = (UINT)type < 6 ? kType[type] : "?";
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    real->GetType(&dim);
    char key[192];
    if (dim == D3D11_RESOURCE_DIMENSION_BUFFER)
    {
        D3D11_BUFFER_DESC bd = {};
        static_cast<ID3D11Buffer*>(real)->GetDesc(&bd);
        sprintf_s(key, "%sbuffer %u B %s cpu 0x%X bind 0x%X %s", right ? "right " : "", bd.ByteWidth, kUsage[bd.Usage & 3],
                  bd.CPUAccessFlags, bd.BindFlags, t);
    }
    else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        D3D11_TEXTURE2D_DESC d = {};
        static_cast<ID3D11Texture2D*>(real)->GetDesc(&d);
        sprintf_s(key, "tex2d %ux%u f%u %s cpu 0x%X bind 0x%X %s", d.Width, d.Height, (UINT)d.Format, kUsage[d.Usage & 3],
                  d.CPUAccessFlags, d.BindFlags, t);
    }
    else sprintf_s(key, "resource dim %d %s", (int)dim, t);
    SlowMaps& s = m_slowMaps[key];
    s.ticks += dt;
    ++s.count;
}

// Why a draw's geometry may reach the right eye uncorrected, for the frame report (CollectDebugInformation).
void Context11Proxy::TallyDrawCoverage()
{
    if (!m_boundVS) return;
    const ShaderAnalysis11Result* info = m_boundInfo[0];
    if (!info || !info->parsed)                 { ++m_drawsVSUnparsedThisFrame; return; }
    if (info->projection.matrixData.cb.empty()) { ++m_drawsVSNoMatrixThisFrame; return; }
    // A shift within the last few Presents is current: a game can present more than once per frame.
    unsigned newest = 0;
    bool anyEver = false, anySlot = false;
    for (auto it = info->projection.matrixData.cb.begin();
         it != info->projection.matrixData.cb.end(); ++it)
    {
        const DWORD slot = it->first;
        if (slot >= kMaxVSCBSlots || !m_boundVSCBs[slot]) continue;
        anySlot = true;
        Buffer11Proxy* b = TryUnwrapBuffer(m_boundVSCBs[slot]);
        if (!b || !b->EverShifted()) continue;
        anyEver = true;
        if (b->ShiftedFrame() > newest) newest = b->ShiftedFrame();
    }
    // The shift lands in the right-eye sibling, so it only reaches the right eye
    // if that sibling is what gets bound for the duplicated draw.
    for (auto it = info->projection.matrixData.cb.begin();
         it != info->projection.matrixData.cb.end(); ++it)
    {
        const DWORD slot = it->first;
        if (slot >= kMaxCBs || !m_boundVSCBs[slot]) continue;
        ID3D11Buffer* r = m_cbSlots[ST_VS].right[slot];
        if (!r || r == m_cbSlots[ST_VS].left[slot])
        {
            ++m_drawsVSNoSiblingCBThisFrame;
            ++m_uncorrectedVSDraws[BoundShaderCRC(m_parent, m_boundVS)];
            break;
        }
    }

    if (!anySlot) return;
    if (!anyEver)
    {
        ++m_drawsVSNeverShiftedThisFrame;
        ++m_uncorrectedVSDraws[BoundShaderCRC(m_parent, m_boundVS)];
        return;
    }
    if (s_frameId > newest + 4)
    {
        ++m_drawsVSStaleShiftThisFrame;
        ++m_uncorrectedVSDraws[BoundShaderCRC(m_parent, m_boundVS)];
    }
}

void Context11Proxy::LogFrameStats(unsigned frame)
{
    DDILog("  [frame %u] ctx=%p: draws=%u dispatches=%u cmdLists=%u recorded=%zu | CB writes: targeted=%u untargeted=%u"
           " | matrices: shifted=%u rejected=%u | duplicated=%u mono=%u uavSkip=%u uavDup=%u monoDSV=%u noRightRTV=%u"
           " depthOnly=%u noRightTarget=%u dispatchesDuplicated=%u\n",
           frame, this, m_drawsThisFrame, m_dispatchesThisFrame, m_cmdListsThisFrame, m_frameCommands.size(),
           m_cbTargetedThisFrame, m_cbUntargetedThisFrame, m_cbMatShiftedThisFrame, m_cbMatRejectedThisFrame,
           m_drawsDuplicatedThisFrame, m_drawsMonoThisFrame, m_drawsUavSkippedThisFrame, m_drawsUavDupThisFrame,
           m_drawsMonoDsvThisFrame, m_drawsNoRightRTVThisFrame, m_drawsDepthOnlyThisFrame,
           m_drawsNoRightTargetThisFrame, m_dispatchesDuplicatedThisFrame);

    // The vertex shaders of draws that reached the right eye uncorrected, most frequent first.
    std::vector<std::pair<unsigned, DWORD>> byUse;
    for (const auto& u : m_uncorrectedVSDraws) byUse.push_back(std::make_pair(u.second, u.first));
    std::sort(byUse.rbegin(), byUse.rend());
    std::string shaders;
    char one[32];
    for (size_t k = 0; k < byUse.size() && k < 12; ++k)
    {
        sprintf_s(one, " %08lX=%u", byUse[k].second, byUse[k].first);
        shaders += one;
    }
    DDILog("  [frame %u] uncorrected draws: vsUnparsed=%u vsNoMatrix=%u cbNeverShifted=%u cbStale=%u noSiblingCB=%u"
           " | by VS:%s\n", frame, m_drawsVSUnparsedThisFrame, m_drawsVSNoMatrixThisFrame,
           m_drawsVSNeverShiftedThisFrame, m_drawsVSStaleShiftThisFrame, m_drawsVSNoSiblingCBThisFrame,
           shaders.empty() ? " none" : shaders.c_str());

    std::string byCS;
    for (const auto& c : m_sharedUavByCS)
    {
        sprintf_s(one, " %08lX x%u", c.first, c.second);
        byCS += one;
    }
    DDILog("  [frame %u] right dispatches with shared UAVs twinned: %u (slots %u, copies %u, %.1f MB,"
           " write-only dropped %u, failed %u, counters unknown %u, twins held %u), by CS:%s\n", frame,
           m_sharedUavDispatchesThisFrame, m_sharedUavSlotsThisFrame, m_twinCopiesThisFrame,
           m_twinBytesThisFrame / 1048576.0, m_twinWriteOnlyThisFrame, m_twinFailedThisFrame,
           m_twinCounterUnknownThisFrame, (unsigned)m_uavTwins.size(), byCS.empty() ? " none" : byCS.c_str());
    DDILog("  [frame %u] left-eye textures read by the right: %u binds\n", frame, m_leftSRVsThisFrame);
    if (s_camP00 > 0.f || s_lightOnlyKept)
        DDILog("  [frame %u] camera: scaled camera matrices shifted with the camera's p00 %.4f: %u (cache %.4f)"
               " | light matrices left unshifted %u\n", frame, s_camP00, s_scaledByCamera, s_mat.projXScale,
               s_lightOnlyKept);
    // ring = cached view-projections to test against; bestMiss = the closest miss over the pass bound (<= 1 passes).
    if (gInfo.DetectInverseMatrix)
        DDILog("  [frame %u] invDetect: ring=%u fresh=%u inserts=%u evictedHot=%u scans=%u hits=%u patched=%u"
               " stale=%u (unshaped %u) learned=%u updCB=%u patched=%u bestMiss=%.2f\n", frame,
               s_mat.RingValid(), s_mat.RingFresh(), s_mat.ringInsertsThisFrame, s_mat.evictedHotThisFrame,
               m_invScansThisFrame, m_invHitsThisFrame, m_invPatchedThisFrame, s_mat.invStaleThisFrame,
               s_mat.invStaleShapeThisFrame, m_invLearnedThisFrame, m_updCBWritesThisFrame, m_updCBPatchedThisFrame,
               m_invBestMissThisFrame < 1e29f ? m_invBestMissThisFrame : -1.f);
    if (gInfo.DetectInverseMatrix)
    {
        std::string byPS;
        for (const auto& p : m_invDrawByPS)
        {
            sprintf_s(one, " %08lX x%u", p.first, p.second);
            byPS += one;
        }
        unsigned knownInverse = 0;
        for (const auto& k : m_psCameraMatrices) knownInverse += k.second.inverse ? 1 : 0;
        DDILog("  [frame %u] shaderCamera: buffers checked at the right draw/dispatch %u, inverses patched %u, previous"
               " cameras %u | known registers %u inverse + %u forward | by PS/CS:%s\n", frame, m_invDrawChecksThisFrame,
               m_invDrawPatchedThisFrame, m_fwdDrawPatchedThisFrame, knownInverse,
               (unsigned)m_psCameraMatrices.size() - knownInverse, byPS.empty() ? " none" : byPS.c_str());
    }
    DDILog("  [frame %u] camera scale: scene p00 %.4f (%u of %u clean camera matrices this frame), last cached %.4f\n",
           frame, s_sceneP00, s_sceneP00Votes, s_cleanP00Total, s_mat.projXScale);
    DDILog("  [frame %u] overlay: back-buffer draws without depth test %u, right copies restored %u | plain constants"
           " restored for matrix-less shaders %u | learned forward shifts withheld at write %u | domain-stage buffers"
           " patched %u\n", frame, m_overlayDrawsThisFrame, m_overlayRestoredThisFrame, m_plainRestoredThisFrame,
           m_learnedWithheldThisFrame, m_dsBuffersPatchedThisFrame);
    DDILog("  [frame %u] untargeted CB writes: noVS=%u vsUnparsed=%u vsNoMatrices=%u notAtVSSlot=%u atNonMatrixSlot=%u"
           " | binding resets %u\n", frame, m_untargetedWhy[0], m_untargetedWhy[1], m_untargetedWhy[2],
           m_untargetedWhy[3], m_untargetedWhy[4], m_bindResetsThisFrame);
    DDILog("  [frame %u] state getters answered with the game's bindings: constant buffers %u, shader resources %u,"
           " uavs %u\n", frame, m_getCBsThisFrame, m_getSRVsThisFrame, m_getUAVsThisFrame);
    if (m_timedFrames)
    {
        static LARGE_INTEGER freq = {};
        if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
        const double ms = 1000.0 / (double)freq.QuadPart, k = ms / m_timedFrames;   // ticks to ms; window to per frame
        const FrameTiming& w = m_timeWindow;
        const LONGLONG right = w.rightConstants + w.rightBinds + w.rightFeatures + w.rightCalls + w.rightTeardown + w.drawDone;
        const LONGLONG ours = w.leftDraws + right + w.maps + w.updates + w.tracking + w.dispatches + w.output;
        DDILog("  [frame %u] time per frame (ms, mean of %u frames): frame %.2f (max %.2f) | real Present %.2f, GPU behind"
               " %.1f frames (max %u) | game outside our calls %.2f, game's own draw/dispatch calls %.2f | ours %.2f: left"
               " draws %.2f (coverage tally %.2f), right eye %.2f (constants %.2f, binds %.2f, features %.2f, draw calls"
               " %.2f, teardown %.2f, after-draw %.2f), maps %.2f (driver Map/Unmap %.2f, snapshot reads %.2f of %.0f KB),"
               " updates %.2f, bind tracking %.2f, dispatches %.2f, output pass %.2f\n", frame, m_timedFrames,
               w.frame * k, m_frameMax * ms, w.present * k, (double)m_gpuBehindSum / m_timedFrames, m_gpuBehindMax,
               (w.frame - ours - w.present - w.gameCalls) * k, w.gameCalls * k, ours * k, w.leftDraws * k,
               w.coverageTally * k, right * k, w.rightConstants * k, w.rightBinds * k, w.rightFeatures * k,
               w.rightCalls * k, w.rightTeardown * k, w.drawDone * k, w.maps * k, w.mapDriver * k, w.mapRead * k,
               w.mapBytes / 1024.0 / m_timedFrames, w.updates * k, w.tracking * k, w.dispatches * k, w.output * k);
        // The driver Maps that waited, by what they mapped: the five costliest kinds.
        struct Kind { LONGLONG ticks; unsigned count; const std::string* key; };
        std::vector<Kind> kinds;
        LONGLONG slowTicks = 0;
        unsigned slowCount = 0;
        for (const auto& s : m_slowMaps)
        {
            kinds.push_back({ s.second.ticks, s.second.count, &s.first });
            slowTicks += s.second.ticks;
            slowCount += s.second.count;
        }
        std::sort(kinds.begin(), kinds.end(), [](const Kind& a, const Kind& b) { return a.ticks > b.ticks; });
        std::string top;
        char one[256];
        for (size_t i = 0; i < kinds.size() && i < 5; ++i)
        {
            sprintf_s(one, " %s x%.1f %.2f ms;", kinds[i].key->c_str(), (double)kinds[i].count / m_timedFrames,
                      kinds[i].ticks * k);
            top += one;
        }
        DDILog("  [frame %u] waiting maps (driver Map over 0.1 ms, per frame, mean of %u frames): %.2f ms in %.1f maps |%s\n",
               frame, m_timedFrames, slowTicks * k, (double)slowCount / m_timedFrames, top.empty() ? " none" : top.c_str());
        // The game's texture creates and frees, and how many right copies were reused rather than allocated.
        if (m_parent)
        {
            unsigned made = 0, freed = 0, reused = 0, fresh = 0, held = 0;
            m_parent->TakeChurn(made, freed, reused, fresh, held);
            const double n = (double)m_timedFrames;
            DDILog("  [frame %u] texture churn (per frame, mean of %u frames): textures made %.1f, freed %.1f | right copies"
                   " reused %.1f, newly allocated %.1f | right copies held for reuse %u\n", frame, m_timedFrames, made / n,
                   freed / n, reused / n, fresh / n, held);
        }
    }
    m_slowMaps.clear();
    if (m_recorder)
    {
        static LARGE_INTEGER freq = {};
        if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
        const double k = m_timedFrames ? 1000.0 / (double)freq.QuadPart / m_timedFrames : 0.0;
        const unsigned* f = m_batchFlushes;
        unsigned replays = 0;
        for (unsigned i = 0; i < kFlushReasons; ++i) replays += f[i];
        DDILog("  [frame %u] right-eye recording (this frame): draws recorded %u, run unrecorded %u | replays %u: pass"
               " ends %u, maps %u, updates %u (textures %u, buffers %u bind 0x%X, held constants %u), copies/clears %u,"
               " dispatches %u, present %u, before an unrecorded draw %u, state %u, queries %u | ms (mean): replaying %.2f"
               " (inside the parts above), state snapshots %.2f | right constant blocks: finding matrices %.2f, writing"
               " %.2f\n", frame, m_batchedDrawsThisFrame, m_unbatchedDrawsThisFrame, replays, f[kFlushTargets],
               f[kFlushMap], f[kFlushUpdate], m_updReplayTex, m_updReplayBuf, m_updReplayBufBinds, m_updReplayCB, f[kFlushWrite],
               f[kFlushCompute], f[kFlushPresent], f[kFlushUnbatched], f[kFlushState], f[kFlushQuery],
               m_timeWindow.rightFlush * k, m_timeWindow.recordBegin * k, m_timeWindow.syncBuild * k,
               m_timeWindow.syncApply * k);
    }
}

// The swap chain's part of this frame, noted just before EndFrame sums the frame.
void Context11Proxy::NoteFrameTiming(LONGLONG outputTicks, LONGLONG presentTicks, unsigned gpuBehind)
{
    m_time.output += outputTicks;
    m_time.present += presentTicks;
    m_gpuBehindSum += gpuBehind;
    if (gpuBehind > m_gpuBehindMax) m_gpuBehindMax = gpuBehind;
}

// The frame just ended joins the report window: its parts and the interval since the previous frame's end. FrameTiming
// holds only LONGLONG parts, summed in order.
void Context11Proxy::AccumulateFrameTiming()
{
    if (!gInfo.CollectDebugInformation) return;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (m_frameStart)
    {
        m_time.frame = now.QuadPart - m_frameStart;
        if (m_time.frame > m_frameMax) m_frameMax = m_time.frame;
        LONGLONG* w = &m_timeWindow.leftDraws;
        const LONGLONG* f = &m_time.leftDraws;
        for (size_t i = 0; i < sizeof(FrameTiming) / sizeof(LONGLONG); ++i) w[i] += f[i];
        ++m_timedFrames;
    }
    m_frameStart = now.QuadPart;
    m_time = {};
}

// CRC of a bound shader, 0 if unknown: what a BaseProfile.xml <VertexShader>/<PixelShader> entry targets.
DWORD BoundShaderCRC(Device11Proxy* parent, void* shader)
{
    if (!parent || !shader) return 0;
    const ShaderAnalysis11Result* info = parent->LookupShaderProjection(shader);
    return info ? info->crc32 : 0;
}

} // namespace wiz3d
