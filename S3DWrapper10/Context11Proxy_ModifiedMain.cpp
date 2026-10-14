/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Frame and state boundaries, as DX9's BaseStereoRenderer_ModifiedMain.cpp (Present, Reset): the frame's end at Present,
// ClearState, context-state swaps and command lists.

#include "StdAfx.h"
#include "Context11Proxy-inl.h"

namespace wiz3d
{

void Context11Proxy::EndFrame()
{
    FlushRightBatch(kFlushPresent);
    AccumulateFrameTiming();
    ClearBlockTargets();
    // The detector ages its cached view-projections by this, so a camera that has moved on no longer confirms a target.
    s_mat.frameId = ++s_frameId;
    ServiceRingReadbacks();
    static unsigned s_drawnFrames = 0;
    const bool report = gInfo.CollectDebugInformation && m_drawsThisFrame &&
                        (++s_drawnFrames <= 3 || s_drawnFrames % 60 == 0);
    if (m_gpuTimer)
    {
        m_gpuTimer->FrameEnd(m_real);
        if (report) m_gpuTimer->Report(m_real, s_frameId);
        m_gpuTimer->FrameBegin(m_real);
    }
    s_sceneP00Votes = s_cleanP00Total = 0;
    for (const auto& v : s_cleanP00Votes)
    {
        s_cleanP00Total += v.second.first;
        if (v.second.first > s_sceneP00Votes) { s_sceneP00Votes = v.second.first; s_sceneP00 = v.second.second; }
    }
    s_cleanP00Votes.clear();
    for (auto& f : Features()) f->OnFrameEnd(*this, s_frameId, report);
    if (report)
    {
        LogFrameStats(s_frameId);
        m_timeWindow = {};
        m_frameMax = 0;
        m_timedFrames = m_gpuBehindSum = m_gpuBehindMax = 0;
    }
    ReleaseUavTwins(600);

    m_flippedWritesThisFrame = 0;
    m_leftSRVsThisFrame = 0;
    m_sharedUavDispatchesThisFrame = m_sharedUavSlotsThisFrame = 0;
    m_sharedUavByCS.clear();
    m_twinCopiesThisFrame = m_twinFailedThisFrame = m_twinCounterUnknownThisFrame = m_twinWriteOnlyThisFrame = 0;
    m_twinBytesThisFrame = 0;
    s_scaledByCamera = s_lightOnlyKept = 0;
    s_mat.ringInsertsThisFrame = s_mat.evictedHotThisFrame = 0;
    s_mat.invStaleThisFrame = s_mat.invStaleShapeThisFrame = 0;
    m_drawsThisFrame = m_dispatchesThisFrame = m_cmdListsThisFrame = 0;
    m_cbTargetedThisFrame = m_cbUntargetedThisFrame = m_cbMatShiftedThisFrame = m_cbMatRejectedThisFrame = 0;
    m_invScansThisFrame = m_invHitsThisFrame = m_invPatchedThisFrame = m_invLearnedThisFrame = 0;
    m_updCBWritesThisFrame = m_updCBPatchedThisFrame = 0;
    m_invBestMissThisFrame = 1e30f;
    m_invDrawChecksThisFrame = m_invDrawPatchedThisFrame = m_fwdDrawPatchedThisFrame = 0;
    m_invDrawByPS.clear();
    m_overlayDrawsThisFrame = m_overlayRestoredThisFrame = m_plainRestoredThisFrame = m_learnedWithheldThisFrame = 0;
    m_dsBuffersPatchedThisFrame = 0;
    for (unsigned& w : m_untargetedWhy) w = 0;
    m_bindResetsThisFrame = 0;
    m_getCBsThisFrame = m_getSRVsThisFrame = m_getUAVsThisFrame = 0;
    m_drawsVSUnparsedThisFrame = m_drawsVSNoMatrixThisFrame = m_drawsVSNeverShiftedThisFrame = 0;
    m_drawsVSStaleShiftThisFrame = m_drawsVSNoSiblingCBThisFrame = 0;
    m_uncorrectedVSDraws.clear();
    m_drawsDuplicatedThisFrame = m_drawsMonoThisFrame = m_drawsUavSkippedThisFrame = m_drawsUavDupThisFrame = 0;
    m_drawsMonoDsvThisFrame = m_drawsNoRightRTVThisFrame = m_drawsDepthOnlyThisFrame = m_drawsNoRightTargetThisFrame = 0;
    m_dispatchesDuplicatedThisFrame = 0;
    m_batchedDrawsThisFrame = m_unbatchedDrawsThisFrame = 0;
    for (unsigned& n : m_batchFlushes) n = 0;
    m_updReplayTex = m_updReplayBuf = m_updReplayCB = m_updReplayBufBinds = 0;
}

void Context11Proxy::ReplayFrameCommands(Eye eye)
{
    // Snapshot + flip the active eye for the replay pass. Each recorded
    // closure re-enters our proxy methods, so OMSet/etc. pick the
    // eye-appropriate real handle via m_activeEye automatically.
    Eye saved = m_activeEye;
    m_activeEye = eye;
    if (FrameTraceActive())
        FrameTrace("--- replay begin (eye=%c, %zu commands) ---\n",
                   eye == Eye::Right ? 'R' : 'L', m_frameCommands.size());
    for (auto& fn : m_frameCommands)
        fn();
    if (FrameTraceActive())
        FrameTrace("--- replay end (eye=%c) ---\n",
                   eye == Eye::Right ? 'R' : 'L');
    m_activeEye = saved;
}

void Context11Proxy::SetPresentHookActive(bool active)
{
    // Draw duplication issues both eyes live, so recording would be overhead
    // and replaying it would double every draw a second time.
    m_presentHookActive = active && !gInfo.DuplicateDraws;
}

// ClearState resets every pipeline slot to defaults (shaders, SRVs, RTs,
// viewports, blend/depth/raster state, etc). Without recording, the right-eye
// replay sees subsequent state-setters applied on top of whatever state was
// in effect from the previous frame's replay tail — so eye-aware bindings
// can leak across the ClearState boundary. Closure has no captures; the real
// runtime handles the reset itself when replayed.
void STDMETHODCALLTYPE Context11Proxy::ClearState()
{
    // BatchRightEye: recorded in order like any other state call, so nothing replays here.
    m_real->ClearState();
    MIRROR(ClearState());
    m_soBound = false;
    m_gameVpCount = m_gameScCount = 0;
    ResetEyeTracking();
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this]() { m_real->ClearState(); });
}

// The swapped-in state's bindings were never tracked; start from none rather than from the swapped-out state's.
void STDMETHODCALLTYPE Context11Proxy::SwapDeviceContextState(ID3DDeviceContextState* pState, ID3DDeviceContextState** ppPreviousState)
{
    if (!m_real1) return;
    FlushRightBatch(kFlushState);
    m_real1->SwapDeviceContextState(pState, ppPreviousState);
    ResetEyeTracking();
}

// ExecuteCommandList submits a previously-built command list (typically from
// a deferred context). Without recording, deferred-context games (some
// DA2-era engines) submit entire batches of state + draws via this single
// call and the right-eye replay never sees them. Command lists are
// idempotent at the API level; replaying the same call is safe.
void STDMETHODCALLTYPE Context11Proxy::ExecuteCommandList(
    ID3D11CommandList* pCommandList, BOOL RestoreContextState)
{
    // A list from an unwrapped deferred context baked left-eye resources and runs once; a paired one (DeferredContext11)
    // runs its left list, then its right half.
    FlushRightBatch(kFlushState);
    ++m_cmdListsThisFrame;
    ExecuteCommandListPair(m_real, pCommandList, RestoreContextState);
    // The buffers its recording wrote no longer hold their last written bytes, as after a copy.
    for (Buffer11Proxy* db : CommandListWrittenBuffers(pCommandList))
    {
        db->SetRightFromShadow(false);
        db->ForgetShadow();
        if (db->IsRing()) NoteRingCopy(db, 0, db->RingSize());
    }
    // Not restoring leaves the context in its default state, as ClearState does.
    if (!RestoreContextState) ResetEyeTracking();
    if (!m_presentHookActive) return;
    ComRefHolder cmdRef(pCommandList);
    m_frameCommands.emplace_back(
        [this, cmdRef, RestoreContextState]()
        {
            ExecuteCommandListRight(m_real, static_cast<ID3D11CommandList*>(cmdRef.p), RestoreContextState);
        });
}

} // namespace wiz3d
