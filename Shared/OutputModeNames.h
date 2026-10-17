/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*
* Unified output-mode naming for every wiz3D wrapper. One canonical shortname
* per mode (e.g. "HalfSideBySide", "RowInterleaved"), one legacy DLL name
* kept for back-compat with iZ3D-era configs (e.g. "SideBySideOutput",
* "InterlacedOutput"), one numeric id used internally.
*
* The resolver below is pure-header, ASCII-only, zero-dependency so every
* consumer (S3DAPI, AmdQbProxy, NvDirectMode, each wrapper if it needs
* its own direct hook) can call into the same table.
*
* Priority at config read: a recognised string (canonical or legacy)
* overrides the numeric OutputMode. The caller writes the legacy DLL name
* back to its config state so DX9's plugin loader still finds the right
* plugin file even when the user wrote a canonical shortname in config.
* Unknown strings pass through unchanged, so DX9 user-added plugins and
* any mode we haven't yet registered keep working.
*
* The numeric ids are also unified across every modern wrapper (DX7-11,
* OGL, HD3D, 3DV Direct). HD3D and NvDirectMode used slightly different
* internal numbering in earlier builds (Anaglyph=7, SR_WEAVE=8 vs the
* OGL-origin numbering adopted here); the switch to this header requires
* those wrappers to renumber their enums to match. See wiz3D's 0.4.0
* release notes for the migration summary.
*/

#pragma once

#include <stddef.h>

namespace wiz3d {

enum OutputModeId
{
    OUTMODE_UNKNOWN              = -1,
    OUTMODE_IZ3D_DUAL            = 0,   // iZ3D dual-panel hardware (legacy)
    OUTMODE_HALF_SIDE_BY_SIDE    = 1,   // default
    OUTMODE_FULL_SIDE_BY_SIDE    = 2,   // game renders at half display width
    OUTMODE_HALF_TOP_AND_BOTTOM  = 3,
    OUTMODE_FULL_TOP_AND_BOTTOM  = 4,   // game renders at half display height
    OUTMODE_ANAGLYPH             = 5,   // Dubois Red/Cyan
    OUTMODE_ROW_INTERLEAVED      = 6,   // horizontal rows (passive 3D TVs)
    OUTMODE_COLUMN_INTERLEAVED   = 7,
    OUTMODE_CHECKERBOARD         = 8,   // DLP-Link projectors
    OUTMODE_SR_WEAVE             = 9,   // Simulated Reality (Leia/Samsung Odyssey 3D)
    OUTMODE_STEREO_MIRROR        = 10,  // second-display mirror
};

struct OutputModeEntry
{
    int         mode;             // one of the OUTMODE_* ids
    const char* canonicalName;    // user-facing shortname
    const char* legacyDll;        // iZ3D-era OutputMethodDll value for back-compat
};

// Canonical names come first; legacy DLL names are the exact filenames the
// DX9 plugin loader looks up at runtime. If a mode's legacy slot is a new
// name (never had a DX9 plugin DLL), that string is reserved for future
// use. Case-insensitive matching, so configs can mix capitalisations.
static const OutputModeEntry kOutputModeTable[] =
{
    { OUTMODE_HALF_SIDE_BY_SIDE,    "HalfSideBySide",         "SideBySideOutput"            },
    { OUTMODE_FULL_SIDE_BY_SIDE,    "FullSideBySide",         "FullSideBySideOutput"        },
    { OUTMODE_HALF_TOP_AND_BOTTOM,  "HalfTopAndBottom",       "TopAndBottomOutput"          },
    { OUTMODE_FULL_TOP_AND_BOTTOM,  "FullTopAndBottom",       "FullTopAndBottomOutput"      },
    { OUTMODE_ANAGLYPH,             "Anaglyph",               "AnaglyphOutput"              },
    { OUTMODE_ROW_INTERLEAVED,      "RowInterleaved",         "InterlacedOutput"            },
    { OUTMODE_COLUMN_INTERLEAVED,   "ColumnInterleaved",      "ColumnInterleavedOutput"     },
    { OUTMODE_CHECKERBOARD,         "Checkerboard",           "CheckerboardOutput"          },
    { OUTMODE_SR_WEAVE,             "SimulatedRealityWeave",  "SimulatedRealityWeaveOutput" },
    { OUTMODE_STEREO_MIRROR,        "StereoMirror",           "StereoMirrorOutput"          },
    { OUTMODE_IZ3D_DUAL,            "iZ3DDualPanel",          "S3DOutput"                   },
};

// Hardware-specific legacy plugin DLL names, kept recognised so the resolver
// doesn't mangle them even though we don't assign them canonical shortnames
// or inline-wrapper support. Users with the matching hardware keep using
// their existing DX9 plugin configs; new wrappers see "unknown" and leave
// the state alone, so the DX9 plugin loader still finds the DLL.
static const char* const kLegacyHardwarePlugins[] =
{
    "CMOShutterOutput",  "iZShutterOutput",  "ShutterOutput",     "S3DShutter",
    "ZalmanOutput",      "VR920Output",      "Z800Output",        "LenovoOutput",
    "DualProjectionOutput", "TaerimOutput",  "AvitridOutput",
    "S3D120HzProjectorsOutput", "3DDLPOutput", "ATI3DDLPOutput",
    "AMDHD3DOutput",     "S3DMarkedOutput",
};

// Case-insensitive ASCII C-string compare (names in config are ASCII).
inline int OutputModes_StrICmp(const char* a, const char* b)
{
    while (*a && *b)
    {
        char ca = *a, cb = *b;
        if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
        if (ca != cb) return (int)(unsigned char)ca - (int)(unsigned char)cb;
        ++a; ++b;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

// Returns an OUTMODE_* id if `name` matches a canonical shortname OR a
// legacy DLL name in the table. Returns OUTMODE_UNKNOWN for null, empty,
// unrecognised, or hardware-plugin names (callers check IsLegacyHardwarePlugin
// if they want to distinguish those).
inline int ResolveOutputModeName(const char* name)
{
    if (!name || !*name) return OUTMODE_UNKNOWN;
    for (size_t i = 0; i < sizeof(kOutputModeTable) / sizeof(kOutputModeTable[0]); ++i)
    {
        const OutputModeEntry& e = kOutputModeTable[i];
        if (OutputModes_StrICmp(name, e.canonicalName) == 0 ||
            OutputModes_StrICmp(name, e.legacyDll)     == 0)
            return e.mode;
    }
    return OUTMODE_UNKNOWN;
}

// Returns the legacy DLL filename stem (no ".dll" suffix) for an OUTMODE_*
// id, or nullptr for an unregistered id. Used by DX9's plugin loader so a
// user who wrote a canonical shortname in config still gets the right
// plugin DLL loaded.
inline const char* OutputModeToLegacyDll(int mode)
{
    for (size_t i = 0; i < sizeof(kOutputModeTable) / sizeof(kOutputModeTable[0]); ++i)
        if (kOutputModeTable[i].mode == mode) return kOutputModeTable[i].legacyDll;
    return nullptr;
}

// Returns the canonical shortname for an OUTMODE_* id, or nullptr.
inline const char* OutputModeToCanonical(int mode)
{
    for (size_t i = 0; i < sizeof(kOutputModeTable) / sizeof(kOutputModeTable[0]); ++i)
        if (kOutputModeTable[i].mode == mode) return kOutputModeTable[i].canonicalName;
    return nullptr;
}

// Returns true if `name` matches a recognised legacy hardware plugin DLL
// name (CMOShutter, Zalman, VR920, etc.). These pass through the resolver
// unchanged so the DX9 plugin pipeline still loads them.
inline bool IsLegacyHardwarePlugin(const char* name)
{
    if (!name || !*name) return false;
    for (size_t i = 0; i < sizeof(kLegacyHardwarePlugins) / sizeof(kLegacyHardwarePlugins[0]); ++i)
        if (OutputModes_StrICmp(name, kLegacyHardwarePlugins[i]) == 0)
            return true;
    return false;
}

}  // namespace wiz3d
