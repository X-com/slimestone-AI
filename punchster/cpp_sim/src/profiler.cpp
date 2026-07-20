#include "profiler.h"

#ifdef MCP_PROFILE

#include <algorithm>
#include <cstdio>

namespace mcp1122 {

namespace {

constexpr int MAX_SLOTS = 64;
ProfSlot slots[MAX_SLOTS];
int slotCount = 0;

} // namespace

ProfSlot* profilerSlot(const char* name) {
    // Linear scan by pointer — string literals have stable addresses.
    for (int i = 0; i < slotCount; ++i) {
        if (slots[i].name == name) return &slots[i];
    }
    if (slotCount < MAX_SLOTS) {
        slots[slotCount].name = name;
        return &slots[slotCount++];
    }
    return &slots[0]; // overflow: bucket into slot 0
}

void Profiler::printReport() {
    // Sort by total time descending
    std::sort(slots, slots + slotCount,
        [](const ProfSlot& a, const ProfSlot& b) { return a.totalNs > b.totalNs; });

    std::uint64_t grandTotal = 0;
    for (int i = 0; i < slotCount; ++i) grandTotal += slots[i].totalNs;

    std::fprintf(stderr, "\n=== MCP PROFILER REPORT ===\n");
    std::fprintf(stderr, "%-38s %8s %12s %10s %10s\n",
                 "Function", "calls", "total(ms)", "avg(us)", "% time");
    std::fprintf(stderr, "%-38s %8s %12s %10s %10s\n",
                 "--------", "-----", "---------", "-------", "------");
    for (int i = 0; i < slotCount; ++i) {
        const ProfSlot& s = slots[i];
        if (s.calls == 0) continue;
        double pct    = grandTotal ? 100.0 * s.totalNs / grandTotal : 0.0;
        double totMs  = s.totalNs / 1e6;
        double avgUs  = s.calls   ? s.totalNs / 1e3 / s.calls : 0.0;
        std::fprintf(stderr, "%-38s %8llu %12.1f %10.3f %9.1f%%\n",
                     s.name,
                     static_cast<unsigned long long>(s.calls),
                     totMs, avgUs, pct);
    }
    std::fprintf(stderr, "===========================\n");
}

} // namespace mcp1122

#endif // MCP_PROFILE
