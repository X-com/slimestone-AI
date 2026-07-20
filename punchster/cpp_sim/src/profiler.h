#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

// Lightweight opt-in profiler. Enable with -DMCP_PROFILE at compile time.
// Reports to stderr at program exit via Profiler::printReport().
// Thread-unsafe: single-threaded use only (fine for this simulator).

#ifndef MCP_PROFILE

#define PROF_SCOPE(name)   do {} while(0)
#define PROF_REPORT()      do {} while(0)

#else

#define PROF_SCOPE(name)  mcp1122::ProfScope _ps_##__LINE__(mcp1122::profilerSlot(name))
#define PROF_REPORT()     mcp1122::Profiler::printReport()

namespace mcp1122 {

struct ProfSlot {
    const char* name = nullptr;
    std::uint64_t totalNs = 0;
    std::uint64_t calls   = 0;
};

// Up to 64 named slots; looked up by pointer identity of string literal.
ProfSlot* profilerSlot(const char* name);

struct ProfScope {
    ProfSlot* slot;
    std::chrono::steady_clock::time_point t0;
    explicit ProfScope(ProfSlot* s)
        : slot(s), t0(std::chrono::steady_clock::now()) {}
    ~ProfScope() {
        auto dt = std::chrono::steady_clock::now() - t0;
        slot->totalNs += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
        ++slot->calls;
    }
};

struct Profiler {
    static void printReport();
};

} // namespace mcp1122

#endif // MCP_PROFILE
