// Host driver for the CUDA kernel (Milestone H). Same compact-binary stdin protocol and JSON
// output shape as ../reference/main.cpp, so this diffs directly against reference/'s output
// with the same tooling used to diff reference/ against cpp extract in ../../VERIFICATION.md -
// see gpu_kernel.cu's runBatch() for how a whole file's worth of candidates becomes one batched
// kernel launch instead of one candidate at a time.

#include "json_stream.h"  // mcp1122::readCandidateCompact, mcp1122::Candidate, mcp1122::quoteJson
#include "launch.cuh"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace mcp1122gpu;

namespace {

std::string resultToJson(const GpuResult& r) {
    std::ostringstream out;
    out << '{'
        << "\"id\":" << r.id
        << ",\"ok\":" << (r.ok ? "true" : "false")
        << ",\"working\":" << (r.working ? "true" : "false")
        << ",\"ticks\":" << r.ticks
        << ",\"start\":" << r.start
        << ",\"end\":" << r.end
        << ",\"period\":" << r.period
        << ",\"shift\":{\"x\":" << r.shift.x << ",\"y\":" << r.shift.y << ",\"z\":" << r.shift.z << '}'
        << ",\"cycles\":" << (r.cycles ? "true" : "false")
        << ",\"settled\":" << (r.settled ? "true" : "false")
        << ",\"validCycle\":" << (r.validCycle ? "true" : "false")
        << ",\"finalShift\":{\"x\":" << r.finalShift.x << ",\"y\":" << r.finalShift.y << ",\"z\":" << r.finalShift.z << '}'
        << ",\"elapsedNs\":" << 0
        << ",\"ticksPerSecond\":" << 0;
    if (!r.ok) {
        // Matches reference/'s single catch-all "simulation_error" - both the bounds check in
        // loadCandidate and any FixedWorld capacity ceiling report through the same code there
        // (one try/catch spanning loadCandidate+trigger+detectShiftCycle); this mirrors that.
        out << ",\"errorCode\":\"simulation_error\",\"error\":\"\"";
    }
    out << '}';
    return out.str();
}

HostCandidate toHostCandidate(const mcp1122::Candidate& c) {
    HostCandidate out;
    out.id = c.id;
    out.trigger = BlockPos{c.trigger.x, c.trigger.y, c.trigger.z};
    out.blocks.reserve(c.blocks.size());
    for (const mcp1122::BlockEntry& b : c.blocks) {
        out.blocks.push_back(GpuBlockEntry{b.x, b.y, b.z, b.state});
    }
    return out;
}

void processStream(std::istream& in, int maxTicks, int batchSize, int threadsPerBlock) {
    std::vector<HostCandidate> batch;
    mcp1122::Candidate candidate;

    auto flush = [&]() {
        if (batch.empty()) return;
        std::vector<GpuResult> results = runBatch(batch, maxTicks, threadsPerBlock);
        for (const GpuResult& r : results) std::cout << resultToJson(r) << '\n';
        batch.clear();
    };

    while (true) {
        try {
            if (!mcp1122::readCandidateCompact(in, candidate)) break;
        } catch (const std::exception& error) {
            flush();
            std::cout << "{\"id\":-1,\"ok\":false,\"working\":false,\"ticks\":0,\"start\":0,\"end\":0,"
                         "\"period\":0,\"shift\":{\"x\":0,\"y\":0,\"z\":0},\"cycles\":false,"
                         "\"settled\":false,\"validCycle\":false,\"finalShift\":{\"x\":0,\"y\":0,\"z\":0},"
                         "\"elapsedNs\":0,\"ticksPerSecond\":0,"
                         "\"errorCode\":\"parse_error\",\"error\":\""
                      << mcp1122::quoteJson(error.what()) << "\"}\n";
            break;
        }
        batch.push_back(toHostCandidate(candidate));
        if (static_cast<int>(batch.size()) >= batchSize) flush();
    }
    flush();
}

void usage() {
    std::cerr << "usage: gpu_kernel_stream [--max-ticks N] [--batch-size N] [--threads-per-block N] "
                 "[input.dat ...]\n";
}

} // namespace

int main(int argc, char** argv) {
    int maxTicks = 6000;
    // Milestone I (persistent-kernel/work-queue dispatch, see gpu_kernel.cu) decoupled device
    // memory from batch size - a small, memory-sized worker pool now processes an arbitrarily
    // large batch by looping. batchSize is only about bounding a single kernel launch's
    // wall-clock time under Windows' ~2s WDDM timeout (TDR), not about memory anymore; bump with
    // --batch-size if a run's candidates are all fast and hitting it would just waste launch
    // overhead, or lower it if a batch contains slow (near-6000-tick) candidates.
    int batchSize = 256;
    // Verified stable on this project's own dev card (2GB, memory-constrained) - see
    // gpu_kernel.cu's runBatch() for why this is a CLI flag rather than a compile-time constant:
    // the right value is GPU-dependent, and a value that's merely "not obviously wrong" isn't
    // enough to trust here (128 crashed a few batches into a full run on that card, not at
    // launch). Users with more VRAM headroom can try raising this, but should re-verify against
    // a full run's output (diffed against reference/'s CPU build), not just a quick smoke test.
    int threadsPerBlock = 64;
    std::vector<std::string> inputFiles;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-ticks" && i + 1 < argc) {
            maxTicks = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batchSize = std::atoi(argv[++i]);
        } else if (arg == "--threads-per-block" && i + 1 < argc) {
            threadsPerBlock = std::atoi(argv[++i]);
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
            return 2;
        } else {
            inputFiles.push_back(arg);
        }
    }

    initBlockRegistryDevice();

    if (inputFiles.empty()) {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        processStream(std::cin, maxTicks, batchSize, threadsPerBlock);
    } else {
        for (const std::string& path : inputFiles) {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                std::cerr << "error: cannot open " << path << '\n';
                continue;
            }
            std::cerr << "========== " << path << " ==========" << '\n';
            processStream(in, maxTicks, batchSize, threadsPerBlock);
        }
    }

    return 0;
}
