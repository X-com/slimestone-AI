#include "json_stream.h"  // mcp1122::readCandidateCompact, mcp1122::Candidate - reused from cpp extract
#include "simulator.h"
#include "trace.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <istream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace mcp1122gpu;

namespace {

void usage() {
    std::cerr << "usage: gpu_simulator_stream [--trace path] [input.dat ...]\n";
}

// Same per-candidate trace file naming as cpp extract's main.cpp: "<dir>/<stem>-<id><ext>",
// so every candidate gets its own trace file instead of a batch scrambling into one stream.
std::string tracePathForCandidate(const std::string& base, std::int64_t id) {
    std::string dir;
    std::string name = base;
    std::size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) {
        dir = base.substr(0, slash + 1);
        name = base.substr(slash + 1);
    }
    std::string stem = name;
    std::string ext;
    std::size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) {
        stem = name.substr(0, dot);
        ext = name.substr(dot);
    }
    return dir + stem + "-" + std::to_string(id) + ext;
}

void processStream(std::istream& in, Simulator& simulator, Trace* trace, const std::string& traceBase) {
    mcp1122::Candidate candidate;
    while (true) {
        try {
            if (!mcp1122::readCandidateCompact(in, candidate)) {
                break; // clean EOF between records - done, not an error
            }
        } catch (const std::exception& error) {
            Result result;
            result.id = -1;
            result.ok = false;
            result.errorCode = "parse_error";
            result.error = error.what();
            std::cout << result.toJson() << '\n';
            break;
        }

        try {
            if (trace != nullptr && !traceBase.empty()) {
                trace->open(tracePathForCandidate(traceBase, candidate.id));
            }
            std::cout << simulator.simulate(candidate).toJson() << '\n';
        } catch (const std::exception& error) {
            Result result;
            result.id = candidate.id;
            result.ok = false;
            result.errorCode = "simulate_error";
            result.error = error.what();
            std::cout << result.toJson() << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string tracePath;
    std::vector<std::string> inputFiles;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trace" && i + 1 < argc) {
            tracePath = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
            return 2;
        } else {
            inputFiles.push_back(arg);
        }
    }

    Trace trace;
    Trace* tracePtr = tracePath.empty() ? nullptr : &trace;

    Simulator simulator(tracePtr);

    if (inputFiles.empty()) {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        processStream(std::cin, simulator, tracePtr, tracePath);
    } else {
        for (const std::string& path : inputFiles) {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                std::cerr << "error: cannot open " << path << '\n';
                continue;
            }
            std::cerr << "========== " << path << " ==========" << '\n';
            processStream(in, simulator, tracePtr, tracePath);
        }
    }

    return 0;
}
