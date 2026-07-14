#include "json_stream.h"
#include "profiler.h"
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

using namespace mcp1122;

namespace {

void usage() {
    std::cerr << "usage: mcp1122_cpp_stream [--trace path] "
                 "[--debug-piston-helper|--debug-piston-move] [input.dat ...]\n";
}

// Builds "<dir>/<stem>-<id><ext>" from a base trace path like "outlog/cpp-update-trace.log",
// so each candidate gets its own file instead of every candidate in a batch scrambling into
// one shared stream.
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

// Process one compact-binary-record-per-candidate stream (see json_stream.h /
// genetic_ml.compact_format for the format). stdout stays a clean one-JSON-per-line stream
// so the genetic-ml worker pool can parse it line by line - only the candidate INPUT side is
// binary, results going back out are still JSON text.
void processStream(std::istream& in, Simulator& simulator, Trace* trace, const std::string& traceBase,
                   bool debugPistonHelper, bool debugPistonMove) {
    Candidate candidate;
    while (true) {
        try {
            if (!readCandidateCompact(in, candidate)) {
                break; // clean EOF between records - done, not an error
            }
        } catch (const std::exception& error) {
            // A truncated record means the stream itself is broken - nothing sensible to
            // recover into for the next iteration, so stop rather than loop on garbage.
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
            if (debugPistonHelper) {
                std::cout << simulator.debugPistonHelper(candidate) << '\n';
            } else if (debugPistonMove) {
                std::cout << simulator.debugPistonMove(candidate) << '\n';
            } else {
                std::cout << simulator.simulate(candidate).toJson() << '\n';
            }
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
    bool debugPistonHelper = false;
    bool debugPistonMove = false;
    std::vector<std::string> inputFiles;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trace" && i + 1 < argc) {
            tracePath = argv[++i];
        } else if (arg == "--debug-piston-helper") {
            debugPistonHelper = true;
        } else if (arg == "--debug-piston-move") {
            debugPistonMove = true;
        } else if (!arg.empty() && arg[0] == '-') {
            usage();
            return 2;
        } else {
            inputFiles.push_back(arg);
        }
    }

    // Trace files are opened per-candidate (see tracePathForCandidate above), named after
    // this base path, so pass the Trace along unconditionally when tracing was requested -
    // the actual file only gets created once a candidate's id is known.
    Trace trace;
    Trace* tracePtr = tracePath.empty() ? nullptr : &trace;

    Simulator simulator(tracePtr);

    if (inputFiles.empty()) {
        // Stream mode: read candidates from stdin (used by the genetic-ml worker pool).
        // Candidates are raw compact-binary bytes, not text, so stdin must not go through the
        // C runtime's default text-mode translation (CRLF rewriting / EOF-byte sniffing on
        // Windows) - that would silently corrupt the stream.
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        processStream(std::cin, simulator, tracePtr, tracePath, debugPistonHelper, debugPistonMove);
    } else {
        // File mode: read each named compact-format file, printing its name so batch runs show
        // which file each block of results came from. The name goes to stderr to keep stdout a
        // clean one-JSON-per-line stream for machine consumers. Opened with ios::binary for the
        // same reason stdin needs _O_BINARY above - a plain ifstream on Windows also defaults
        // to text-mode translation, which would corrupt a binary file just the same.
        for (const std::string& path : inputFiles) {
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                std::cerr << "error: cannot open " << path << '\n';
                continue;
            }
            std::cerr << "========== " << path << " ==========" << '\n';
            processStream(in, simulator, tracePtr, tracePath, debugPistonHelper, debugPistonMove);
        }
    }

    PROF_REPORT();
    return 0;
}
