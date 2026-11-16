#include "breeze.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace breeze {
namespace {
constexpr const char* DEFAULT_EVAL_FILE = "nn-68752b755c93.nnue";

std::string resolveEvalPath(const std::string& filename) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(filename, ec)) return filename;

    fs::path exeDir;
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) exeDir = fs::path(buf).parent_path();
#elif defined(__linux__)
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        exeDir = fs::path(buf).parent_path();
    }
#endif
    if (!exeDir.empty()) {
        fs::path candidate = exeDir / filename;
        if (fs::exists(candidate, ec)) return candidate.string();
    }
    return filename;
}

bool isGoKeyword(const std::string& s) {
    static const char* keys[] = {"depth", "nodes", "movetime", "wtime", "btime", "winc", "binc",
                                 "movestogo", "infinite", "ponder", "mate", "perft"};
    for (const char* k : keys) if (s == k) return true;
    return false;
}

std::vector<std::string> splitWords(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> words;
    std::string word;
    while (in >> word) words.push_back(word);
    return words;
}
}

UCI::UCI() {
    board.setStartPosition();
    std::string report;
    const std::string path = resolveEvalPath(DEFAULT_EVAL_FILE);
    useNNUE = network.load(path, report);
    if (useNNUE) {
        print("info string Loaded " + path + " (SFNNv20)");
    } else {
        print("info string Warning: " + report + " (using HCE fallback)");
    }
}

UCI::~UCI() { stopSearch(); }

void UCI::print(const std::string& s) {
    std::lock_guard<std::mutex> lock(outputMutex);
    std::cout << s << std::endl;
}

void UCI::stopSearch() {
    stopFlag.store(true, std::memory_order_relaxed);
    if (searchThread.joinable()) searchThread.join();
    stopFlag.store(false, std::memory_order_relaxed);
}

int UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream in(line);
        std::string command;
        in >> command;
        if (command == "quit") {
            stopSearch();
            break;
        }
        handleCommand(line);
    }
    stopSearch();
    return 0;
}

void UCI::handleCommand(const std::string& line) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;

    if (cmd == "uci") {
        print("id name Breeze 5.1.0");
        print("id author Oliver");
        print("option name Hash type spin default 64 min 1 max 4096");
        print("option name Threads type spin default 1 min 1 max 64");
        print("option name Move Overhead type spin default 15 min 0 max 5000");
        print("option name Use NNUE type check default true");
        print(std::string("option name EvalFile type string default ") + DEFAULT_EVAL_FILE);
        print("option name Clear Hash type button");
        print("option name Ponder type check default true");
        print("uciok");
    } else if (cmd == "isready") {
        print("readyok");
    } else if (cmd == "setoption") {
        stopSearch();
        setOption(line.substr(line.find(cmd) + cmd.size()));
    } else if (cmd == "ucinewgame") {
        stopSearch();
        board.setStartPosition();
        clearSearchHash();
    } else if (cmd == "position") {
        stopSearch();
        std::istringstream p(line.substr(line.find(cmd) + cmd.size()));
        std::string token;
        p >> token;
        Board candidate = board;
        bool valid = true;
        bool hasMoves = false;
        if (token == "startpos") {
            candidate.setStartPosition();
            p >> token;
            hasMoves = (token == "moves");
        } else if (token == "fen") {
            std::string fen, part;
            while (p >> part) {
                if (part == "moves") { hasMoves = true; break; }
                if (!fen.empty()) fen += ' ';
                fen += part;
            }
            std::string error;
            if (!candidate.setFEN(fen, &error)) {
                print("info string rejected FEN: " + error);
                valid = false;
            }
        } else {
            print("info string position expects startpos or fen");
            valid = false;
        }
        if (valid && hasMoves) {
            std::string moveText;
            while (p >> moveText) {
                Move move;
                if (!candidate.parseUciMove(moveText, move)) {
                    print("info string rejected move in position command: " + moveText);
                    valid = false;
                    break;
                }
                Undo undo;
                if (!candidate.makeMove(move, undo)) { valid = false; break; }
            }
        }
        if (valid) board = candidate;
    } else if (cmd == "go") {
        handleGo(line.substr(line.find(cmd) + cmd.size()));
    } else if (cmd == "stop") {
        stopSearch();
    } else if (cmd == "ponderhit") {
        ponderHitFlag.store(true, std::memory_order_relaxed);
    } else if (cmd == "debug") {
        // UCI debug mode is accepted; diagnostics remain on the protocol info channel.
    } else if (cmd == "d") {
        print("info string FEN: " + board.fen());
    } else if (cmd == "eval") {
        const bool activeNNUE = useNNUE && network.loaded();
        const int score = activeNNUE ? network.evaluate(board) : evaluate(board);
        print(std::string("info string Breeze ") + (activeNNUE ? "SFNNv20" : "HCE") + " evaluation: " + std::to_string(score) + " cp");
    } else if (cmd == "perft") {
        stopSearch();
        int depth = 0;
        in >> depth;
        if (depth < 0 || depth > 10) { print("info string perft depth must be between 0 and 10"); return; }
        const auto start = std::chrono::steady_clock::now();
        const uint64_t nodes = perft(board, depth);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        print("info string perft depth " + std::to_string(depth) + " nodes " + std::to_string(nodes) + " time " + std::to_string(ms) + " ms");
    } else if (cmd == "help") {
        print("info string UCI commands: uci isready ucinewgame setoption position go stop ponderhit quit eval d perft");
    }
}

void UCI::setOption(const std::string& rest) {
    std::string s = rest;
    const auto namePos = s.find("name ");
    if (namePos == std::string::npos) return;
    s = s.substr(namePos + 5);
    const auto valuePos = s.find(" value ");
    std::string name = valuePos == std::string::npos ? s : s.substr(0, valuePos);
    std::string value = valuePos == std::string::npos ? "" : s.substr(valuePos + 7);
    auto trim = [](std::string& x) {
        while (!x.empty() && std::isspace(static_cast<unsigned char>(x.front()))) x.erase(x.begin());
        while (!x.empty() && std::isspace(static_cast<unsigned char>(x.back()))) x.pop_back();
    };
    trim(name); trim(value);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        value = value.substr(1, value.size() - 2);

    if (name == "Hash") {
        try { hashMb = std::clamp(std::stoi(value), 1, 4096); clearSearchHash(); }
        catch (...) { print("info string invalid Hash value"); }
    } else if (name == "Threads") {
        try { threads = std::clamp(std::stoi(value), 1, 64); }
        catch (...) { print("info string invalid Threads value"); }
    } else if (name == "Move Overhead") {
        try { moveOverhead = std::clamp(std::stoi(value), 0, 5000); }
        catch (...) { print("info string invalid Move Overhead value"); }
    } else if (name == "Use NNUE") {
        const bool requested = (value == "true" || value == "1");
        if (requested && !network.loaded()) {
            std::string report;
            const std::string path = resolveEvalPath(DEFAULT_EVAL_FILE);
            if (network.load(path, report)) {
                clearSearchHash();
                useNNUE = true;
                print("info string evaluation mode: SFNNv20");
            } else {
                useNNUE = false;
                print("info string cannot enable NNUE: load a compatible SFNNv20 file with EvalFile first");
            }
        } else {
            if (useNNUE != requested) clearSearchHash();
            useNNUE = requested;
            print(std::string("info string evaluation mode: ") + (useNNUE ? "SFNNv20" : "HCE"));
        }
    } else if (name == "EvalFile") {
        if (value.empty() || value == "<empty>") {
            network.clear();
            useNNUE = false;
            clearSearchHash();
            print("info string EvalFile cleared; HCE fallback selected");
        } else {
            std::string report;
            const std::string path = resolveEvalPath(value);
            const bool ok = network.load(path, report);
            std::istringstream r(report);
            std::string reportLine;
            while (std::getline(r, reportLine)) print("info string " + reportLine);
            clearSearchHash();
            if (!ok) {
                useNNUE = false;
                print("info string network load failed; HCE fallback selected");
            } else {
                useNNUE = true;
                print("info string evaluation mode: SFNNv20");
            }
        }
    } else if (name == "Clear Hash") {
        clearSearchHash();
    } else if (name == "Ponder") {
        ponderEnabled = (value == "true" || value == "1");
    }
}

void UCI::handleGo(const std::string& rest) {
    stopSearch();
    SearchLimits limits;
    const std::vector<std::string> tokens = splitWords(rest);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        auto parseInt = [&](int& dest) {
            if (i + 1 < tokens.size()) {
                try { dest = std::stoi(tokens[++i]); } catch (...) { dest = 0; }
            }
        };
        if (t == "depth") parseInt(limits.depth);
        else if (t == "nodes") {
            if (i + 1 < tokens.size()) {
                try { limits.nodes = std::stoull(tokens[++i]); } catch (...) { limits.nodes = 0; }
            }
        } else if (t == "movetime") parseInt(limits.movetimeMs);
        else if (t == "wtime") parseInt(limits.wtimeMs);
        else if (t == "btime") parseInt(limits.btimeMs);
        else if (t == "winc") parseInt(limits.wincMs);
        else if (t == "binc") parseInt(limits.bincMs);
        else if (t == "movestogo") parseInt(limits.movesToGo);
        else if (t == "mate") {
            int mateMoves = 0;
            parseInt(mateMoves);
            limits.depth = std::max(limits.depth, std::clamp(mateMoves * 2, 1, MAX_PLY - 4));
        } else if (t == "infinite") limits.infinite = true;
        else if (t == "ponder") limits.ponder = ponderEnabled;
        else if (t == "searchmoves") {
            limits.searchMovesSpecified = true;
            while (i + 1 < tokens.size() && !isGoKeyword(tokens[i + 1])) {
                Move move;
                if (board.parseUciMove(tokens[i + 1], move)) limits.searchMoves.push_back(move);
                else print("info string ignored invalid searchmove " + tokens[i + 1]);
                ++i;
            }
        } else if (t == "perft") {
            int depth = 0;
            parseInt(depth);
            if (depth < 0 || depth > 10) { print("info string go perft depth must be between 0 and 10"); return; }
            const auto start = std::chrono::steady_clock::now();
            const uint64_t nodes = perft(board, depth);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            print("info string perft depth " + std::to_string(depth) + " nodes " + std::to_string(nodes) + " time " + std::to_string(ms) + " ms");
            return;
        }
    }
    limits.depth = limits.depth > 0 ? std::clamp(limits.depth, 1, MAX_PLY - 4) : 0;
    if (limits.movetimeMs > 0) limits.movetimeMs = std::max(1, limits.movetimeMs - moveOverhead);
    if (limits.wtimeMs >= 0) limits.wtimeMs = std::max(0, limits.wtimeMs - moveOverhead);
    if (limits.btimeMs >= 0) limits.btimeMs = std::max(0, limits.btimeMs - moveOverhead);
    ponderHitFlag.store(false, std::memory_order_relaxed);
    limits.ponderHitFlag = &ponderHitFlag;
    stopFlag.store(false, std::memory_order_relaxed);
    Board root = board;
    searchThread = std::thread([this, root, limits]() mutable {
        auto callback = [this](const std::string& s) { print(s); };
        const auto* netPtr = (useNNUE && network.loaded()) ? &network : nullptr;
        SearchResult result = searchPosition(root, limits, stopFlag, callback, hashMb, netPtr, threads);
        std::string best = (result.best.from == result.best.to) ? "0000" : Board::moveToUci(result.best);
        std::string line = "bestmove " + best;
        if (result.pv.size() > 1) line += " ponder " + Board::moveToUci(result.pv[1]);
        print(line);
    });
}
} // namespace breeze
