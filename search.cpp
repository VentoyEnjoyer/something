#include "breeze.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <system_error>
#include <vector>

namespace breeze {
namespace {
enum Bound : uint8_t { EXACT = 0, LOWER = 1, UPPER = 2 };

struct TTEntry {
    std::atomic<uint64_t> keyXor{0};
    std::atomic<uint64_t> data{0};
    // Sequence counter makes lock-free readers reject torn key/data snapshots.
    std::atomic<uint32_t> sequence{0};
    std::atomic_flag writer = ATOMIC_FLAG_INIT;
};

struct alignas(64) TTCluster {
    TTEntry entries[4];
};

struct DecodedTT {
    uint32_t move = 0;
    int16_t score = 0;
    int8_t depth = -1;
    uint8_t bound = UPPER;
    uint8_t gen = 0;
};

struct alignas(64) AlignedCounter {
    std::atomic<uint64_t> value{0};
};

struct alignas(64) AlignedSeldepth {
    std::atomic<int> value{0};
};

struct SharedHeuristics {
    std::atomic<int32_t> history[2][64][64]{};
    std::atomic<uint32_t> counterMoves[2][64][64]{};

    void clear() {
        for (int c = 0; c < 2; ++c) {
            for (int f = 0; f < 64; ++f) {
                for (int t = 0; t < 64; ++t) {
                    history[c][f][t].store(0, std::memory_order_relaxed);
                    counterMoves[c][f][t].store(0, std::memory_order_relaxed);
                }
            }
        }
    }

    void ageHistory() {
        for (int c = 0; c < 2; ++c) {
            for (int f = 0; f < 64; ++f) {
                for (int t = 0; t < 64; ++t) {
                    int32_t cur = history[c][f][t].load(std::memory_order_relaxed);
                    history[c][f][t].store(cur / 2, std::memory_order_relaxed);
                }
            }
        }
    }

    int32_t getHistory(Color side, int from, int to) const {
        return history[int(side)][from][to].load(std::memory_order_relaxed);
    }

    void updateHistory(Color side, int from, int to, int bonus) {
        std::atomic<int32_t>& cell = history[int(side)][from][to];
        int32_t clampedBonus = std::clamp(bonus, -16384, 16384);
        int32_t cur = cell.load(std::memory_order_relaxed);
        for (;;) {
            // A CAS loop prevents concurrent workers from losing history updates.
            // The clamped bonus keeps the gravity multiply inside signed 32-bit range.
            int32_t next = cur + clampedBonus
                         - (cur * std::abs(clampedBonus)) / 32768;
            next = std::clamp(next, -32768, 32768);
            if (cell.compare_exchange_weak(cur, next,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
                break;
            }
        }
    }

    Move getCounterMove(Color side, const Move& prevMove) const {
        if (prevMove.from == prevMove.to) return Move{};
        uint32_t c = counterMoves[int(side)][prevMove.from][prevMove.to].load(std::memory_order_relaxed);
        Move m;
        m.from = c & 63;
        m.to = (c >> 6) & 63;
        m.promo = (c >> 12) & 15;
        m.flags = (c >> 16) & 255;
        return m;
    }

    void setCounterMove(Color side, const Move& prevMove, const Move& counter) {
        if (prevMove.from == prevMove.to) return;
        counterMoves[int(side)][prevMove.from][prevMove.to].store(counter.code(), std::memory_order_relaxed);
    }
};

uint64_t packTT(uint32_t move, int16_t score, int8_t depth, uint8_t bound, uint8_t gen) {
    uint8_t meta = (bound & 0x03u) | uint8_t((gen & 0x3Fu) << 2);
    return uint64_t(move)
         | (uint64_t(uint16_t(score)) << 32)
         | (uint64_t(uint8_t(depth)) << 48)
         | (uint64_t(meta) << 56);
}

DecodedTT unpackTT(uint64_t d) {
    DecodedTT out;
    out.move = uint32_t(d & 0xFFFFFFFFULL);
    out.score = int16_t((d >> 32) & 0xFFFFULL);
    out.depth = int8_t((d >> 48) & 0xFFULL);
    uint8_t meta = uint8_t((d >> 56) & 0xFFULL);
    out.bound = meta & 0x03u;
    out.gen = (meta >> 2) & 0x3Fu;
    return out;
}

std::vector<TTCluster> persistentTT;
int persistentTTMb = 0;
std::atomic<uint8_t> ttGeneration{1};
std::mutex ttResizeMutex;
SharedHeuristics sharedHeuristics;
std::mutex searchStateMutex;

void resizePersistentTT(int mb) {
    std::lock_guard<std::mutex> lock(ttResizeMutex);
    mb = std::clamp(mb, 1, 4096);
    if (persistentTTMb == mb && !persistentTT.empty()) return;
    size_t bytes = size_t(mb) * 1024u * 1024u;
    size_t count = std::max<size_t>(1, bytes / sizeof(TTCluster));
    while (count >= 1) {
        try {
            persistentTT = std::vector<TTCluster>(count);
            persistentTTMb = mb;
            return;
        } catch (const std::bad_alloc&) {
            count /= 2;
        }
    }
    persistentTT = std::vector<TTCluster>(1);
    persistentTTMb = 0;
}

struct ThreadOutcome {
    Move bestMove{};
    int score = -INF;
    int depth = 0;
    std::vector<Move> pv{};
    bool valid = false;
};

class Searcher {
public:
    Searcher(const SearchLimits& l, std::atomic<bool>& st, std::atomic<bool>& localSt,
             InfoCallback cb, const NNUEModel* model, int tid,
             AlignedCounter* allCounters, AlignedSeldepth* allSeldepths,
             int numThreads, uint8_t currentGen)
      : limits(l), stop(st), localStop(localSt), info(std::move(cb)),
        start(std::chrono::steady_clock::now()), tt(persistentTT),
        nnue(model), threadId(tid), counters(allCounters),
        seldepths(allSeldepths), threadCount(numThreads),
        generation(currentGen) {}

    SearchResult run(Board root, ThreadOutcome& outcome) {
        start = std::chrono::steady_clock::now();
        int remain = limits.movetimeMs;
        if (threadId == 0 && !limits.infinite && remain <= 0) {
            const bool whiteToMove = root.sideToMove() == WHITE;
            const int clock = whiteToMove ? limits.wtimeMs : limits.btimeMs;
            const int increment = whiteToMove ? limits.wincMs : limits.bincMs;
            if (clock >= 0) {
                const int movesToGo = limits.movesToGo > 0 ? limits.movesToGo : 28;
                remain = std::max(10, clock / (movesToGo + 1) + increment * 3 / 4 - 20);
                remain = std::min(remain, std::max(10, clock - 10));
            }
        }
        if (threadId == 0 && remain > 0 && !limits.infinite) {
            if (limits.ponder) ponderBudgetMs = remain;
            else {
                deadline = start + std::chrono::milliseconds(remain);
                ponderDeadlineActive = true;
            }
        }

        SearchResult result;
        MoveList legal;
        root.generateLegal(legal);
        if (limits.searchMovesSpecified) {
            int kept = 0;
            for (int i = 0; i < legal.count; ++i) {
                if (std::find(limits.searchMoves.begin(), limits.searchMoves.end(), legal[i]) != limits.searchMoves.end()) {
                    legal[kept++] = legal[i];
                }
            }
            legal.count = kept;
        }
        if (legal.empty()) {
            result.best = Move{};
            result.score = root.inCheck(root.sideToMove()) ? -MATE : 0;
            outcome.bestMove = result.best;
            outcome.score = result.score;
            outcome.depth = 0;
            outcome.valid = true;
            return result;
        }
        result.best = legal[0];
        outcome.bestMove = legal[0];

        int maxDepth = limits.depth > 0 ? std::min(limits.depth, MAX_PLY - 4) : MAX_PLY - 4;
        int previous = 0;

        int depth = 1;
        int helperCycle = 0;

        while (depth <= maxDepth) {
            if (stopped()) break;

            if (threadId > 0) {
                ++helperCycle;
                if (depth > 1 && ((depth + threadId + (helperCycle / 3)) % 2 == 0) && depth < maxDepth) {
                    ++depth;
                    continue;
                }
            }

            pathCount = 0;
            pushPath(root.hash());

            int baseWindow = 24 + (threadId % 5) * 4;
            int window = depth >= 4 ? baseWindow : INF;
            int alpha = window == INF ? -INF : std::max(-INF, previous - window);
            int beta = window == INF ? INF : std::min(INF, previous + window);
            int score = 0;

            for (;;) {
                score = searchRoot(root, depth, alpha, beta, result.best);
                if (stopped()) break;
                if (score <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = std::max(-INF, alpha - window);
                    window += window / 2 + 4;
                    continue;
                }
                if (score >= beta) {
                    beta = std::min(INF, beta + window);
                    window += window / 2 + 4;
                    continue;
                }
                break;
            }
            if (stopped()) break;

            previous = score;
            result.score = score;
            result.depth = depth;
            result.pv.clear();
            for (int i = 0; i < pvLength[0]; ++i) result.pv.push_back(pv[0][i]);
            if (!result.pv.empty()) result.best = result.pv[0];

            outcome.bestMove = result.best;
            outcome.score = result.score;
            outcome.depth = result.depth;
            outcome.pv = result.pv;
            outcome.valid = true;

            if (threadId == 0) {
                syncStats();
                result.nodes = aggregateNodes();
                emitInfo(result);
                if (limits.nodes && result.nodes >= limits.nodes) {
                    stop.store(true, std::memory_order_relaxed);
                    localStop.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            ++depth;
        }
        syncStats();
        result.nodes = aggregateNodes();
        return result;
    }

private:
    static constexpr int MAX_PATH = MAX_PLY + 256;

    const SearchLimits& limits;
    std::atomic<bool>& stop;
    std::atomic<bool>& localStop;
    InfoCallback info;
    std::chrono::steady_clock::time_point start, deadline{};
    int ponderBudgetMs = 0;
    bool ponderDeadlineActive = false;
    uint64_t nodes = 0;
    uint64_t lastNodeSync = 0;
    int maxSeldepth = 0;
    std::vector<TTCluster>& tt;
    const NNUEModel* nnue = nullptr;
    int threadId = 0;
    AlignedCounter* counters = nullptr;
    AlignedSeldepth* seldepths = nullptr;
    int threadCount = 1;
    uint8_t generation = 1;

    int evaluatePosition(const Board& b) const { return (nnue && nnue->loaded()) ? nnue->evaluate(b) : evaluate(b); }

    int localHistory[2][64][64]{};
    Move killers[MAX_PLY][2]{};
    Move moveStack[MAX_PLY + 4]{};
    Move pv[MAX_PLY][MAX_PLY]{};
    int pvLength[MAX_PLY]{};
    uint64_t path[MAX_PATH]{};
    int pathCount = 0;

    void pushPath(uint64_t h) {
        if (pathCount < MAX_PATH) path[pathCount++] = h;
    }

    void popPath() {
        if (pathCount > 0) --pathCount;
    }

    void syncStats() {
        if (counters) {
            counters[threadId].value.store(nodes, std::memory_order_relaxed);
            lastNodeSync = nodes;
        }
        if (seldepths) {
            seldepths[threadId].value.store(maxSeldepth, std::memory_order_relaxed);
        }
    }

    uint64_t aggregateNodes() const {
        if (!counters) return nodes;
        uint64_t sum = 0;
        for (int i = 0; i < threadCount; ++i) {
            sum += counters[i].value.load(std::memory_order_relaxed);
        }
        return sum;
    }

    int aggregateSeldepth() const {
        int best = maxSeldepth;
        if (!seldepths) return best;
        for (int i = 0; i < threadCount; ++i) {
            best = std::max(best, seldepths[i].value.load(std::memory_order_relaxed));
        }
        return best;
    }

    bool stopped() {
        if (stop.load(std::memory_order_relaxed) || localStop.load(std::memory_order_relaxed)) return true;

        if ((nodes - lastNodeSync) >= 512u) {
            syncStats();
            if (threadId == 0) {
                if (limits.nodes && aggregateNodes() >= limits.nodes) {
                    stop.store(true, std::memory_order_relaxed);
                    localStop.store(true, std::memory_order_relaxed);
                    return true;
                }
                if (limits.ponder && limits.ponderHitFlag && limits.ponderHitFlag->load(std::memory_order_relaxed) && !ponderDeadlineActive) {
                    if (ponderBudgetMs > 0) deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ponderBudgetMs);
                    ponderDeadlineActive = true;
                }
                if (ponderDeadlineActive && deadline.time_since_epoch().count() != 0 && std::chrono::steady_clock::now() >= deadline) {
                    stop.store(true, std::memory_order_relaxed);
                    localStop.store(true, std::memory_order_relaxed);
                    return true;
                }
            }
        }
        return false;
    }

    static int scoreToTT(int score, int ply) {
        if (score > MATE_BOUND) return score + ply;
        if (score < -MATE_BOUND) return score - ply;
        return score;
    }
    static int scoreFromTT(int score, int ply) {
        if (score > MATE_BOUND) return score - ply;
        if (score < -MATE_BOUND) return score + ply;
        return score;
    }

    bool probe(uint64_t key, DecodedTT& out) {
        if (tt.empty()) return false;
        TTCluster& cluster = tt[key % tt.size()];
        for (int i = 0; i < 4; ++i) {
            TTEntry& e = cluster.entries[i];
            uint64_t k = 0, d = 0;
            for (;;) {
                const uint32_t before = e.sequence.load(std::memory_order_acquire);
                if (before & 1u) {
                    std::this_thread::yield();
                    continue;
                }
                k = e.keyXor.load(std::memory_order_relaxed);
                d = e.data.load(std::memory_order_relaxed);
                const uint32_t after = e.sequence.load(std::memory_order_acquire);
                if (before == after) break;
            }
            if ((k ^ d) == key && d != 0) {
                out = unpackTT(d);
                return true;
            }
        }
        return false;
    }

    void store(uint64_t key, int depth, int score, Bound bound, const Move& m, int ply) {
        if (tt.empty()) return;
        TTCluster& cluster = tt[key % tt.size()];

        int replaceIdx = 0;
        int lowestPriority = std::numeric_limits<int>::max();

        for (int i = 0; i < 4; ++i) {
            TTEntry& e = cluster.entries[i];
            uint64_t oldK = 0, oldD = 0;
            for (;;) {
                const uint32_t before = e.sequence.load(std::memory_order_acquire);
                if (before & 1u) {
                    std::this_thread::yield();
                    continue;
                }
                oldK = e.keyXor.load(std::memory_order_relaxed);
                oldD = e.data.load(std::memory_order_relaxed);
                const uint32_t after = e.sequence.load(std::memory_order_acquire);
                if (before == after) break;
            }

            if (oldD == 0) {
                replaceIdx = i;
                break;
            }

            DecodedTT existing = unpackTT(oldD);
            bool sameKey = ((oldK ^ oldD) == key);

            if (sameKey) {
                if (depth < existing.depth && bound != EXACT && existing.gen == generation) {
                    return;
                }
                replaceIdx = i;
                break;
            }

            int ageDiff = int((generation - existing.gen) & 0x3Fu);
            int priority = int(existing.depth) - ageDiff * 4 + (existing.bound == EXACT ? 2 : 0);
            if (priority < lowestPriority) {
                lowestPriority = priority;
                replaceIdx = i;
            }
        }

        int8_t d = int8_t(std::clamp(depth, -1, 127));
        int16_t s = int16_t(std::clamp(scoreToTT(score, ply), -32767, 32767));
        uint64_t packed = packTT(m.code(), s, d, uint8_t(bound), generation);

        TTEntry& target = cluster.entries[replaceIdx];
        while (target.writer.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        // Re-check a same-key entry after taking the writer lock.
        target.sequence.fetch_add(1, std::memory_order_acq_rel);
        const uint64_t liveKeyXor = target.keyXor.load(std::memory_order_relaxed);
        const uint64_t liveData = target.data.load(std::memory_order_relaxed);
        if (liveData != 0 && (liveKeyXor ^ liveData) == key) {
            const DecodedTT live = unpackTT(liveData);
            if (depth < live.depth && bound != EXACT && live.gen == generation) {
                target.sequence.fetch_add(1, std::memory_order_release);
                target.writer.clear(std::memory_order_release);
                return;
            }
        }
        target.data.store(packed, std::memory_order_relaxed);
        target.keyXor.store(key ^ packed, std::memory_order_relaxed);
        target.sequence.fetch_add(1, std::memory_order_release);
        target.writer.clear(std::memory_order_release);
    }

    Move decodeMove(uint32_t c) {
        Move m;
        m.from = c & 63;
        m.to = (c >> 6) & 63;
        m.promo = (c >> 12) & 15;
        m.flags = (c >> 16) & 255;
        return m;
    }

    int moveScore(const Board& b, const Move& m, const Move& ttMove, int ply) {
        if (m == ttMove) return 2000000;
        int p = b.pieceAt(m.from), q = b.pieceAt(m.to);
        if (m.flags & Move::ENPASSANT) q = makePiece(opposite(b.sideToMove()), PAWN);
        if (m.isCapture()) return 1000000 + 10 * (q ? int(typeOf(q)) : 0) - int(typeOf(p));
        if (m.isPromotion()) return 900000 + int(m.promo) * 100;
        if (ply < MAX_PLY) {
            if (m == killers[ply][0]) return 800000;
            if (m == killers[ply][1]) return 799000;
            if (ply > 0 && m == sharedHeuristics.getCounterMove(b.sideToMove(), moveStack[ply - 1])) {
                return 798000;
            }
        }

        const Color side = b.sideToMove();
        int localH = localHistory[int(side)][m.from][m.to];
        int sharedH = sharedHeuristics.getHistory(side, m.from, m.to);
        int base = (localH + sharedH) / 2;

        if (threadId > 0) {
            uint32_t h = uint32_t(b.hash() >> 32) ^ uint32_t(ply * 1315423911u);
            uint32_t mix = (uint32_t(m.code()) * 2654435761u) ^ (uint32_t(threadId) * 0x9e3779b9u) ^ h;
            base += int(mix & 511u) - 256;
        }
        return base;
    }

    void orderMoves(const Board& b, MoveList& moves, const Move& ttMove, int ply) {
        int scores[MAX_MOVES];
        for (int i = 0; i < moves.count; ++i) {
            scores[i] = moveScore(b, moves[i], ttMove, ply);
        }
        for (int i = 1; i < moves.count; ++i) {
            Move keyMove = moves[i];
            int keyScore = scores[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < keyScore) {
                moves[j + 1] = moves[j];
                scores[j + 1] = scores[j];
                --j;
            }
            moves[j + 1] = keyMove;
            scores[j + 1] = keyScore;
        }
    }

    bool isRepetition(uint64_t h) const {
        int count = 0;
        for (int i = 0; i + 1 < pathCount; ++i) {
            if (path[i] == h && ++count >= 2) return true;
        }
        return false;
    }

    int qsearch(Board& b, int alpha, int beta, int ply) {
        pvLength[ply] = ply;
        maxSeldepth = std::max(maxSeldepth, ply);
        if (stopped()) return 0;
        ++nodes;
        if (b.raw().halfmove >= 100 || isRepetition(b.hash()) || b.isInsufficientMaterial()) return 0;
        bool check = b.inCheck(b.sideToMove());
        int stand = check ? -INF : evaluatePosition(b);
        if (!check) {
            if (stand >= beta) return stand;
            if (stand > alpha) alpha = stand;
        }
        if (ply >= MAX_PLY - 1) return stand;
        MoveList moves;
        b.generateLegal(moves, !check);
        if (moves.empty()) return check ? -MATE + ply : alpha;
        orderMoves(b, moves, Move{}, ply);
        for (int i = 0; i < moves.count; ++i) {
            const Move& m = moves[i];
            Undo u; b.makeMove(m, u); pushPath(b.hash());
            if (ply < MAX_PLY + 4) moveStack[ply] = m;
            int score = -qsearch(b, -beta, -alpha, ply + 1);
            popPath(); b.unmakeMove(u);
            if (stopped()) return 0;
            if (score >= beta) return score;
            if (score > alpha) {
                alpha = score;
                pv[ply][ply] = m;
                for (int j = ply + 1; j < pvLength[ply + 1]; ++j) pv[ply][j] = pv[ply + 1][j];
                pvLength[ply] = pvLength[ply + 1];
            }
        }
        return alpha;
    }

    int negamax(Board& b, int depth, int alpha, int beta, int ply, bool allowNull = true) {
        pvLength[ply] = ply;
        maxSeldepth = std::max(maxSeldepth, ply);
        if (stopped()) return 0;
        if (ply >= MAX_PLY - 2) return evaluatePosition(b);
        if (b.raw().halfmove >= 100 || isRepetition(b.hash()) || b.isInsufficientMaterial()) return 0;
        bool check = b.inCheck(b.sideToMove());
        if (depth <= 0) return qsearch(b, alpha, beta, ply);
        ++nodes;
        int originalAlpha = alpha;
        Move ttMove{};
        DecodedTT tte;
        if (probe(b.hash(), tte)) {
            ttMove = decodeMove(tte.move);
            int tscore = scoreFromTT(tte.score, ply);
            if (tte.depth >= depth && ply > 0) {
                if (tte.bound == EXACT) return tscore;
                if (tte.bound == LOWER && tscore >= beta) return tscore;
                if (tte.bound == UPPER && tscore <= alpha) return tscore;
            }
        }
        int staticEval = check ? -INF : evaluatePosition(b);
        if (!check && depth <= 2 && staticEval - 120 * depth >= beta) return staticEval;
        if (allowNull && !check && depth >= 3 && b.nonPawnMaterial(b.sideToMove()) > 500) {
            Undo nu; b.makeNullMove(nu); pushPath(b.hash());
            if (ply < MAX_PLY + 4) moveStack[ply] = Move{};
            int reduction = 2 + depth / 4;
            int score = -negamax(b, depth - 1 - reduction, -beta, -beta + 1, ply + 1, false);
            popPath(); b.unmakeMove(nu);
            if (stopped()) return 0;
            if (score >= beta) return score;
        }
        MoveList moves;
        b.generateLegal(moves);
        if (moves.empty()) return check ? -MATE + ply : 0;
        orderMoves(b, moves, ttMove, ply);

        Move quietsTried[MAX_MOVES];
        int quietsCount = 0;

        int best = -INF; Move bestMove{}; int moveNo = 0;
        for (int i = 0; i < moves.count; ++i) {
            const Move& m = moves[i];
            bool quiet = !m.isCapture() && !m.isPromotion();
            Undo u; b.makeMove(m, u); pushPath(b.hash());
            if (ply < MAX_PLY + 4) moveStack[ply] = m;

            int childDepth = depth - 1;
            const bool reduced = moveNo > 0 && depth >= 3 && moveNo >= 4 && quiet && !check;
            if (reduced) childDepth = std::max(0, childDepth - 1);
            int score;
            if (moveNo == 0) {
                score = -negamax(b, childDepth, -beta, -alpha, ply + 1, true);
            } else {
                score = -negamax(b, childDepth, -alpha - 1, -alpha, ply + 1, true);
                if ((reduced && score > alpha) || (!reduced && score > alpha && score < beta))
                    score = -negamax(b, depth - 1, -beta, -alpha, ply + 1, true);
            }
            popPath(); b.unmakeMove(u);
            if (stopped()) return 0;
            ++moveNo;
            if (score > best) { best = score; bestMove = m; }
            if (score > alpha) {
                alpha = score;
                pv[ply][ply] = m;
                for (int j = ply + 1; j < pvLength[ply + 1]; ++j) pv[ply][j] = pv[ply + 1][j];
                pvLength[ply] = pvLength[ply + 1];
            }
            if (alpha >= beta) {
                if (quiet && ply < MAX_PLY) {
                    if (!(m == killers[ply][0])) {
                        killers[ply][1] = killers[ply][0];
                        killers[ply][0] = m;
                    }
                    const Color side = b.sideToMove();
                    // Bound the history arithmetic before multiplying by its gravity term.
                    const int bonus = std::clamp(depth * depth * 24, 0, 16384);
                    int& lh = localHistory[int(side)][m.from][m.to];
                    lh = std::clamp(lh + bonus - (lh * std::abs(bonus)) / 32768, -32768, 32768);
                    sharedHeuristics.updateHistory(side, m.from, m.to, bonus);

                    for (int q = 0; q < quietsCount; ++q) {
                        const Move& qm = quietsTried[q];
                        int& qlh = localHistory[int(side)][qm.from][qm.to];
                        qlh = std::clamp(qlh - bonus - (qlh * std::abs(bonus)) / 32768, -32768, 32768);
                        sharedHeuristics.updateHistory(side, qm.from, qm.to, -bonus / 2);
                    }

                    if (ply > 0) {
                        sharedHeuristics.setCounterMove(side, moveStack[ply - 1], m);
                    }
                }
                break;
            }
            if (quiet && quietsCount < MAX_MOVES) {
                quietsTried[quietsCount++] = m;
            }
        }
        Bound bound = best <= originalAlpha ? UPPER : best >= beta ? LOWER : EXACT;
        store(b.hash(), depth, best, bound, bestMove, ply);
        return best;
    }

    int searchRoot(Board& b, int depth, int alpha, int beta, Move& bestMove) {
        MoveList moves;
        b.generateLegal(moves);
        if (limits.searchMovesSpecified) {
            int kept = 0;
            for (int i = 0; i < moves.count; ++i) {
                if (std::find(limits.searchMoves.begin(), limits.searchMoves.end(), moves[i]) != limits.searchMoves.end()) {
                    moves[kept++] = moves[i];
                }
            }
            moves.count = kept;
        }
        if (moves.empty()) return b.inCheck(b.sideToMove()) ? -MATE : 0;
        const int originalAlpha = alpha;
        Move ttMove{};
        DecodedTT tte;
        if (probe(b.hash(), tte)) ttMove = decodeMove(tte.move);
        orderMoves(b, moves, ttMove, 0);
        int best = -INF, moveNo = 0; Move localBest = moves[0];
        pvLength[0] = 0;
        for (int i = 0; i < moves.count; ++i) {
            const Move& m = moves[i];
            if (stopped()) break;
            Undo u; b.makeMove(m, u); pushPath(b.hash());
            moveStack[0] = m;
            int score;
            if (moveNo == 0) score = -negamax(b, depth - 1, -beta, -alpha, 1, true);
            else {
                score = -negamax(b, depth - 1, -alpha - 1, -alpha, 1, true);
                if (score > alpha && score < beta) score = -negamax(b, depth - 1, -beta, -alpha, 1, true);
            }
            popPath(); b.unmakeMove(u);
            if (stopped()) break;
            ++moveNo;
            if (score > best) { best = score; localBest = m; }
            if (score > alpha) {
                alpha = score;
                pv[0][0] = m;
                for (int j = 1; j < pvLength[1]; ++j) pv[0][j] = pv[1][j];
                pvLength[0] = pvLength[1];
            }
            if (alpha >= beta) break;
        }
        if (moveNo > 0) bestMove = localBest;
        store(b.hash(), depth, best, best <= originalAlpha ? UPPER : best >= beta ? LOWER : EXACT, bestMove, 0);
        return best;
    }

    void emitInfo(const SearchResult& r) {
        if (!info) return;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        uint64_t reportedNodes = r.nodes > 0 ? r.nodes : nodes;
        uint64_t nps = ms > 0 ? reportedNodes * 1000 / uint64_t(ms) : reportedNodes * 1000;
        int globalSeldepth = aggregateSeldepth();
        std::ostringstream o;
        o << "info depth " << r.depth << " seldepth " << globalSeldepth << " score ";
        if (std::abs(r.score) > MATE_BOUND) {
            int plies = MATE - std::abs(r.score);
            int mate = (plies + 1) / 2;
            if (r.score < 0) mate = -mate;
            o << "mate " << mate;
        } else {
            o << "cp " << r.score;
        }
        size_t sample = std::min<size_t>(tt.size(), 250), used = 0;
        uint8_t curGen = generation;
        for (size_t i = 0; i < sample; ++i) {
            for (int j = 0; j < 4; ++j) {
                uint64_t d = tt[i].entries[j].data.load(std::memory_order_relaxed);
                if (d != 0 && unpackTT(d).gen == curGen) ++used;
            }
        }
        int hashfull = sample ? int(used * 1000 / (sample * 4)) : 0;
        o << " nodes " << reportedNodes << " nps " << nps << " time " << ms << " hashfull " << hashfull << " pv";
        for (const Move& m : r.pv) o << ' ' << Board::moveToUci(m);
        info(o.str());
    }
};

SearchResult selectBestThreadResult(const std::vector<ThreadOutcome>& outcomes,
                                    const SearchResult& mainResult,
                                    uint64_t totalNodes) {
    SearchResult finalResult = mainResult;
    finalResult.nodes = totalNodes;

    int bestIdx = 0;
    for (int t = 1; t < int(outcomes.size()); ++t) {
        if (!outcomes[t].valid) continue;
        const ThreadOutcome& cand = outcomes[t];
        

        // Scores from different depths are not directly comparable. Prefer the
        // deepest fully completed iteration; keep thread 0 on ties.
        if (!outcomes[bestIdx].valid || cand.depth > outcomes[bestIdx].depth) {
            bestIdx = t;
        }
    }

    if (bestIdx > 0 && outcomes[bestIdx].valid) {
        finalResult.best = outcomes[bestIdx].bestMove;
        finalResult.score = outcomes[bestIdx].score;
        finalResult.depth = outcomes[bestIdx].depth;
        finalResult.pv = outcomes[bestIdx].pv;
    }
    return finalResult;
}

} // namespace

SearchResult searchPosition(Board root, const SearchLimits& limits, std::atomic<bool>& stop,
                            InfoCallback info, int hashMb, const NNUEModel* nnue, int threadCount) {
    std::lock_guard<std::mutex> stateLock(searchStateMutex);
    threadCount = std::clamp(threadCount, 1, 64);

    resizePersistentTT(hashMb);
    uint8_t prevGen = ttGeneration.load(std::memory_order_relaxed);
    uint8_t nextGen = uint8_t((prevGen + 1) & 0x3Fu);
    if (nextGen == 0) nextGen = 1;
    ttGeneration.store(nextGen, std::memory_order_relaxed);
    sharedHeuristics.ageHistory();

    std::atomic<bool> localStop{false};
    std::vector<AlignedCounter> counters(static_cast<size_t>(threadCount));
    std::vector<AlignedSeldepth> seldepths(static_cast<size_t>(threadCount));
    std::vector<ThreadOutcome> outcomes(static_cast<size_t>(threadCount));
    std::vector<std::thread> helpers;
    helpers.reserve(size_t(threadCount - 1));

    int activeThreads = 1;
    for (int t = 1; t < threadCount; ++t) {
        try {
            helpers.emplace_back([&, t, nextGen]() {
                try {
                    Searcher helper(limits, stop, localStop, nullptr, nnue, t,
                                    counters.data(), seldepths.data(), threadCount, nextGen);
                    helper.run(root, outcomes[t]);
                } catch (...) {
                    outcomes[t].valid = false;
                }
            });
            ++activeThreads;
        } catch (const std::system_error&) {
            break;
        }
    }

    if (info) {
        std::ostringstream message;
        message << "info string Lazy SMP: " << activeThreads << " threads";
        info(message.str());
    }

    InfoCallback infoCopy = info;
    Searcher mainSearcher(limits, stop, localStop, std::move(info), nnue, 0,
                          counters.data(), seldepths.data(), threadCount, nextGen);
    // Keep the root available while helpers copy it.
    SearchResult mainResult = mainSearcher.run(root, outcomes[0]);

    localStop.store(true, std::memory_order_relaxed);
    for (auto& h : helpers) {
        if (h.joinable()) h.join();
    }

    uint64_t sum = 0;
    int maxSel = 0;
    for (int i = 0; i < threadCount; ++i) {
        sum += counters[i].value.load(std::memory_order_relaxed);
        maxSel = std::max(maxSel, seldepths[i].value.load(std::memory_order_relaxed));
    }

    SearchResult finalResult = selectBestThreadResult(outcomes, mainResult, sum);
    if (infoCopy && (finalResult.best != mainResult.best || finalResult.depth > mainResult.depth)) {
        std::ostringstream o;
        o << "info depth " << finalResult.depth << " seldepth " << maxSel << " score ";
        if (std::abs(finalResult.score) > MATE_BOUND) {
            int plies = MATE - std::abs(finalResult.score);
            int mate = (plies + 1) / 2;
            if (finalResult.score < 0) mate = -mate;
            o << "mate " << mate;
        } else {
            o << "cp " << finalResult.score;
        }
        o << " nodes " << finalResult.nodes << " pv";
        for (const Move& m : finalResult.pv) o << ' ' << Board::moveToUci(m);
        infoCopy(o.str());
    }
    return finalResult;
}

void clearSearchHash() {
    std::lock_guard<std::mutex> stateLock(searchStateMutex);
    for (auto& cluster : persistentTT) {
        for (int i = 0; i < 4; ++i) {
            cluster.entries[i].keyXor.store(0, std::memory_order_relaxed);
            cluster.entries[i].data.store(0, std::memory_order_relaxed);
        }
    }
    ttGeneration.store(1, std::memory_order_relaxed);
    sharedHeuristics.clear();
}

uint64_t perft(Board& b, int depth) {
    if (depth <= 0) return 1;
    MoveList moves;
    b.generateLegal(moves);
    uint64_t nodes = 0;
    for (int i = 0; i < moves.count; ++i) {
        Undo u; b.makeMove(moves[i], u);
        nodes += perft(b, depth - 1);
        b.unmakeMove(u);
    }
    return nodes;
}
} // namespace breeze
