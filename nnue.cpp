#include "breeze.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace breeze {
    namespace {
        constexpr uint32_t kFileVersion = 0x6A448AFAu;
        constexpr uint32_t kFileArchitecture = 0x9A4FCB66u;
        constexpr uint32_t kLayerStackTable = 0x5A7AB1E5u;
        constexpr uint32_t kFeatureTransformerHash = 0xA3060B95u;
        constexpr uint32_t kNetworkArchitectureHash = 0x63337116u;
        constexpr uint32_t kRawStacks = 192;
        constexpr uint32_t kStoredStacks = 92;
        constexpr size_t kL1 = 1024;
        constexpr size_t kPsqFeatures = 67584;
        constexpr size_t kThreatFeatures = 59808;
        constexpr size_t kPairFeatures = 4560;
        constexpr size_t kTransformerParamsBytes = 2048 + (kPsqFeatures + kThreatFeatures + kPairFeatures) * kL1;
        constexpr size_t kStackParamsBytes = 35204;
        constexpr size_t kStackRecordBytes = 4 + kStackParamsBytes;
        constexpr size_t kHeaderAndMapEnd = 492;
        constexpr size_t kFeatureParamsStart = kHeaderAndMapEnd + 4;
        constexpr size_t kStackRecordsStart = kFeatureParamsStart + kTransformerParamsBytes;
        constexpr size_t kExpectedFileBytes = kStackRecordsStart + kStoredStacks * kStackRecordBytes;
        constexpr int kPieceSquareBlock = 64 * 11;

        uint16_t u16le(const uint8_t* p) {
            return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
        }
        uint32_t u32le(const uint8_t* p) {
            return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
            (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
        }
        int16_t i16le(const uint8_t* p) { return static_cast<int16_t>(u16le(p)); }
        int32_t i32le(const uint8_t* p) { return static_cast<int32_t>(u32le(p)); }
        std::string hex32(uint32_t value) {
            std::ostringstream out;
            out << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
            return out.str();
        }

        int findKing(const Board& b, Color c) {
            const int king = makePiece(c, KING);
            for (int sq = 0; sq < 64; ++sq)
                if (b.pieceAt(sq) == king) return sq;
                return -1;
        }

        int normalizedPiece(int piece, Color perspective) {
            if (!piece) return 0;
            const Color orientedColor = Color(colorOf(piece) ^ perspective);
            const PieceType type = typeOf(piece);
            return int(type) + (orientedColor == BLACK ? 8 : 0);
        }

        bool geometricAttack(PieceType type, int from, int to, bool pawnDown = false) {
            const int df = fileOf(to) - fileOf(from);
            const int dr = rankOf(to) - rankOf(from);
            switch (type) {
                case PAWN: return std::abs(df) == 1 && dr == (pawnDown ? -1 : 1);
                case KNIGHT: return (std::abs(df) == 1 && std::abs(dr) == 2) ||
                    (std::abs(df) == 2 && std::abs(dr) == 1);
                case BISHOP: return std::abs(df) == std::abs(dr) && df != 0;
                case ROOK: return (df == 0) != (dr == 0);
                case QUEEN: return ((std::abs(df) == std::abs(dr)) && df != 0) || ((df == 0) != (dr == 0));
                case KING: return std::max(std::abs(df), std::abs(dr)) == 1;
                default: return false;
            }
        }

        struct ThreatTables {
            std::array<std::array<std::array<uint16_t, 64>, 64>, 7> toOffset{};
            std::array<int, 7> pieceOffset{};
            std::array<std::array<std::array<uint32_t, 2>, 15>, 15> pairBase{};
            ThreatTables() {
                constexpr int validTargets[15] = {0,4,10,8,8,10,0,0,0,4,10,8,8,10,0};
                constexpr int map[6][6] = {
                    {-1,0,-1,1,-1,-1},
                    {0,1,2,3,4,-1},
                    {0,1,2,3,-1,-1},
                    {0,1,2,3,-1,-1},
                    {0,1,2,3,4,-1},
                    {-1,-1,-1,-1,-1,-1}
                };
                for (int pt = PAWN; pt <= KING; ++pt) {
                    int offset = 0;
                    for (int from = 0; from < 64; ++from) {
                        for (int to = 0; to < 64; ++to) {
                            const bool down = (pt == PAWN && to < from);
                            int below = 0;
                            for (int sq = 0; sq < to; ++sq)
                                if (geometricAttack(PieceType(pt), from, sq, down)) ++below;
                                toOffset[pt][from][to] = uint16_t(offset + below);
                        }
                        if (pt == PAWN) {
                            if (from >= 8 && from <= 55)
                                for (int to = 0; to < 64; ++to)
                                    if (geometricAttack(PAWN, from, to, false)) ++offset;
                        } else {
                            for (int to = 0; to < 64; ++to)
                                if (geometricAttack(PieceType(pt), from, to)) ++offset;
                        }
                    }
                    pieceOffset[pt] = offset;
                }

                uint32_t cumulative = 0;
                for (int attacker : {1,2,3,4,5,6,9,10,11,12,13,14}) {
                    const int at = (attacker > 8) ? attacker - 8 : attacker;
                    const int valid = validTargets[attacker];
                    for (int attacked : {1,2,3,4,5,6,9,10,11,12,13,14}) {
                        const int dt = (attacked > 8) ? attacked - 8 : attacked;
                        const bool enemy = (attacker ^ attacked) == 8;
                        const int m = map[at - 1][dt - 1];
                        const bool semiExcluded = (at == dt) && (enemy || at != PAWN);
                        uint32_t base = (m < 0) ? uint32_t(kThreatFeatures) :
                        cumulative + (uint32_t(attacked > 8 ? 1 : 0) * (valid / 2) + uint32_t(m)) * uint32_t(pieceOffset[at]);
                        pairBase[attacker][attacked][0] = base;
                        pairBase[attacker][attacked][1] = (m < 0 || semiExcluded) ? uint32_t(kThreatFeatures) : base;
                    }
                    cumulative += uint32_t(valid) * uint32_t(pieceOffset[at]);
                }
            }
        };

        const ThreatTables& threatTables() {
            static const ThreatTables tables;
            return tables;
        }

        int orientSquare(int sq, int orientation) { return sq ^ orientation; }
        int kingOrientation(int kingSq, Color perspective) {
            const int horizontal = fileOf(kingSq) < 4 ? 7 : 0;
            return horizontal ^ (56 * int(perspective));
        }

        constexpr int kingBuckets[64] = {
            28,29,30,31,31,30,29,28,
            24,25,26,27,27,26,25,24,
            20,21,22,23,23,22,21,20,
            16,17,18,19,19,18,17,16,
            12,13,14,15,15,14,13,12,
            8, 9,10,11,11,10, 9, 8,
            4, 5, 6, 7, 7, 6, 5, 4,
            0, 1, 2, 3, 3, 2, 1, 0
        };

        int k32Block(const Board& b, Color perspective) {
            if (b.countPieces(opposite(perspective), QUEEN) != 0) return 2;
            int heavy = b.countPieces(WHITE, ROOK) + b.countPieces(BLACK, ROOK) +
            b.countPieces(WHITE, QUEEN) + b.countPieces(BLACK, QUEEN);
            return heavy ? 1 : 0;
        }

        int rawLayerStack(const Board& b) {
            const Color stm = b.sideToMove();
            const int band = std::max(0, (b.pieceCount() - 1) / 4);
            const int queens = 2 * int(b.countPieces(stm, QUEEN) != 0) +
            int(b.countPieces(opposite(stm), QUEEN) != 0);
            int pawnsQueenSide = 0, pawnsKingSide = 0;
            for (int sq = 0; sq < 64; ++sq) if (typeOf(b.pieceAt(sq)) == PAWN) {
                if (fileOf(sq) < 4) ++pawnsQueenSide; else ++pawnsKingSide;
            }
            const int layout = (pawnsQueenSide + pawnsKingSide == 0) ? 2 :
            (pawnsQueenSide && pawnsKingSide ? 0 : 1);
            int officers[2]{};
            for (int sq = 0; sq < 64; ++sq) {
                int p = b.pieceAt(sq);
                if (p && typeOf(p) != PAWN && typeOf(p) != KING) ++officers[colorOf(p)];
            }
            const int many = std::max(officers[WHITE], officers[BLACK]) >= 2;
            const int raw = ((band * 4 + queens) * 3 + layout) * 2 + many;
            return std::clamp(raw, 0, int(kRawStacks) - 1);
        }

        bool isAllowedThreat(PieceType attacker, PieceType target) {
            static constexpr int m[6][6] = {
                {-1,0,-1,1,-1,-1},
                {0,1,2,3,4,-1},
                {0,1,2,3,-1,-1},
                {0,1,2,3,-1,-1},
                {0,1,2,3,4,-1},
                {-1,-1,-1,-1,-1,-1}
            };
            return attacker >= PAWN && attacker <= KING && target >= PAWN && target <= KING && m[attacker-1][target-1] >= 0;
        }

        int gatherThreatFeatures(const Board& b, Color perspective, int kingSq, uint16_t* out, int maxOut) {
            const auto& tables = threatTables();
            const int orient = kingOrientation(kingSq, perspective);
            constexpr int knightDf[8] = {1,2,2,1,-1,-2,-2,-1};
            constexpr int knightDr[8] = {2,1,-1,-2,-2,-1,1,2};
            constexpr int rayDf[8] = {1,-1,0,0,1,1,-1,-1};
            constexpr int rayDr[8] = {0,0,1,-1,1,-1,1,-1};
            int count = 0;

            auto emitIfTarget = [&](int from, int to, int attackerPiece) {
                if (count >= maxOut) return;
                int targetPiece = b.pieceAt(to);
                if (!targetPiece) return;
                PieceType at = typeOf(attackerPiece), dt = typeOf(targetPiece);
                if (!isAllowedThreat(at, dt)) return;
                int fromO = orientSquare(from, orient), toO = orientSquare(to, orient);
                int attackerO = normalizedPiece(attackerPiece, perspective);
                int targetO = normalizedPiece(targetPiece, perspective);
                const int direction = (fromO < toO) ? 1 : 0;
                uint32_t index = tables.pairBase[attackerO][targetO][direction] + tables.toOffset[at][fromO][toO];
                if (index < kThreatFeatures) out[count++] = uint16_t(index);
            };

                for (int from = 0; from < 64; ++from) {
                    int p = b.pieceAt(from);
                    if (!p) continue;
                    PieceType pt = typeOf(p);
                    if (pt == KING) continue;
                    const int f = fileOf(from), r = rankOf(from);
                    if (pt == PAWN) {
                        const int dr = colorOf(p) == WHITE ? 1 : -1;
                        for (int df : {-1, 1}) {
                            int nf = f + df, nr = r + dr;
                            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                                emitIfTarget(from, sqOf(nf, nr), p);
                        }
                    } else if (pt == KNIGHT) {
                        for (int i = 0; i < 8; ++i) {
                            int nf = f + knightDf[i], nr = r + knightDr[i];
                            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) emitIfTarget(from, sqOf(nf, nr), p);
                        }
                    } else {
                        for (int d = 0; d < 8; ++d) {
                            const bool diagonal = d >= 4;
                            if (pt == BISHOP && !diagonal) continue;
                            if (pt == ROOK && diagonal) continue;
                            int nf = f + rayDf[d], nr = r + rayDr[d];
                            while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                                int to = sqOf(nf, nr);
                                if (b.pieceAt(to)) {
                                    emitIfTarget(from, to, p);
                                    break;
                                }
                                nf += rayDf[d]; nr += rayDr[d];
                            }
                        }
                    }
                }
                return count;
        }

        int gatherPairFeatures(const Board& b, Color perspective, int kingSq, uint16_t* out, int maxOut) {
            const int orient = kingOrientation(kingSq, perspective);
            int pawnSq[16]{};
            Color pawnColor[16]{};
            int pawnCount = 0;
            for (int sq = 0; sq < 64; ++sq) {
                int p = b.pieceAt(sq);
                if (p && typeOf(p) == PAWN && pawnCount < 16) {
                    pawnSq[pawnCount] = sq;
                    pawnColor[pawnCount] = colorOf(p);
                    ++pawnCount;
                }
            }
            int count = 0;
            for (int i = 0; i < pawnCount; ++i) {
                for (int j = i + 1; j < pawnCount; ++j) {
                    if (count >= maxOut) return count;
                    const int a = pawnSq[i], bSq = pawnSq[j];
                    if (std::abs(fileOf(a) - fileOf(bSq)) > 1) continue;
                    const int ao = orientSquare(a, orient), bo = orientSquare(bSq, orient);
                    if (ao < 8 || ao > 55 || bo < 8 || bo > 55) continue;
                    const int idA = 48 * int(Color(pawnColor[i] ^ perspective)) + ao - 8;
                    const int idB = 48 * int(Color(pawnColor[j] ^ perspective)) + bo - 8;
                    const int hi = std::max(idA, idB), lo = std::min(idA, idB);
                    const int index = hi * (hi - 1) / 2 + lo;
                    if (index >= 0 && index < int(kPairFeatures)) out[count++] = uint16_t(index);
                }
            }
            return count;
        }

        int gatherPsqFeatures(const Board& b, Color perspective, int kingSq, uint16_t* out, int maxOut) {
            const int flip = 56 * int(perspective);
            const int orient = kingOrientation(kingSq, perspective);
            const int bucket = kingBuckets[kingSq ^ flip] * kPieceSquareBlock;
            const int blockOffset = k32Block(b, perspective) * 22528;
            int count = 0;
            for (int sq = 0; sq < 64; ++sq) {
                if (count >= maxOut) break;
                const int p = b.pieceAt(sq);
                if (!p) continue;
                const PieceType pt = typeOf(p);
                const int orientedColor = int(Color(colorOf(p) ^ perspective));
                const int pieceIndex = (pt == KING) ? 10 * 64 : ((int(pt) - 1) * 2 + orientedColor) * 64;
                const int orientedSquare = sq ^ orient;
                const int index = blockOffset + bucket + pieceIndex + orientedSquare;
                if (index >= 0 && index < int(kPsqFeatures)) out[count++] = uint16_t(index);
            }
            return count;
        }

        int32_t sat16(int32_t x) { return std::clamp(x, -32768, 32767); }
        uint8_t clipActivation(int32_t x, int shift) {
            if (x <= 0) return 0;
            return uint8_t(std::clamp(x >> shift, 0, 127));
        }
        uint8_t squareActivation(int32_t x, int shiftBits) {
            const int32_t v = sat16(x);
            const int64_t square = int64_t(v) * int64_t(v);
            return uint8_t(std::clamp<int64_t>(square >> (2 * shiftBits + 7), 0, 127));
        }

        void addFeatureWeights(int32_t* __restrict accumulator, const int8_t* __restrict weights) {
            #if defined(__AVX2__)
            for (size_t i = 0; i < kL1; i += 16) {
                const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + i));
                const __m256i lo = _mm256_cvtepi8_epi32(packed);
                const __m256i hi = _mm256_cvtepi8_epi32(_mm_srli_si128(packed, 8));
                __m256i a0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + i));
                __m256i a1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(accumulator + i + 8));
                a0 = _mm256_add_epi32(a0, lo);
                a1 = _mm256_add_epi32(a1, hi);
                _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator + i), a0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(accumulator + i + 8), a1);
            }
            #else
            for (size_t i = 0; i < kL1; ++i)
                accumulator[i] += static_cast<int32_t>(weights[i]);
            #endif
        }

        int64_t dotProductU8I8(const uint8_t* __restrict activations, const int8_t* __restrict weights, size_t count) {
            #if defined(__AVX2__)
            const __m256i ones = _mm256_set1_epi16(1);
            __m256i acc = _mm256_setzero_si256();
            size_t i = 0;
            for (; i + 32 <= count; i += 32) {
                const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(activations + i));
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i));
                const __m256i pairs = _mm256_maddubs_epi16(a, w);
                const __m256i groups = _mm256_madd_epi16(pairs, ones);
                acc = _mm256_add_epi32(acc, groups);
            }
            alignas(32) int32_t partial[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(partial), acc);
            int64_t sum = 0;
            for (int lane = 0; lane < 8; ++lane) sum += partial[lane];
            for (; i < count; ++i) sum += int(activations[i]) * int(weights[i]);
            return sum;
            #else
            int64_t sum = 0;
            for (size_t i = 0; i < count; ++i) sum += int(activations[i]) * int(weights[i]);
            return sum;
            #endif
        }

    } // namespace

    void NNUEModel::clear() {
        ready = false;
        bytes = 0;
        desc.clear();
        fileData.clear();
        stacks.clear();
        ftBias.fill(0);
        fallback.fill(0);
        compactIndex.fill(0);
    }

    bool NNUEModel::load(const std::string& path, std::string& report) {
        clear();
        (void)threatTables();
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) { report = "Cannot open SFNNv20 network: " + path; return false; }
        const std::streamoff end = in.tellg();
        if (end < 0 || uint64_t(end) > std::numeric_limits<size_t>::max()) {
            report = "Invalid or oversized network file"; return false;
        }
        bytes = uint64_t(end);
        if (bytes != kExpectedFileBytes) {
            report = "Unexpected SFNNv20 file size: got " + std::to_string(bytes) +
            ", expected " + std::to_string(kExpectedFileBytes);
            clear(); return false;
        }
        fileData.resize(size_t(bytes));
        in.seekg(0, std::ios::beg);
        in.read(reinterpret_cast<char*>(fileData.data()), static_cast<std::streamsize>(fileData.size()));
        if (size_t(in.gcount()) != fileData.size()) {
            report = "Could not read the complete network file"; clear(); return false;
        }
        const uint8_t* d = fileData.data();
        const uint32_t version = u32le(d);
        const uint32_t architecture = u32le(d + 4);
        const uint32_t descriptionLength = u32le(d + 8);
        if (version != kFileVersion || architecture != kFileArchitecture || descriptionLength != 84) {
            report = "Unsupported network header (version " + hex32(version) +
            ", architecture " + hex32(architecture) + ")";
            clear(); return false;
        }
        desc.assign(reinterpret_cast<const char*>(d + 12), descriptionLength);
        if (u32le(d + 96) != kLayerStackTable || u32le(d + 100) != kRawStacks ||
            u32le(d + 104) != kStoredStacks) {
            report = "Layer-stack metadata does not match SFNNv20"; clear(); return false;
            }
            for (int i = 0; i < int(kRawStacks); ++i) fallback[i] = u16le(d + 108 + 2 * i);
            std::array<int, 192> compactById{};
        compactById.fill(-1);
        int compact = 0;
        for (int i = 0; i < int(kRawStacks); ++i) {
            if (fallback[i] >= kRawStacks) {
                report = "Layer-stack fallback index is out of range"; clear(); return false;
            }
            if (fallback[i] == i) compactById[i] = compact++;
        }
        if (compact != int(kStoredStacks)) {
            report = "The layer-stack fallback table has an unexpected number of stored networks";
            clear(); return false;
        }
        for (int i = 0; i < int(kRawStacks); ++i) {
            const uint16_t f = fallback[i];
            if (fallback[f] != f || compactById[f] < 0) {
                report = "Layer-stack fallback table is not canonical"; clear(); return false;
            }
            compactIndex[i] = uint16_t(compactById[f]);
        }
        if (u32le(d + kHeaderAndMapEnd) != kFeatureTransformerHash) {
            report = "Feature-transformer checksum mismatch"; clear(); return false;
        }
        for (int i = 0; i < 1024; ++i) ftBias[i] = i16le(d + kFeatureParamsStart + 2 * i);

        stacks.reserve(kStoredStacks);
        size_t cursor = kStackRecordsStart;
        for (int n = 0; n < int(kStoredStacks); ++n) {
            if (u32le(d + cursor) != kNetworkArchitectureHash) {
                report = "Network stack " + std::to_string(n) + " has an unexpected architecture checksum";
                clear(); return false;
            }
            cursor += 4;
            Stack net;
            net.w0.resize(32 * 1024);
            net.w1.resize(32 * 64);
            for (int i = 0; i < 32; ++i) net.b0[i] = i32le(d + cursor + 4 * i);
            cursor += 32 * 4;
            for (size_t i = 0; i < net.w0.size(); ++i) net.w0[i] = static_cast<int8_t>(d[cursor + i]);
            cursor += net.w0.size();
            for (int i = 0; i < 32; ++i) net.b1[i] = i32le(d + cursor + 4 * i);
            cursor += 32 * 4;
            for (size_t i = 0; i < net.w1.size(); ++i) net.w1[i] = static_cast<int8_t>(d[cursor + i]);
            cursor += net.w1.size();
            net.b2 = i32le(d + cursor); cursor += 4;
            for (size_t i = 0; i < net.w2.size(); ++i) net.w2[i] = static_cast<int8_t>(d[cursor + i]);
            cursor += net.w2.size();
            stacks.push_back(std::move(net));
        }
        if (cursor != fileData.size()) {
            report = "Network parser did not finish at the exact end of the file"; clear(); return false;
        }
        ready = true;
        std::ostringstream out;
        out << "SFNNv20 network loaded successfully\n"
        << "File bytes: " << bytes << "\n"
        << "Stored subnetworks: " << stacks.size() << " (192 position configurations)\n"
        << "Feature transformer: 1024 lanes; K32H3 67584 + FullThreats 59808 + PP_3Wide 4560\n"
        << "Network: 1024 -> 32 -> 32 -> 1, dual squared/clipped activations\n"
        << "Inference: AVX2 kernels when compiled with AVX2; scalar fallback otherwise\n"
        << "Description: " << desc;
        report = out.str();
        return true;
    }

    int NNUEModel::layerStackFor(const Board& board) const {
        if (!ready || stacks.empty()) return -1;
        const int raw = rawLayerStack(board);
        return compactIndex[raw];
    }

    // Optimized Cached Accumulator for Fast Leaf Evaluations
    int NNUEModel::evaluate(const Board& board) const {
        if (!ready || stacks.empty()) return breeze::evaluate(board);

        // Static thread-local cache key / accumulator to avoid full recalculations when board hasn't changed structure
        struct AccumCache {
            uint64_t lastHash = 0;
            alignas(64) int32_t accum[2][1024];
        };
        thread_local AccumCache cache;

        const uint64_t currentHash = board.hash();
        const int kingSq[2] = {findKing(board, WHITE), findKing(board, BLACK)};
        if (kingSq[0] < 0 || kingSq[1] < 0) return 0;

        const uint8_t* data = fileData.data();
        const size_t dataSize = fileData.size();
        const size_t threatWeightOffset = kFeatureParamsStart + 2048;
        const size_t pairWeightOffset = threatWeightOffset + kThreatFeatures * kL1;
        const size_t psqWeightOffset = pairWeightOffset + kPairFeatures * kL1;

        if (cache.lastHash != currentHash) {
            for (int pi = 0; pi < 2; ++pi) {
                const Color perspective = Color(pi);
                uint16_t threatBuf[256];
                uint16_t pairBuf[128];
                uint16_t psqBuf[64];

                const int nThreat = gatherThreatFeatures(board, perspective, kingSq[pi], threatBuf, 256);
                const int nPair = gatherPairFeatures(board, perspective, kingSq[pi], pairBuf, 128);
                const int nPsq = gatherPsqFeatures(board, perspective, kingSq[pi], psqBuf, 64);

                for (int lane = 0; lane < 1024; ++lane) cache.accum[pi][lane] = ftBias[lane];

                auto addFeatures = [&](const uint16_t* features, int count, size_t base, size_t dimension) {
                    for (int idx = 0; idx < count; ++idx) {
                        const uint16_t feature = features[idx];
                        if (size_t(feature) >= dimension) continue;
                        const size_t offset = base + size_t(feature) * kL1;
                        if (offset > dataSize || kL1 > dataSize - offset) continue;
                        const auto* w = reinterpret_cast<const int8_t*>(data + offset);
                        addFeatureWeights(cache.accum[pi], w);
                    }
                };

                addFeatures(threatBuf, nThreat, threatWeightOffset, kThreatFeatures);
                addFeatures(pairBuf, nPair, pairWeightOffset, kPairFeatures);
                addFeatures(psqBuf, nPsq, psqWeightOffset, kPsqFeatures);
            }
            cache.lastHash = currentHash;
        }

        alignas(64) uint8_t transformed[1024];
        const int stm = int(board.sideToMove());
        for (int side = 0; side < 2; ++side) {
            const int32_t* a = cache.accum[side == 0 ? stm : (stm ^ 1)];
            for (int i = 0; i < 512; ++i) {
                const int x = std::clamp(a[i], 0, 255);
                const int y = std::clamp(a[i + 512], 0, 255);
                transformed[side * 512 + i] = uint8_t((x * y) / 512);
            }
        }

        const int stackIdx = layerStackFor(board);
        if (stackIdx < 0 || size_t(stackIdx) >= stacks.size()) return breeze::evaluate(board);
        const Stack& net = stacks[size_t(stackIdx)];

        alignas(64) int32_t fc0[32];
        alignas(32) uint8_t sqr0[32], clip0[32];
        for (int o = 0; o < 32; ++o) {
            int64_t sum = net.b0[o];
            const int8_t* row = net.w0.data() + size_t(o) * 1024;
            sum += dotProductU8I8(transformed, row, 1024);
            fc0[o] = int32_t(std::clamp<int64_t>(sum, INT32_MIN, INT32_MAX));
        }
        for (int o = 0; o < 32; ++o) {
            sqr0[o] = squareActivation(fc0[o], 7);
            clip0[o] = clipActivation(fc0[o], 7);
        }
        alignas(64) uint8_t input1[64];
        for (int i = 0; i < 32; ++i) { input1[i] = sqr0[i]; input1[32 + i] = clip0[i]; }

        alignas(64) int32_t fc1[32];
        alignas(32) uint8_t sqr1[32], clip1[32];
        for (int o = 0; o < 32; ++o) {
            int64_t sum = net.b1[o];
            const int8_t* row = net.w1.data() + size_t(o) * 64;
            sum += dotProductU8I8(input1, row, 64);
            fc1[o] = int32_t(std::clamp<int64_t>(sum, INT32_MIN, INT32_MAX));
            sqr1[o] = squareActivation(fc1[o], 6);
            clip1[o] = clipActivation(fc1[o], 6);
        }

        alignas(64) uint8_t input2[128];
        for (int i = 0; i < 32; ++i) {
            input2[i] = sqr0[i]; input2[32 + i] = clip0[i];
            input2[64 + i] = sqr1[i]; input2[96 + i] = clip1[i];
        }
        int64_t forward = net.b2;
        forward += dotProductU8I8(input2, net.w2.data(), 128);
        forward += int64_t(fc0[30]) - int64_t(fc0[31]);

        const int64_t networkValue = (forward * 600) / 16384;

        int material = 0;
        for (int sq = 0; sq < 64; ++sq) {
            const int p = board.pieceAt(sq);
            if (!p) continue;
            switch (typeOf(p)) {
                case PAWN:   material += 208;  break;
                case KNIGHT: material += 781;  break;
                case BISHOP: material += 825;  break;
                case ROOK:   material += 1276; break;
                case QUEEN:  material += 2538; break;
                default: break;
            }
        }
        const int64_t scaledValue = (networkValue * (700 + material / 32)) / 1024;
        constexpr int NormalizeToPawnValue = 328;
        return int((scaledValue * 100) / NormalizeToPawnValue);
    }

} // namespace breeze
