#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

namespace breeze {

constexpr int MAX_PLY = 128;
constexpr int MAX_MOVES = 256;
constexpr int INF = 32000;
constexpr int MATE = 30000;
constexpr int MATE_BOUND = MATE - MAX_PLY;

enum Color : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { NO_TYPE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };
inline Color opposite(Color c) { return Color(c ^ 1); }
inline PieceType typeOf(int p) { return p ? PieceType((p - 1) % 6 + 1) : NO_TYPE; }
inline Color colorOf(int p) { return p <= 6 ? WHITE : BLACK; }
inline int makePiece(Color c, PieceType t) { return t == NO_TYPE ? 0 : int(t) + (c == BLACK ? 6 : 0); }
inline int fileOf(int sq) { return sq & 7; }
inline int rankOf(int sq) { return sq >> 3; }
inline int sqOf(int file, int rank) { return rank * 8 + file; }
std::string squareName(int sq);
int parseSquare(const std::string& s);

struct Move {
    uint8_t from = 0, to = 0, promo = 0, flags = 0;
    enum : uint8_t { CAPTURE = 1, ENPASSANT = 2, CASTLE = 4, DOUBLE_PUSH = 8, PROMOTION = 16 };
    uint32_t code() const { return uint32_t(from) | (uint32_t(to) << 6) | (uint32_t(promo) << 12) | (uint32_t(flags) << 16); }
    bool operator==(const Move& o) const { return code() == o.code(); }
    bool operator!=(const Move& o) const { return code() != o.code(); }
    bool isCapture() const { return flags & CAPTURE; }
    bool isPromotion() const { return flags & PROMOTION; }
};

struct MoveList {
    Move moves[MAX_MOVES]{};
    int count = 0;
    void clear() { count = 0; }
    void push_back(const Move& m) { if (count < MAX_MOVES) moves[count++] = m; }
    bool empty() const { return count == 0; }
    int size() const { return count; }
    Move& operator[](int i) { return moves[i]; }
    const Move& operator[](int i) const { return moves[i]; }
    Move* begin() { return moves; }
    Move* end() { return moves + count; }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + count; }
};

struct BoardState {
    std::array<uint8_t, 64> squares{};
    Color side = WHITE;
    uint8_t castling = 0;
    int8_t ep = -1;
    uint16_t halfmove = 0;
    uint16_t fullmove = 1;
    uint64_t hash = 0;
};

struct Undo { BoardState before; };

class Board {
public:
    Board();
    void setStartPosition();
    bool setFEN(const std::string& fen, std::string* error = nullptr);
    std::string fen() const;
    const BoardState& state() const { return s; }
    Color sideToMove() const { return s.side; }
    int pieceAt(int sq) const { return sq >= 0 && sq < 64 ? s.squares[sq] : 0; }
    uint64_t hash() const { return s.hash; }
    bool isSquareAttacked(int sq, Color by) const;
    bool inCheck(Color c) const;
    void generatePseudoLegal(MoveList& out, bool capturesOnly = false) const;
    void generateLegal(MoveList& out, bool capturesOnly = false);
    std::vector<Move> pseudoLegalMoves(bool capturesOnly = false) const;
    std::vector<Move> legalMoves(bool capturesOnly = false);
    bool makeMove(const Move& m, Undo& u);
    void unmakeMove(const Undo& u) { s = u.before; }
    void makeNullMove(Undo& u);
    bool parseUciMove(const std::string& text, Move& out);
    static std::string moveToUci(const Move& m);
    int pieceCount() const;
    bool isInsufficientMaterial() const;
    int nonPawnMaterial(Color c) const;
    int countPieces(Color c, PieceType t) const;
    const BoardState& raw() const { return s; }
private:
    BoardState s;
    void recomputeHash();
};

class NNUEModel {
public:
    NNUEModel() = default;
    bool load(const std::string& path, std::string& report);
    void clear();
    bool loaded() const { return ready; }
    bool inferenceReady() const { return ready; }
    uint64_t fileBytes() const { return bytes; }
    const std::string& description() const { return desc; }
    int evaluate(const Board& board) const;
    int layerStackFor(const Board& board) const;
private:
    struct Stack {
        std::array<int32_t, 32> b0{};
        std::vector<int8_t> w0;
        std::array<int32_t, 32> b1{};
        std::vector<int8_t> w1;
        int32_t b2 = 0;
        std::array<int8_t, 128> w2{};
    };
    bool ready = false;
    uint64_t bytes = 0;
    std::string desc;
    std::array<int16_t, 1024> ftBias{};
    std::vector<uint8_t> fileData;
    std::vector<Stack> stacks;
    std::array<uint16_t, 192> fallback{};
    std::array<uint16_t, 192> compactIndex{};
};

int evaluate(const Board& b);

struct SearchLimits {
    int depth = 0;
    uint64_t nodes = 0;
    int movetimeMs = 0;
    int wtimeMs = -1, btimeMs = -1;
    int wincMs = 0, bincMs = 0;
    int movesToGo = 0;
    bool infinite = false;
    bool ponder = false;
    bool searchMovesSpecified = false;
    std::vector<Move> searchMoves;
    std::atomic<bool>* ponderHitFlag = nullptr;
};
struct SearchResult {
    Move best{};
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    std::vector<Move> pv;
};
using InfoCallback = std::function<void(const std::string&)>;
SearchResult searchPosition(Board root, const SearchLimits& limits,
                            std::atomic<bool>& stop, InfoCallback info,
                            int hashMb = 64, const NNUEModel* nnue = nullptr,
                            int threadCount = 1);
void clearSearchHash();
uint64_t perft(Board& b, int depth);

class UCI {
public:
    UCI();
    ~UCI();
    int loop();
private:
    Board board;
    NNUEModel network;
    bool useNNUE = false;
    int hashMb = 64;
    int threads = 1;
    int moveOverhead = 15;
    bool ponderEnabled = true;
    std::atomic<bool> stopFlag{false};
    std::atomic<bool> ponderHitFlag{false};
    std::thread searchThread;
    std::mutex outputMutex;
    void stopSearch();
    void print(const std::string& s);
    void handleCommand(const std::string& line);
    void handleGo(const std::string& rest);
    void setOption(const std::string& rest);
};

} // namespace breeze
