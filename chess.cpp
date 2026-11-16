#include "breeze.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <random>

namespace breeze {
namespace {
constexpr uint8_t WK = 1, WQ = 2, BK = 4, BQ = 8;
struct Zobrist {
    uint64_t piece[13][64]{};
    uint64_t side = 0;
    uint64_t castle[16]{};
    uint64_t ep[8]{};
    Zobrist() {
        uint64_t x = 0xBEE5EED5A17C9E31ULL;
        auto next = [&]() { x += 0x9e3779b97f4a7c15ULL; uint64_t z=x; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); };
        for (auto& p : piece) for (auto& q : p) q = next();
        side = next();
        for (auto& q : castle) q = next();
        for (auto& q : ep) q = next();
    }
};
const Zobrist& zob() { static const Zobrist z; return z; }
const int knightDf[8] = {1,2,2,1,-1,-2,-2,-1};
const int knightDr[8] = {2,1,-1,-2,-2,-1,1,2};
const int kingDf[8] = {1,1,0,-1,-1,-1,0,1};
const int kingDr[8] = {0,1,1,1,0,-1,-1,-1};
const int bishopDf[4] = {1,1,-1,-1};
const int bishopDr[4] = {1,-1,1,-1};
const int rookDf[4] = {1,-1,0,0};
const int rookDr[4] = {0,0,1,-1};
char pieceChar(int p) {
    static const char chars[] = ".PNBRQKpnbrqk";
    return p >= 0 && p <= 12 ? chars[p] : '?';
}
int charPiece(char c) {
    const std::string chars = ".PNBRQKpnbrqk";
    auto p = chars.find(c);
    return p == std::string::npos ? 0 : int(p);
}
} // namespace

std::string squareName(int sq) {
    if (sq < 0 || sq >= 64) return "??";
    std::string s(2, ' '); s[0] = char('a' + fileOf(sq)); s[1] = char('1' + rankOf(sq)); return s;
}
int parseSquare(const std::string& s) {
    if (s.size() < 2 || s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return -1;
    return sqOf(s[0]-'a', s[1]-'1');
}

Board::Board() { setStartPosition(); }
void Board::recomputeHash() {
    const auto& z = zob();
    uint64_t h = 0;
    for (int sq=0; sq<64; ++sq) if (s.squares[sq]) h ^= z.piece[s.squares[sq]][sq];
    if (s.side == BLACK) h ^= z.side;
    h ^= z.castle[s.castling & 15];
    if (s.ep >= 0) h ^= z.ep[fileOf(s.ep)];
    s.hash = h;
}
void Board::setStartPosition() {
    s = BoardState{};
    const char* back = "RNBQKBNR";
    for (int f=0; f<8; ++f) {
        s.squares[f] = uint8_t(charPiece(back[f]));
        s.squares[8+f] = makePiece(WHITE, PAWN);
        s.squares[48+f] = makePiece(BLACK, PAWN);
        s.squares[56+f] = makePiece(BLACK, PieceType(typeOf(charPiece(back[f]))));
    }
    s.side = WHITE; s.castling = WK|WQ|BK|BQ; s.ep = -1; s.halfmove=0; s.fullmove=1;
    recomputeHash();
}

bool Board::setFEN(const std::string& fenText, std::string* error) {
    BoardState n{}; n.fullmove = 1; n.ep = -1;
    std::istringstream in(fenText);
    std::string placement, side, castle, ep; int half=0, full=1;
    if (!(in >> placement >> side >> castle >> ep)) { if(error)*error="FEN needs at least four fields"; return false; }
    in >> half >> full;
    int rank=7, file=0;
    for (char c : placement) {
        if (c=='/') { if(file!=8 || rank==0) { if(error)*error="bad rank separator"; return false; } --rank; file=0; continue; }
        if (c>='1' && c<='8') { file += c-'0'; if(file>8) { if(error)*error="rank overflows"; return false; } continue; }
        int p=charPiece(c);
        if(!p || file>=8 || rank<0) { if(error)*error="invalid piece placement"; return false; }
        n.squares[sqOf(file++,rank)] = uint8_t(p);
    }
    if(rank!=0 || file!=8) { if(error)*error="piece placement does not describe 8 ranks"; return false; }
    if(side=="w") n.side=WHITE; else if(side=="b") n.side=BLACK; else { if(error)*error="side must be w or b"; return false; }
    n.castling=0;
    if(castle!="-") for(char c:castle) {
        if(c=='K') n.castling|=WK; else if(c=='Q') n.castling|=WQ; else if(c=='k') n.castling|=BK; else if(c=='q') n.castling|=BQ;
        else { if(error)*error="invalid castling field"; return false; }
    }
    if(ep!="-") { n.ep=int8_t(parseSquare(ep)); if(n.ep<0) { if(error)*error="invalid en-passant square"; return false; } }
    n.halfmove=uint16_t(std::clamp(half,0,65535)); n.fullmove=uint16_t(std::clamp(full,1,65535));
    int wk=0,bk=0;
    for(int p:n.squares) { if(p==makePiece(WHITE,KING))++wk; if(p==makePiece(BLACK,KING))++bk; }
    if(wk!=1 || bk!=1) { if(error)*error="FEN must contain exactly one king per side"; return false; }
    s=n; recomputeHash(); return true;
}
std::string Board::fen() const {
    std::ostringstream o;
    for(int r=7;r>=0;--r) {
        int empty=0;
        for(int f=0;f<8;++f) { int p=s.squares[sqOf(f,r)]; if(!p) ++empty; else { if(empty){o<<empty;empty=0;} o<<pieceChar(p); } }
        if(empty) o<<empty;
        if(r) o<<'/';
    }
    o << (s.side==WHITE?" w ":" b ");
    if(!s.castling)o<<'-'; else { if(s.castling&WK)o<<'K'; if(s.castling&WQ)o<<'Q'; if(s.castling&BK)o<<'k'; if(s.castling&BQ)o<<'q'; }
    o << ' ' << (s.ep<0?"-":squareName(s.ep)) << ' ' << s.halfmove << ' ' << s.fullmove;
    return o.str();
}

bool Board::isSquareAttacked(int sq, Color by) const {
    const int f=fileOf(sq), r=rankOf(sq);
    int sourceRank = r + (by==WHITE ? -1 : 1);
    if(sourceRank>=0 && sourceRank<8) for(int df : {-1,1}) {
        int sf=f+df; if(sf<0||sf>7)continue;
        if(s.squares[sqOf(sf,sourceRank)]==makePiece(by,PAWN)) return true;
    }
    for(int i=0;i<8;++i) {
        int nf=f+knightDf[i], nr=r+knightDr[i];
        if(nf>=0&&nf<8&&nr>=0&&nr<8&&s.squares[sqOf(nf,nr)]==makePiece(by,KNIGHT))return true;
    }
    for(int i=0;i<8;++i) {
        int nf=f+kingDf[i], nr=r+kingDr[i];
        if(nf>=0&&nf<8&&nr>=0&&nr<8&&s.squares[sqOf(nf,nr)]==makePiece(by,KING))return true;
    }
    for(int d=0;d<8;++d) {
        int nf=f+kingDf[d], nr=r+kingDr[d], dist=1;
        while(nf>=0&&nf<8&&nr>=0&&nr<8) {
            int p=s.squares[sqOf(nf,nr)];
            if(p) {
                if(colorOf(p)==by) {
                    PieceType t=typeOf(p);
                    if(t==QUEEN || (dist==1 && t==KING) || ((d%2==1) && t==BISHOP) || ((d%2==0) && t==ROOK)) return true;
                }
                break;
            }
            ++dist; nf+=kingDf[d]; nr+=kingDr[d];
        }
    }
    return false;
}
bool Board::inCheck(Color c) const {
    int king=-1; const int kp=makePiece(c,KING);
    for(int sq=0;sq<64;++sq)if(s.squares[sq]==kp){king=sq;break;}
    return king<0 || isSquareAttacked(king,opposite(c));
}

void Board::generatePseudoLegal(MoveList& out, bool capturesOnly) const {
    out.clear();
    const Color us=s.side, them=opposite(us);
    auto push=[&](int from,int to,uint8_t flags=0,uint8_t promo=0){
        if(to<0||to>=64)return;
        int target=s.squares[to];
        if(target && colorOf(target)==us)return;
        if(target && typeOf(target)==KING)return;
        if(target)flags|=Move::CAPTURE;
        if(capturesOnly && !(flags&(Move::CAPTURE|Move::ENPASSANT|Move::PROMOTION)))return;
        out.push_back(Move{uint8_t(from),uint8_t(to),promo,flags});
    };
    for(int from=0;from<64;++from) {
        int p=s.squares[from]; if(!p||colorOf(p)!=us)continue;
        PieceType pt=typeOf(p); int f=fileOf(from),r=rankOf(from);
        if(pt==PAWN) {
            int dir=us==WHITE?1:-1, start=us==WHITE?1:6, last=us==WHITE?7:0;
            int nr=r+dir;
            if(nr>=0&&nr<8&&!s.squares[sqOf(f,nr)]) {
                int to=sqOf(f,nr);
                if(nr==last) {
                    for(uint8_t pr:{uint8_t(QUEEN),uint8_t(ROOK),uint8_t(BISHOP),uint8_t(KNIGHT)})
                        push(from,to,Move::PROMOTION,pr);
                } else if(!capturesOnly) {
                    push(from,to);
                    if(r==start && !s.squares[sqOf(f,r+2*dir)])
                        push(from,sqOf(f,r+2*dir),Move::DOUBLE_PUSH);
                }
            }
            for(int df:{-1,1}) { int nf=f+df; if(nf<0||nf>7)continue; int tr=r+dir;if(tr<0||tr>7)continue; int to=sqOf(nf,tr);
                if(s.squares[to]&&colorOf(s.squares[to])==them) {
                    if(tr==last) for(uint8_t pr:{uint8_t(QUEEN),uint8_t(ROOK),uint8_t(BISHOP),uint8_t(KNIGHT)})push(from,to,Move::PROMOTION,pr);
                    else push(from,to);
                } else if(to==s.ep) push(from,to,Move::ENPASSANT|Move::CAPTURE);
            }
        } else if(pt==KNIGHT) {
            for(int i=0;i<8;++i){int nf=f+knightDf[i],nr=r+knightDr[i];if(nf>=0&&nf<8&&nr>=0&&nr<8)push(from,sqOf(nf,nr));}
        } else if(pt==KING) {
            for(int i=0;i<8;++i){int nf=f+kingDf[i],nr=r+kingDr[i];if(nf>=0&&nf<8&&nr>=0&&nr<8)push(from,sqOf(nf,nr));}
            if(!capturesOnly && !inCheck(us)) {
                if(us==WHITE && from==4) {
                    if((s.castling&WK)&&s.squares[5]==0&&s.squares[6]==0&&s.squares[7]==makePiece(WHITE,ROOK)&&!isSquareAttacked(5,them)&&!isSquareAttacked(6,them))push(4,6,Move::CASTLE);
                    if((s.castling&WQ)&&s.squares[3]==0&&s.squares[2]==0&&s.squares[1]==0&&s.squares[0]==makePiece(WHITE,ROOK)&&!isSquareAttacked(3,them)&&!isSquareAttacked(2,them))push(4,2,Move::CASTLE);
                }
                if(us==BLACK && from==60) {
                    if((s.castling&BK)&&s.squares[61]==0&&s.squares[62]==0&&s.squares[63]==makePiece(BLACK,ROOK)&&!isSquareAttacked(61,them)&&!isSquareAttacked(62,them))push(60,62,Move::CASTLE);
                    if((s.castling&BQ)&&s.squares[59]==0&&s.squares[58]==0&&s.squares[57]==0&&s.squares[56]==makePiece(BLACK,ROOK)&&!isSquareAttacked(59,them)&&!isSquareAttacked(58,them))push(60,58,Move::CASTLE);
                }
            }
        } else {
            auto slide=[&](const int* dfs,const int* drs,int n){for(int d=0;d<n;++d){int nf=f+dfs[d],nr=r+drs[d];while(nf>=0&&nf<8&&nr>=0&&nr<8){int to=sqOf(nf,nr);if(s.squares[to]){if(colorOf(s.squares[to])==them)push(from,to);break;}push(from,to);nf+=dfs[d];nr+=drs[d];}}};
            if(pt==BISHOP||pt==QUEEN)slide(bishopDf,bishopDr,4);
            if(pt==ROOK||pt==QUEEN)slide(rookDf,rookDr,4);
        }
    }
}

void Board::generateLegal(MoveList& out, bool capturesOnly) {
    MoveList ps;
    generatePseudoLegal(ps, capturesOnly);
    out.clear();
    const Color us = s.side;
    for (int i = 0; i < ps.count; ++i) {
        const Move& m = ps.moves[i];
        Undo u;
        if (!makeMove(m, u)) continue;
        bool ok = !inCheck(us);
        unmakeMove(u);
        if (ok) out.push_back(m);
    }
}

std::vector<Move> Board::pseudoLegalMoves(bool capturesOnly) const {
    MoveList list;
    generatePseudoLegal(list, capturesOnly);
    return std::vector<Move>(list.begin(), list.end());
}

std::vector<Move> Board::legalMoves(bool capturesOnly) {
    MoveList list;
    generateLegal(list, capturesOnly);
    return std::vector<Move>(list.begin(), list.end());
}

bool Board::makeMove(const Move& m, Undo& u) {
    if(m.from>=64||m.to>=64||m.from==m.to) return false;
    const int moving=s.squares[m.from];
    if(!moving||colorOf(moving)!=s.side) return false;
    u.before=s;
    const auto& z=zob();
    const Color us=s.side;
    uint64_t h=s.hash;
    h^=z.castle[s.castling&15];
    if(s.ep>=0) h^=z.ep[fileOf(s.ep)];
    h^=z.side;
    h^=z.piece[moving][m.from];

    int captured=s.squares[m.to];
    int capturedSquare=m.to;
    if(m.flags&Move::ENPASSANT) {
        capturedSquare=int(m.to)+(us==WHITE?-8:8);
        if(capturedSquare>=0&&capturedSquare<64) captured=s.squares[capturedSquare];
        else return false;
    }
    if(captured) h^=z.piece[captured][capturedSquare];

    s.squares[m.from]=0;
    if(m.flags&Move::ENPASSANT) s.squares[capturedSquare]=0;
    int placed=moving;
    if(m.flags&Move::PROMOTION) placed=makePiece(us,PieceType(m.promo));
    s.squares[m.to]=uint8_t(placed);
    h^=z.piece[placed][m.to];

    if(m.flags&Move::CASTLE) {
        int rookFrom=-1,rookTo=-1;
        if(m.to==6){rookFrom=7;rookTo=5;}
        else if(m.to==2){rookFrom=0;rookTo=3;}
        else if(m.to==62){rookFrom=63;rookTo=61;}
        else if(m.to==58){rookFrom=56;rookTo=59;}
        if(rookFrom>=0) {
            const int rook=s.squares[rookFrom];
            if(rook) { h^=z.piece[rook][rookFrom]; h^=z.piece[rook][rookTo]; }
            s.squares[rookTo]=s.squares[rookFrom];
            s.squares[rookFrom]=0;
        }
    }
    if(typeOf(moving)==KING) s.castling &= us==WHITE ? uint8_t(~(WK|WQ)) : uint8_t(~(BK|BQ));
    if(m.from==0||m.to==0) s.castling &= uint8_t(~WQ);
    if(m.from==7||m.to==7) s.castling &= uint8_t(~WK);
    if(m.from==56||m.to==56) s.castling &= uint8_t(~BQ);
    if(m.from==63||m.to==63) s.castling &= uint8_t(~BK);
    s.ep=-1;
    if(m.flags&Move::DOUBLE_PUSH) s.ep=int8_t((int(m.from)+int(m.to))/2);
    if(typeOf(moving)==PAWN||captured) s.halfmove=0;
    else if(s.halfmove<65535) ++s.halfmove;
    if(us==BLACK&&s.fullmove<65535) ++s.fullmove;
    s.side=opposite(us);
    h^=z.castle[s.castling&15];
    if(s.ep>=0) h^=z.ep[fileOf(s.ep)];
    s.hash=h;
    return true;
}
void Board::makeNullMove(Undo& u) {
    u.before=s;
    const auto& z=zob();
    uint64_t h=s.hash;
    if(s.ep>=0) h^=z.ep[fileOf(s.ep)];
    h^=z.side;
    s.ep=-1;
    if(s.halfmove<65535) ++s.halfmove;
    if(s.side==BLACK&&s.fullmove<65535) ++s.fullmove;
    s.side=opposite(s.side);
    s.hash=h;
}
bool Board::parseUciMove(const std::string& text, Move& out) {
    if(text.size()<4) return false;
    int from=parseSquare(text.substr(0,2)),to=parseSquare(text.substr(2,2));
    if(from<0||to<0) return false;
    char prom=text.size()>4?char(std::tolower(text[4])):0; uint8_t p=prom=='q'?QUEEN:prom=='r'?ROOK:prom=='b'?BISHOP:prom=='n'?KNIGHT:0;
    MoveList list;
    generateLegal(list);
    for(const Move& m:list)if(m.from==from&&m.to==to&&m.promo==p){out=m;return true;}
    return false;
}
std::string Board::moveToUci(const Move& m) {
    std::string s=squareName(m.from)+squareName(m.to);
    if(m.promo){char c='q';if(m.promo==ROOK)c='r';else if(m.promo==BISHOP)c='b';else if(m.promo==KNIGHT)c='n';s+=c;}
    return s;
}
int Board::pieceCount() const { int n=0;for(int p:s.squares)if(p)++n;return n; }
bool Board::isInsufficientMaterial() const {
    int minors=0;
    int bishopSquareColor=-1;
    bool bishopsSameColor=true;
    for(int sq=0;sq<64;++sq) {
        const int p=s.squares[sq];
        if(!p) continue;
        const PieceType t=typeOf(p);
        if(t==PAWN || t==ROOK || t==QUEEN) return false;
        if(t==KNIGHT || t==BISHOP) ++minors;
        if(t==BISHOP) {
            const int color=(fileOf(sq)+rankOf(sq))&1;
            if(bishopSquareColor<0) bishopSquareColor=color;
            else if(bishopSquareColor!=color) bishopsSameColor=false;
        }
    }
    return minors<=1 || (minors>0 && bishopsSameColor &&
        countPieces(WHITE,KNIGHT)==0 && countPieces(BLACK,KNIGHT)==0);
}
int Board::countPieces(Color c, PieceType t) const { int n=0;for(int p:s.squares)if(p&&colorOf(p)==c&&typeOf(p)==t)++n;return n; }
int Board::nonPawnMaterial(Color c) const { int n=0;for(int p:s.squares)if(p&&colorOf(p)==c){switch(typeOf(p)){case KNIGHT:n+=320;break;case BISHOP:n+=335;break;case ROOK:n+=500;break;case QUEEN:n+=900;break;default:break;}}return n; }

} // namespace breeze
