#include "breeze.hpp"
#include <algorithm>
#include <cmath>

namespace breeze {
namespace {
constexpr int VALUE[7] = {0,100,318,332,505,935,0};
int centerDistance(int sq) { int f=fileOf(sq),r=rankOf(sq); return std::abs(2*f-7)+std::abs(2*r-7); }
int pst(PieceType t, Color c, int sq, bool endgame) {
    int r = c==WHITE ? rankOf(sq) : 7-rankOf(sq);
    int f = fileOf(sq);
    int center = 14-centerDistance(sq);
    switch(t) {
        case PAWN: return r*7 + (center/3) + ((f==3||f==4)?3:0);
        case KNIGHT: return center*4 - (r==0?12:0);
        case BISHOP: return center*2 + r*2;
        case ROOK: return r*2 + ((r==6)?12:0) + ((f==3||f==4)?2:0);
        case QUEEN: return center + r;
        case KING: return endgame ? center*5 : -center*3 - r*2;
        default: return 0;
    }
}
}

int evaluate(const Board& b) {
    int material[2]{};
    int score[2]{};
    int pawns[2][8]{};
    bool bishop[2]{};
    int kingSq[2] = {-1,-1};
    bool endgame = (b.nonPawnMaterial(WHITE)+b.nonPawnMaterial(BLACK)) < 2600;
    const auto& s=b.raw();
    for(int sq=0;sq<64;++sq) {
        int p=s.squares[sq]; if(!p)continue;
        Color c=colorOf(p); PieceType t=typeOf(p);
        material[c]+=VALUE[t];
        score[c]+=VALUE[t]+pst(t,c,sq,endgame);
        if(t==PAWN)++pawns[c][fileOf(sq)];
        if(t==BISHOP)bishop[c]=true;
        if(t==KING)kingSq[c]=sq;
    }
    if(bishop[WHITE])score[WHITE]+=24;
    if(bishop[BLACK])score[BLACK]+=24;

    // Pawn structure: doubled, isolated, connected and passed pawns.
    for(Color c:{WHITE,BLACK}) {
        Color enemy=opposite(c);
        for(int f=0;f<8;++f) if(pawns[c][f]>1) score[c]-=13*(pawns[c][f]-1);
        for(int sq=0;sq<64;++sq) if(s.squares[sq]==makePiece(c,PAWN)) {
            int f=fileOf(sq),r=rankOf(sq),rel=c==WHITE?r:7-r;
            bool adjacent=(f>0&&pawns[c][f-1])||(f<7&&pawns[c][f+1]);
            if(!adjacent)score[c]-=10;
            bool passed=true;
            for(int ef=std::max(0,f-1);ef<=std::min(7,f+1);++ef)
                for(int er=0;er<8;++er) if(s.squares[sqOf(ef,er)]==makePiece(enemy,PAWN)) {
                    int enemyRel=enemy==WHITE?er:7-er;
                    if(enemyRel>rel){passed=false;break;}
                }
            if(passed)score[c]+=rel*rel*2+rel*4;
            const int supportRank = r + (c==WHITE ? -1 : 1);
            if(supportRank>=0 && supportRank<8) {
                bool supported=false;
                for(int df : {-1,1}) {
                    int sf=f+df;
                    if(sf>=0 && sf<8 && s.squares[sqOf(sf,supportRank)]==makePiece(c,PAWN)) supported=true;
                }
                if(supported) score[c]+=8;
            }
        }
    }

    // Mobility, rook files, and king shelter.
    for(int sq=0;sq<64;++sq) {
        int p=s.squares[sq]; if(!p)continue;
        Color c=colorOf(p); PieceType t=typeOf(p); int f=fileOf(sq),r=rankOf(sq);
        if(t==KNIGHT||t==BISHOP||t==ROOK||t==QUEEN) {
            int mobility=0;
            // Local mobility estimate; slider rays stop at the first occupied square.
            if(t==KNIGHT) {
                const int df[8]={1,2,2,1,-1,-2,-2,-1},dr[8]={2,1,-1,-2,-2,-1,1,2};
                for(int i=0;i<8;++i){int nf=f+df[i],nr=r+dr[i];if(nf>=0&&nf<8&&nr>=0&&nr<8&&(!s.squares[sqOf(nf,nr)]||colorOf(s.squares[sqOf(nf,nr)])!=c))++mobility;}
            } else {
                const int dfx[8]={1,-1,0,0,1,1,-1,-1},drx[8]={0,0,1,-1,1,-1,1,-1};
                for(int d=0;d<8;++d){bool diag=d>=4;if((t==BISHOP&&!diag)||(t==ROOK&&diag))continue;int nf=f+dfx[d],nr=r+drx[d];while(nf>=0&&nf<8&&nr>=0&&nr<8){int q=s.squares[sqOf(nf,nr)];if(q&&colorOf(q)==c)break;++mobility;if(q)break;nf+=dfx[d];nr+=drx[d];}}
            }
            score[c]+=mobility*(t==KNIGHT?4:t==BISHOP?3:t==ROOK?2:1);
        }
        if(t==ROOK) {
            bool ownPawn=false, enemyPawn=false;
            for(int rr=0;rr<8;++rr){int q=s.squares[sqOf(f,rr)];if(q&&typeOf(q)==PAWN){if(colorOf(q)==c)ownPawn=true;else enemyPawn=true;}}
            if(!ownPawn)score[c]+=enemyPawn?10:19;
        }
    }
    for(Color c:{WHITE,BLACK}) if(kingSq[c]>=0) {
        int sq=kingSq[c],f=fileOf(sq),r=rankOf(sq);
        if(!endgame) {
            int relRank=c==WHITE?r:7-r;
            score[c]+=(relRank<=1?12:0);
            for(int df=-1;df<=1;++df){int nf=f+df;if(nf<0||nf>7)continue;int shieldRank=r+(c==WHITE?1:-1);if(shieldRank>=0&&shieldRank<8&&s.squares[sqOf(nf,shieldRank)]==makePiece(c,PAWN))score[c]+=9;}
        }
    }
    int result=score[WHITE]-score[BLACK];
    // A small tempo bonus helps stable move selection in equal positions.
    result += b.sideToMove()==WHITE ? 10 : -10;
    return b.sideToMove()==WHITE ? result : -result;
}
} // namespace breeze
