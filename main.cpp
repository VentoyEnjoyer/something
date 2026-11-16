#include "breeze.hpp"
#include <iostream>
#include <string>

int main(int argc,char** argv) {
    if(argc>=2 && std::string(argv[1])=="--probe") {
        if(argc<3){std::cerr<<"usage: Breeze5 --probe network.nnue\n";return 2;}
        breeze::NNUEModel p;std::string report;bool ok=p.load(argv[2],report);std::cout<<report<<'\n';if(ok){breeze::Board b;std::cout<<"Start-position SFNNv20 evaluation (Breeze cp scale): "<<p.evaluate(b)<<" cp\n";std::cout<<"Selected compact stack: "<<p.layerStackFor(b)<<"\n";}return ok?0:1;
    }
    if(argc>=2 && std::string(argv[1])=="--perft") {
        int d=argc>=3?std::stoi(argv[2]):4;breeze::Board b;auto start=std::chrono::steady_clock::now();auto n=breeze::perft(b,d);auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();std::cout<<"depth "<<d<<" nodes "<<n<<" time "<<ms<<" ms\n";return 0;
    }
    breeze::UCI uci;return uci.loop();
}
