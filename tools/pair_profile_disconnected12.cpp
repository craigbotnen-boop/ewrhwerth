#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using u64=uint64_t; using u32=uint32_t; using u16=uint16_t; using u128=unsigned __int128;
static constexpr int N=12;

static inline u64 mix64(u64 x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline u64 combine(u64 h,u64 x){ return mix64(h ^ (x + 0x9e3779b97f4a7c15ULL + (h<<6)+(h>>2))); }

struct Graph { u128 edge=0; u32 row[N]{}; };

static bool parse_g6(const std::string&s, Graph&g){
    if(s.size()!=12 || (unsigned char)s[0] != N+63) return false;
    g=Graph(); int pos=0, ci=1, remain=0; unsigned val=0;
    auto bit=[&]()->int{
        if(remain==0){ if(ci>=(int)s.size()) return -1; val=(unsigned char)s[ci++]-63; remain=6; }
        int b=(val>>(remain-1))&1U; --remain; return b;
    };
    for(int j=1;j<N;++j) for(int i=0;i<j;++i){
        int b=bit(); if(b<0) return false;
        if(b){ g.edge |= (u128(1)<<pos); g.row[i]|=1U<<j; g.row[j]|=1U<<i; }
        ++pos;
    }
    return true;
}
static Graph from_edge(u128 e){
    Graph g; g.edge=e; int pos=0;
    for(int j=1;j<N;++j) for(int i=0;i<j;++i){
        if((e>>pos)&1){g.row[i]|=1U<<j; g.row[j]|=1U<<i;} ++pos;
    }
    return g;
}
static std::string encode_g6(u128 e){
    std::string s; s.push_back(char(N+63)); int pos=0;
    for(int block=0;block<11;++block){
        unsigned v=0; for(int k=0;k<6;++k){v<<=1; if((e>>pos)&1) v|=1; ++pos;} s.push_back(char(v+63));
    }
    return s;
}
static inline int pc(u32 x){return __builtin_popcount(x);}

struct Small{
    u128 edge; u32 row[N]; uint8_t deg[N],b[N][N]; u16 d3[N],d4[N];
};
static void build_small(const Graph&g,Small&s){
    s.edge=g.edge; memcpy(s.row,g.row,sizeof(s.row));
    for(int i=0;i<N;++i)s.deg[i]=pc(s.row[i]);
    for(int i=0;i<N;++i)for(int j=0;j<N;++j)s.b[i][j]=pc(s.row[i]&s.row[j]);
    for(int i=0;i<N;++i){unsigned a3=0,a4=0;for(int k=0;k<N;++k){if((s.row[i]>>k)&1U)a3+=s.b[k][i];unsigned z=s.b[i][k];a4+=z*z;}s.d3[i]=a3;s.d4[i]=a4;}
}
struct P2M{u16 a3[N][N],a4[N][N];};
static void p2m(const Small&s,P2M&m){
    for(int i=0;i<N;++i)for(int j=0;j<N;++j){unsigned v3=0,v4=0;for(int k=0;k<N;++k){if((s.row[i]>>k)&1U)v3+=s.b[k][j];v4+=(unsigned)s.b[i][k]*s.b[k][j];}m.a3[i][j]=v3;m.a4[i][j]=v4;}
}
static int cmpd(const Small&s,int a,int b){
    if(s.deg[a]!=s.deg[b])return s.deg[a]<s.deg[b]?-1:1;
    if(s.d3[a]!=s.d3[b])return s.d3[a]<s.d3[b]?-1:1;
    if(s.d4[a]!=s.d4[b])return s.d4[a]<s.d4[b]?-1:1;
    return 0;
}
static std::pair<u64,u64> p2hash(const Small&s){
    P2M m;p2m(s,m);u64 S1=0,S2=0;
    for(int i=0;i<N;++i)for(int j=i;j<N;++j){int a=i,b=j;if(cmpd(s,a,b)>0)std::swap(a,b);
        u64 h1=0x6a09e667f3bcc909ULL,h2=0xbb67ae8584caa73bULL;
        auto add=[&](u64 x){h1=combine(h1,x);h2=combine(h2,x^0x9e3779b97f4a7c15ULL);};
        add(s.deg[a]);add(s.d3[a]);add(s.d4[a]);add(i==j);add((i==j)?0:((s.row[i]>>j)&1U));add(s.b[i][j]);add(m.a3[i][j]);add(m.a4[i][j]);add(s.deg[b]);add(s.d3[b]);add(s.d4[b]);
        S1+=mix64(h1);S2+=mix64(h2);
    }
    return {mix64(S1^N),mix64(S2^((u64)N<<40))};
}
using P2Rec=std::array<u32,11>;
static std::vector<P2Rec> p2records(u128 e){
    Small s;build_small(from_edge(e),s);P2M m;p2m(s,m);std::vector<P2Rec>v;v.reserve(78);
    for(int i=0;i<N;++i)for(int j=i;j<N;++j){int a=i,b=j;if(cmpd(s,a,b)>0)std::swap(a,b);
        v.push_back({s.deg[a],s.d3[a],s.d4[a],(u32)(i==j),(u32)((i==j)?0:((s.row[i]>>j)&1U)),s.b[i][j],m.a3[i][j],m.a4[i][j],s.deg[b],s.d3[b],s.d4[b]});
    }
    std::sort(v.begin(),v.end());return v;
}
static bool p2equal(u128 a,u128 b){return p2records(a)==p2records(b);}

struct Full{std::vector<u64> p;};
static Full powers(u128 e){
    Graph g=from_edge(e); Full F{std::vector<u64>((N+1)*N*N)};
    auto at=[&](int k,int i,int j)->u64&{return F.p[(size_t)k*N*N+i*N+j];};
    for(int i=0;i<N;++i)at(0,i,i)=1;
    for(int i=0;i<N;++i)for(int j=0;j<N;++j)at(1,i,j)=(g.row[i]>>j)&1U;
    for(int k=2;k<=N;++k)for(int i=0;i<N;++i)for(int j=0;j<N;++j){u64 z=0;for(int t=0;t<N;++t)if((g.row[i]>>t)&1U)z+=at(k-1,t,j);at(k,i,j)=z;}
    return F;
}
static int cmpdf(const Full&F,int a,int b){
    for(int k=0;k<=N;++k){u64 x=F.p[(size_t)k*N*N+a*N+a],y=F.p[(size_t)k*N*N+b*N+b];if(x!=y)return x<y?-1:1;}return 0;
}
using Rec=std::vector<u64>;
static std::vector<Rec> i3recs(const Full&F){
    std::vector<Rec>v;v.reserve(78);
    for(int i=0;i<N;++i)for(int j=i;j<N;++j){int a=i,b=j;if(cmpdf(F,a,b)>0)std::swap(a,b);Rec r;r.reserve(3*(N+1));
        for(int k=0;k<=N;++k)r.push_back(F.p[(size_t)k*N*N+a*N+a]);
        for(int k=0;k<=N;++k)r.push_back(F.p[(size_t)k*N*N+i*N+j]);
        for(int k=0;k<=N;++k)r.push_back(F.p[(size_t)k*N*N+b*N+b]);
        v.push_back(std::move(r));
    }std::sort(v.begin(),v.end());return v;
}
static std::array<u64,N> traces(const Full&F){
    std::array<u64,N>t{};for(int k=1;k<=N;++k)for(int i=0;i<N;++i)t[k-1]+=F.p[(size_t)k*N*N+i*N+i];return t;
}
static bool i3all_equal(u128 a,u128 b){
    Full A=powers(a),B=powers(b);
    if(traces(A)!=traces(B))return false; // Newton identities => same characteristic polynomial.
    return i3recs(A)==i3recs(B);           // includes k=0..12; hence same recurrence continuation.
}
static std::vector<Rec> i4rows(const Full&F){
    std::vector<Rec> rows;rows.reserve(N);
    for(int x=0;x<N;++x){std::vector<Rec>inner;inner.reserve(N);
        for(int y=0;y<N;++y){Rec r;r.reserve(2*(N+1));for(int k=0;k<=N;++k)r.push_back(F.p[(size_t)k*N*N+x*N+y]);for(int k=0;k<=N;++k)r.push_back(F.p[(size_t)k*N*N+y*N+y]);inner.push_back(std::move(r));}
        std::sort(inner.begin(),inner.end());Rec row;row.reserve((N+1)+N*2*(N+1));for(int k=0;k<=N;++k)row.push_back(F.p[(size_t)k*N*N+x*N+x]);for(auto&r:inner)row.insert(row.end(),r.begin(),r.end());rows.push_back(std::move(row));
    }std::sort(rows.begin(),rows.end());return rows;
}
static bool i4equal(u128 a,u128 b){return i4rows(powers(a))==i4rows(powers(b));}

struct Slot{u64 h1=0,h2=0,lo=0,hi=UINT64_MAX;};
int main(){
    const size_t SZ=1ULL<<24, MASK=SZ-1; std::vector<Slot>T(SZ);
    std::string line;u64 n=0,p2c=0,i3c=0;Graph g;
    while(std::getline(std::cin,line)){
        if(!parse_g6(line,g)){std::cerr<<"BAD_GRAPH6 at "<<n<<"\n";return 2;}
        for(int i=0;i<N;++i)if(pc(g.row[i])==0){std::cerr<<"ISOLATE_PRESENT at "<<n<<"\n";return 3;}
        Small s;build_small(g,s);auto h=p2hash(s);u64 lo=(u64)g.edge,hi=(u64)(g.edge>>64);
        size_t pos=h.first&MASK, probes=0;
        for(;;){
            Slot &q=T[pos];
            if(q.hi==UINT64_MAX){q.h1=h.first;q.h2=h.second;q.lo=lo;q.hi=hi;break;}
            if(q.h1==h.first && q.h2==h.second){
                u128 old=(u128(q.hi)<<64)|q.lo;
                if(!p2equal(old,g.edge)){std::cerr<<"DOUBLE_HASH_COLLISION_FAIL_CLOSED\n";return 4;}
                ++p2c;
                if(!i3all_equal(old,g.edge)){
                    std::cerr<<"P2_GROUP_CONTAINS_MULTIPLE_FULL_I3_CLASSES_FAIL_CLOSED\n";
                    std::cerr<<"A="<<encode_g6(old)<<" B="<<line<<"\n";return 5;
                }
                ++i3c;
                if(!i4equal(old,g.edge)){
                    std::cout<<"WITNESS_FOUND n=12 disconnected_no_isolates\n";
                    std::cout<<"graph6_A="<<encode_g6(old)<<"\n";
                    std::cout<<"graph6_B="<<line<<"\n";
                    std::cout<<"processed="<<(n+1)<<" p2_collisions="<<p2c<<" full_i3_collisions="<<i3c<<"\n";
                    return 0;
                }
                break;
            }
            pos=(pos+1)&MASK;if(++probes>SZ/2){std::cerr<<"TABLE_SATURATED\n";return 6;}
        }
        ++n;if(n%1000000ULL==0)std::cerr<<"processed="<<n<<" p2_collisions="<<p2c<<" full_i3_collisions="<<i3c<<"\n";
    }
    std::cout<<"SCAN_COMPLETE graphs="<<n<<" p2_collisions="<<p2c<<" full_i3_collisions="<<i3c<<" WITNESS_NONE\n";
    if(n!=12344252ULL){std::cerr<<"COUNT_MISMATCH expected=12344252 got="<<n<<"\n";return 7;}
    return 0;
}
