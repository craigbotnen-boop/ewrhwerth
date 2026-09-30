#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using u64=uint64_t; using u32=uint32_t; using u16=uint16_t;
static constexpr int MAXN=17;

static inline u64 mix64(u64 x){
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static inline u64 combine(u64 h,u64 x){ return mix64(h ^ (x + 0x9e3779b97f4a7c15ULL + (h<<6)+(h>>2))); }

struct Parsed { int n=0; u64 edge=0; u32 row[MAXN]{}; };

static bool parse_g6(const std::string&s, Parsed&g){
    if(s.empty()) return false;
    unsigned char c=s[0];
    if(c<63 || c>125) return false;
    int n=(int)c-63;
    if(n<1 || n>17) return false;
    g=Parsed(); g.n=n;
    int bitpos=0, ci=1, remain=0; unsigned val=0;
    auto nextbit=[&]()->int{
        if(remain==0){ if(ci>=(int)s.size()) return -1; val=((unsigned char)s[ci++])-63; remain=6; }
        int b=(val>>(remain-1))&1U; --remain; return b;
    };
    for(int j=1;j<n;++j) for(int i=0;i<j;++i){
        int b=nextbit(); if(b<0) return false;
        if(b){ g.row[i]|=(1U<<j); g.row[j]|=(1U<<i); g.edge|=(1ULL<<bitpos); }
        ++bitpos;
    }
    return true;
}

static std::string encode_g6(u64 edge,int n){
    std::string s; s.push_back(char(n+63));
    int total=n*(n-1)/2, pos=0;
    while(pos<total){ unsigned v=0; for(int k=0;k<6;++k){v<<=1; if(pos<total && ((edge>>pos)&1ULL)) v|=1; ++pos;} s.push_back(char(v+63)); }
    return s;
}

struct Small {
    int n; u64 edge; u32 row[MAXN]; uint8_t deg[MAXN]; uint8_t b[MAXN][MAXN];
    u16 d3[MAXN]; u16 d4[MAXN];
};
static inline int pc32(u32 x){ return __builtin_popcount(x); }
static void build_small(const Parsed&g, Small&s){
    s.n=g.n; s.edge=g.edge; memcpy(s.row,g.row,sizeof(s.row));
    int n=s.n;
    for(int i=0;i<n;++i) s.deg[i]=pc32(s.row[i]);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) s.b[i][j]=pc32(s.row[i]&s.row[j]);
    for(int i=0;i<n;++i){
        unsigned a3=0,a4=0; u32 r=s.row[i];
        for(int k=0;k<n;++k){ if((r>>k)&1U) a3 += s.b[k][i]; unsigned z=s.b[i][k]; a4 += z*z; }
        s.d3[i]=(u16)a3; s.d4[i]=(u16)a4;
    }
}
static void build_small_edge(u64 edge,int n,Small&s){
    Parsed g; g.n=n; g.edge=edge; int pos=0;
    for(int j=1;j<n;++j) for(int i=0;i<j;++i){ if((edge>>pos)&1ULL){g.row[i]|=1U<<j; g.row[j]|=1U<<i;} ++pos; }
    build_small(g,s);
}

static std::pair<u64,u64> u2_hash(const Small&s){
    u64 h1=0x123456789abcdef0ULL, h2=0xfedcba9876543210ULL; int n=s.n;
    for(int i=0;i<n;++i) for(int j=i;j<n;++j){
        unsigned da=s.deg[i], db=s.deg[j]; if(da>db) std::swap(da,db);
        unsigned delta=(i==j), adj= delta?0:((s.row[i]>>j)&1U), common=s.b[i][j];
        u64 code=delta | ((u64)adj<<1) | ((u64)common<<2) | ((u64)da<<7) | ((u64)db<<11);
        h1 += mix64(code ^ 0x243f6a8885a308d3ULL);
        h2 += mix64(code ^ 0x13198a2e03707344ULL);
    }
    return {mix64(h1 ^ n), mix64(h2 ^ (u64)n<<32)};
}

static inline int cmp_diag4(const Small&s,int a,int b){
    if(s.deg[a]!=s.deg[b]) return s.deg[a]<s.deg[b]?-1:1;
    if(s.d3[a]!=s.d3[b]) return s.d3[a]<s.d3[b]?-1:1;
    if(s.d4[a]!=s.d4[b]) return s.d4[a]<s.d4[b]?-1:1;
    return 0;
}
struct P2Mats { u16 a3[MAXN][MAXN]; u16 a4[MAXN][MAXN]; };
static void build_p2m(const Small&s,P2Mats&m){
    int n=s.n;
    for(int i=0;i<n;++i) for(int j=0;j<n;++j){
        unsigned v3=0,v4=0; u32 ri=s.row[i];
        for(int k=0;k<n;++k){ if((ri>>k)&1U) v3 += s.b[k][j]; v4 += (unsigned)s.b[i][k]*(unsigned)s.b[k][j]; }
        m.a3[i][j]=(u16)v3; m.a4[i][j]=(u16)v4;
    }
}
static u64 diag4_hash(const Small&s,int i,u64 seed){
    u64 h=seed; h=combine(h,s.deg[i]); h=combine(h,s.d3[i]); h=combine(h,s.d4[i]); return h;
}
static std::pair<u64,u64> p2_hash(const Small&s){
    P2Mats m; build_p2m(s,m); int n=s.n; u64 S1=0,S2=0;
    for(int i=0;i<n;++i) for(int j=i;j<n;++j){
        int a=i,b=j; if(cmp_diag4(s,a,b)>0) std::swap(a,b);
        u64 r1=0x6a09e667f3bcc909ULL, r2=0xbb67ae8584caa73bULL;
        auto add=[&](u64 x){r1=combine(r1,x); r2=combine(r2,x^0x9e3779b97f4a7c15ULL);};
        add(s.deg[a]); add(s.d3[a]); add(s.d4[a]);
        add(i==j); add((i==j)?0:((s.row[i]>>j)&1U)); add(s.b[i][j]); add(m.a3[i][j]); add(m.a4[i][j]);
        add(s.deg[b]); add(s.d3[b]); add(s.d4[b]);
        S1 += mix64(r1); S2 += mix64(r2);
    }
    return {mix64(S1^n),mix64(S2^((u64)n<<40))};
}
static std::pair<u64,u64> q2_hash(const Small&s){
    u64 S1=0,S2=0; for(int i=0;i<s.n;++i){
        u64 r1=diag4_hash(s,i,0x510e527fade682d1ULL), r2=diag4_hash(s,i,0x9b05688c2b3e6c1fULL);
        S1+=mix64(r1); S2+=mix64(r2);
    } return {mix64(S1^s.n),mix64(S2)};
}

struct P2RecExact { std::array<u32,11> a; bool operator<(P2RecExact const&o)const{return a<o.a;} bool operator==(P2RecExact const&o)const{return a==o.a;} };
static std::vector<P2RecExact> p2_exact_records(u64 edge,int n){
    Small s; build_small_edge(edge,n,s); P2Mats m; build_p2m(s,m); std::vector<P2RecExact> v; v.reserve(n*(n+1)/2);
    for(int i=0;i<n;++i) for(int j=i;j<n;++j){ int a=i,b=j; if(cmp_diag4(s,a,b)>0) std::swap(a,b); P2RecExact r{};
        r.a={s.deg[a],s.d3[a],s.d4[a],(u32)(i==j),(u32)((i==j)?0:((s.row[i]>>j)&1U)),s.b[i][j],m.a3[i][j],m.a4[i][j],s.deg[b],s.d3[b],s.d4[b]}; v.push_back(r);
    } std::sort(v.begin(),v.end()); return v;
}
static bool exact_p2_equal(u64 e1,u64 e2,int n){return p2_exact_records(e1,n)==p2_exact_records(e2,n);}
static std::vector<std::array<u32,3>> q2_exact(u64 edge,int n){ Small s;build_small_edge(edge,n,s);std::vector<std::array<u32,3>>v;for(int i=0;i<n;++i)v.push_back({s.deg[i],s.d3[i],s.d4[i]});std::sort(v.begin(),v.end());return v;}
static bool exact_q2_equal(u64 e1,u64 e2,int n){return q2_exact(e1,n)==q2_exact(e2,n);}

struct Full {int n; std::vector<u64> p;};
static Full full_powers(u64 edge,int n){
    Small s;build_small_edge(edge,n,s); Full F{n,std::vector<u64>((size_t)n*n*n)};
    auto at=[&](int k,int i,int j)->u64&{return F.p[(size_t)k*n*n+i*n+j];};
    for(int i=0;i<n;++i) at(0,i,i)=1;
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) at(1,i,j)=(s.row[i]>>j)&1U;
    for(int k=2;k<n;++k) for(int i=0;i<n;++i) for(int j=0;j<n;++j){u64 z=0;u32 ri=s.row[i];for(int t=0;t<n;++t)if((ri>>t)&1U)z+=at(k-1,t,j);at(k,i,j)=z;}
    return F;
}
static int cmp_diag_full(const Full&F,int a,int b){int n=F.n;for(int k=0;k<n;++k){u64 x=F.p[(size_t)k*n*n+a*n+a],y=F.p[(size_t)k*n*n+b*n+b];if(x!=y)return x<y?-1:1;}return 0;}
static std::pair<u64,u64> full_hash_from(const Full&F){int n=F.n;u64 S1=0,S2=0;for(int i=0;i<n;++i)for(int j=i;j<n;++j){int a=i,b=j;if(cmp_diag_full(F,a,b)>0)std::swap(a,b);u64 r1=0x1f83d9abfb41bd6bULL,r2=0x5be0cd19137e2179ULL;for(int k=0;k<n;++k){r1=combine(r1,F.p[(size_t)k*n*n+a*n+a]);r2=combine(r2,F.p[(size_t)k*n*n+a*n+a]^0x1111111111111111ULL);}for(int k=0;k<n;++k){u64 x=F.p[(size_t)k*n*n+i*n+j];r1=combine(r1,x);r2=combine(r2,x^0x2222222222222222ULL);}for(int k=0;k<n;++k){u64 x=F.p[(size_t)k*n*n+b*n+b];r1=combine(r1,x);r2=combine(r2,x^0x3333333333333333ULL);}S1+=mix64(r1);S2+=mix64(r2);}return{mix64(S1^n),mix64(S2)};}
static std::pair<u64,u64> full_hash(u64 edge,int n){return full_hash_from(full_powers(edge,n));}
using FullRec=std::vector<u64>;
static std::vector<FullRec> full_records(const Full&F){int n=F.n;std::vector<FullRec>v;v.reserve(n*(n+1)/2);for(int i=0;i<n;++i)for(int j=i;j<n;++j){int a=i,b=j;if(cmp_diag_full(F,a,b)>0)std::swap(a,b);FullRec r;r.reserve(3*n);for(int k=0;k<n;++k)r.push_back(F.p[(size_t)k*n*n+a*n+a]);for(int k=0;k<n;++k)r.push_back(F.p[(size_t)k*n*n+i*n+j]);for(int k=0;k<n;++k)r.push_back(F.p[(size_t)k*n*n+b*n+b]);v.push_back(std::move(r));}std::sort(v.begin(),v.end());return v;}
static bool exact_full_equal(u64 e1,u64 e2,int n){auto A=full_powers(e1,n),B=full_powers(e2,n);return full_records(A)==full_records(B);}
static std::vector<FullRec> i4_rows(const Full&F){int n=F.n;std::vector<FullRec>rows;rows.reserve(n);for(int x=0;x<n;++x){std::vector<FullRec> inner;inner.reserve(n);for(int y=0;y<n;++y){FullRec r;r.reserve(2*n);for(int k=0;k<n;++k)r.push_back(F.p[(size_t)k*n*n+x*n+y]);for(int k=0;k<n;++k)r.push_back(F.p[(size_t)k*n*n+y*n+y]);inner.push_back(std::move(r));}std::sort(inner.begin(),inner.end());FullRec row;row.reserve(n+2*n*n);for(int k=0;k<n;++k)row.push_back(F.p[(size_t)k*n*n+x*n+x]);for(auto &r:inner)row.insert(row.end(),r.begin(),r.end());rows.push_back(std::move(row));}std::sort(rows.begin(),rows.end());return rows;}
static bool exact_i4_equal(u64 e1,u64 e2,int n){return i4_rows(full_powers(e1,n))==i4_rows(full_powers(e2,n));}

struct Bloom{u64 bits;int k;std::vector<u64>a;Bloom(){}Bloom(int exp,int kk):bits(1ULL<<exp),k(kk),a((1ULL<<exp)/64){};bool test(u64 h1,u64 h2)const{u64 mask=bits-1;h2|=1;for(int i=0;i<k;++i){u64 p=(h1+(u64)i*h2)&mask;if(!(a[p>>6]&(1ULL<<(p&63))))return false;}return true;}void add(u64 h1,u64 h2){u64 mask=bits-1;h2|=1;for(int i=0;i<k;++i){u64 p=(h1+(u64)i*h2)&mask;a[p>>6]|=1ULL<<(p&63);}}u64 setbits()const{u64 z=0;for(u64 x:a)z+=__builtin_popcountll(x);return z;}void save(const char*fn)const{std::ofstream f(fn,std::ios::binary);u64 magic=0x424c4f4f4d763031ULL;f.write((char*)&magic,8);f.write((char*)&bits,8);f.write((char*)&k,4);f.write((char*)a.data(),a.size()*8);}static Bloom load(const char*fn){std::ifstream f(fn,std::ios::binary);u64 magic,bits;int k;f.read((char*)&magic,8);f.read((char*)&bits,8);f.read((char*)&k,4);if(magic!=0x424c4f4f4d763031ULL){std::cerr<<"bad bloom\n";exit(2);}int exp=0;while((1ULL<<exp)<bits)++exp;Bloom b(exp,k);f.read((char*)b.a.data(),b.a.size()*8);return b;}};

struct Table{size_t sz,mask;std::vector<u64> h,edge;std::vector<u32> count;Table(int exp,bool counts):sz(1ULL<<exp),mask(sz-1),h(sz),edge(sz,UINT64_MAX),count(counts?sz:0){};};

static void progress(u64 c,const char*tag){if(c%10000000ULL==0){std::cerr<<tag<<" graphs="<<c<<"\n";}}

static int bench10(){
    Table ptab(24,true),qtab(24,true);u64 N=0,p_groups=0,q_groups=0,p_pairs=0;std::string line;Parsed g;
    while(std::getline(std::cin,line)){if(!parse_g6(line,g)||g.n!=10){std::cerr<<"bad graph6\n";return 2;}Small s;build_small(g,s);auto ph=p2_hash(s);auto qh=q2_hash(s);
        auto ins=[&](Table&T,u64 hv,bool isp2,u64 &groups,u64 &pairs){size_t pos=hv&T.mask;for(;;){if(T.edge[pos]==UINT64_MAX){T.h[pos]=hv;T.edge[pos]=g.edge;T.count[pos]=1;return;}if(T.h[pos]==hv){bool eq=isp2?exact_p2_equal(T.edge[pos],g.edge,10):exact_q2_equal(T.edge[pos],g.edge,10);if(eq){if(T.count[pos]==1)++groups;++T.count[pos];++pairs;return;}}pos=(pos+1)&T.mask;}};
        ins(ptab,ph.first,true,p_groups,p_pairs);u64 dummy=0;ins(qtab,qh.first,false,q_groups,dummy);++N;progress(N,"bench10");
    }
    std::cout<<"BENCH10 graphs="<<N<<" p2_collision_groups="<<p_groups<<" p2_duplicate_graphs="<<p_pairs<<" q2_collision_groups="<<q_groups<<"\n";
    bool ok=N==12005168ULL && p_groups==0 && q_groups==8874ULL; std::cout<<(ok?"BENCH10_PASS":"BENCH10_FAIL")<<"\n";return ok?0:3;
}

static int phase1(const char*out){
    Bloom once(33,5),rep(32,4);u64 N=0,trig=0;std::string line;Parsed g;while(std::getline(std::cin,line)){if(!parse_g6(line,g)||g.n!=11)return 2;Small s;build_small(g,s);auto h=u2_hash(s);if(once.test(h.first,h.second)){rep.add(h.first,h.second);++trig;}else once.add(h.first,h.second);++N;progress(N,"phase1");}rep.save(out);std::cout<<"PHASE1 graphs="<<N<<" repeat_triggers="<<trig<<" rep_setbits="<<rep.setbits()<<" rep_fraction="<<(double)rep.setbits()/rep.bits<<"\n";return 0;}
static int phase2(const char*u2file,const char*out){
    Bloom u2=Bloom::load(u2file),once(32,4),rep(31,4);u64 N=0,sel=0,trig=0;std::string line;Parsed g;while(std::getline(std::cin,line)){if(!parse_g6(line,g)||g.n!=11)return 2;Small s;build_small(g,s);auto h=u2_hash(s);if(u2.test(h.first,h.second)){++sel;auto p=p2_hash(s);if(once.test(p.first,p.second)){rep.add(p.first,p.second);++trig;}else once.add(p.first,p.second);}++N;progress(N,"phase2");}rep.save(out);std::cout<<"PHASE2 graphs="<<N<<" u2_selected="<<sel<<" p2_repeat_triggers="<<trig<<" p2rep_setbits="<<rep.setbits()<<" p2rep_fraction="<<(double)rep.setbits()/rep.bits<<"\n";return 0;}
static int phase3(const char*u2file,const char*p2file,const char*result){
    Bloom u2=Bloom::load(u2file),p2rep=Bloom::load(p2file);Table tab(24,false);u64 N=0,u2sel=0,p2sel=0,fullcoll=0;std::string line;Parsed g;std::ofstream out(result);bool found=false;
    while(std::getline(std::cin,line)){if(!parse_g6(line,g)||g.n!=11)return 2;if(found){++N;continue;}Small s;build_small(g,s);auto uh=u2_hash(s);if(u2.test(uh.first,uh.second)){++u2sel;auto ph=p2_hash(s);if(p2rep.test(ph.first,ph.second)){++p2sel;auto fh=full_hash(g.edge,11);size_t pos=fh.first&tab.mask;size_t probes=0;for(;;){if(tab.edge[pos]==UINT64_MAX){tab.h[pos]=fh.first;tab.edge[pos]=g.edge;break;}if(tab.h[pos]==fh.first && exact_full_equal(tab.edge[pos],g.edge,11)){++fullcoll;if(!exact_i4_equal(tab.edge[pos],g.edge,11)){std::string a=encode_g6(tab.edge[pos],11),b=encode_g6(g.edge,11);std::cout<<"WITNESS_FOUND n=11 graph6_A="<<a<<" graph6_B="<<b<<"\n";out<<"WITNESS_FOUND\nn=11\ngraph6_A="<<a<<"\ngraph6_B="<<b<<"\n";out.flush();return 0;}break;}pos=(pos+1)&tab.mask;if(++probes>tab.sz/2){std::cerr<<"table saturated\n";return 7;}}
        }}++N;progress(N,"phase3");}
    if(!found){std::cout<<"PHASE3 graphs="<<N<<" u2_selected="<<u2sel<<" p2_selected="<<p2sel<<" full_i3_collisions="<<fullcoll<<" WITNESS_NONE\n";out<<"WITNESS_NONE\nn=11\ngraphs="<<N<<"\nfull_i3_collisions="<<fullcoll<<"\n";}return 0;
}

static int selftest(){
    Parsed a,b; if(!parse_g6("BW",a)||!parse_g6("Bw",b)) return 2;
    bool ok=(a.n==3 && b.n==3 && __builtin_popcountll(a.edge)==2 && __builtin_popcountll(b.edge)==3 && encode_g6(a.edge,3)=="BW" && encode_g6(b.edge,3)=="Bw");
    u64 e=0; for(int k=0;k<55;k+=3) e|=1ULL<<k; std::string z=encode_g6(e,11); Parsed c; ok=ok&&parse_g6(z,c)&&c.edge==e&&c.n==11;
    std::cout<<"SELFTEST "<<(ok?"PASS":"FAIL")<<"\n"; return ok?0:5;
}

int main(int argc,char**argv){if(argc<2){std::cerr<<"modes: selftest bench10 phase1 OUT phase2 U2 OUT phase3 U2 P2 RESULT\n";return 2;}std::string m=argv[1];if(m=="selftest")return selftest();if(m=="bench10")return bench10();if(m=="phase1"&&argc==3)return phase1(argv[2]);if(m=="phase2"&&argc==4)return phase2(argv[2],argv[3]);if(m=="phase3"&&argc==5)return phase3(argv[2],argv[3],argv[4]);return 2;}
