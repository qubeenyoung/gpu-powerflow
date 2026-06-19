// cuDSS harness: fp32/fp64 + UBATCH, warmup + per-iter MEDIAN (symmetric with custom runner).
// usage: cudss_bench2 A.mtx [repeat] [B] [fp32|fp64] [warmup]   factor=REFACTORIZATION.
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cudss.h>
using clk=std::chrono::high_resolution_clock;
static double ms(clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static double median(std::vector<double>&v){std::sort(v.begin(),v.end());size_t n=v.size();return n? (n&1?v[n/2]:0.5*(v[n/2-1]+v[n/2])):0.0;}
#define CU(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
#define DS(x) do{cudssStatus_t s=(x);if(s!=CUDSS_STATUS_SUCCESS){std::printf("cuDSS %s:%d status=%d\n",__FILE__,__LINE__,(int)s);std::exit(1);}}while(0)
struct CSR{int n=0;long nnz=0;std::vector<int>rp,ci;std::vector<double>v;};
static bool read_mm(const std::string&path,CSR&A){
  std::ifstream f(path); if(!f){std::printf("open failed: %s\n",path.c_str());return false;}
  std::string line; std::getline(f,line);
  bool pat=line.find("pattern")!=std::string::npos;
  bool sym=line.find("symmetric")!=std::string::npos||line.find("hermitian")!=std::string::npos;
  while(std::getline(f,line)) if(!line.empty()&&line[0]!='%') break;
  int nr,nc; long nz; {std::istringstream is(line); is>>nr>>nc>>nz;}
  std::vector<int>I,J; std::vector<double>V;
  for(long k=0;k<nz;++k){std::getline(f,line);std::istringstream is(line);int i,j;double val=1.0;is>>i>>j;if(!pat)is>>val;--i;--j;
    I.push_back(i);J.push_back(j);V.push_back(val); if(sym&&i!=j){I.push_back(j);J.push_back(i);V.push_back(val);}}
  A.n=nr;A.nnz=(long)I.size();A.rp.assign(nr+1,0);
  for(long k=0;k<A.nnz;++k)A.rp[I[k]+1]++; for(int i=0;i<nr;++i)A.rp[i+1]+=A.rp[i];
  A.ci.resize(A.nnz);A.v.resize(A.nnz);std::vector<int>cur(A.rp.begin(),A.rp.end()-1);
  for(long k=0;k<A.nnz;++k){int r=I[k];int d=cur[r]++;A.ci[d]=J[k];A.v[d]=V[k];}
  return true;
}
int main(int argc,char**argv){
  if(argc<2){std::printf("usage: %s A.mtx [repeat] [B] [fp32|fp64] [warmup]\n",argv[0]);return 1;}
  std::string file=argv[1]; int repeat=(argc>2)?std::atoi(argv[2]):10; int B=(argc>3)?std::atoi(argv[3]):1;
  bool fp32=(argc>4)&&std::strcmp(argv[4],"fp32")==0; int W=(argc>5)?std::atoi(argv[5]):5;
  cudaDataType vt=fp32?CUDA_R_32F:CUDA_R_64F; size_t es=fp32?sizeof(float):sizeof(double);
  CSR A; if(!read_mm(file,A))return 1; int n=A.n; long nnz=A.nnz;
  int*d_rp,*d_ci; CU(cudaMalloc(&d_rp,(n+1)*sizeof(int))); CU(cudaMalloc(&d_ci,nnz*sizeof(int)));
  CU(cudaMemcpy(d_rp,A.rp.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
  CU(cudaMemcpy(d_ci,A.ci.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
  void*d_v,*d_b,*d_x; CU(cudaMalloc(&d_v,(size_t)B*nnz*es)); CU(cudaMalloc(&d_b,(size_t)B*n*es)); CU(cudaMalloc(&d_x,(size_t)B*n*es));
  CU(cudaMemset(d_x,0,(size_t)B*n*es));
  if(fp32){std::vector<float>vv((size_t)B*nnz),bb((size_t)B*n,1.f);
    for(int k=0;k<B;++k)for(long t=0;t<nnz;++t)vv[(size_t)k*nnz+t]=(float)A.v[t];
    CU(cudaMemcpy(d_v,vv.data(),(size_t)B*nnz*es,cudaMemcpyHostToDevice));CU(cudaMemcpy(d_b,bb.data(),(size_t)B*n*es,cudaMemcpyHostToDevice));}
  else{std::vector<double>vv((size_t)B*nnz),bb((size_t)B*n,1.0);
    for(int k=0;k<B;++k)for(long t=0;t<nnz;++t)vv[(size_t)k*nnz+t]=A.v[t];
    CU(cudaMemcpy(d_v,vv.data(),(size_t)B*nnz*es,cudaMemcpyHostToDevice));CU(cudaMemcpy(d_b,bb.data(),(size_t)B*n*es,cudaMemcpyHostToDevice));}
  cudssHandle_t h; DS(cudssCreate(&h)); cudssConfig_t cfg; DS(cudssConfigCreate(&cfg)); cudssData_t dat; DS(cudssDataCreate(h,&dat));
  if(B>1){int ub=B; DS(cudssConfigSet(cfg,CUDSS_CONFIG_UBATCH_SIZE,&ub,sizeof(ub)));}
  cudssMatrix_t Am,Bm,Xm;
  DS(cudssMatrixCreateCsr(&Am,n,n,nnz,d_rp,nullptr,d_ci,d_v,CUDA_R_32I,vt,CUDSS_MTYPE_GENERAL,CUDSS_MVIEW_FULL,CUDSS_BASE_ZERO));
  DS(cudssMatrixCreateDn(&Bm,n,1,n,d_b,vt,CUDSS_LAYOUT_COL_MAJOR));
  DS(cudssMatrixCreateDn(&Xm,n,1,n,d_x,vt,CUDSS_LAYOUT_COL_MAJOR));
  DS(cudssExecute(h,CUDSS_PHASE_ANALYSIS,cfg,dat,Am,Xm,Bm)); DS(cudssExecute(h,CUDSS_PHASE_FACTORIZATION,cfg,dat,Am,Xm,Bm)); DS(cudssExecute(h,CUDSS_PHASE_SOLVE,cfg,dat,Am,Xm,Bm)); CU(cudaDeviceSynchronize());
  // warmup (W untimed refactor+solve), then per-iter MEDIAN
  for(int r=0;r<W;++r){ DS(cudssExecute(h,CUDSS_PHASE_REFACTORIZATION,cfg,dat,Am,Xm,Bm)); DS(cudssExecute(h,CUDSS_PHASE_SOLVE,cfg,dat,Am,Xm,Bm)); }
  CU(cudaDeviceSynchronize());
  std::vector<double> ff(repeat),ss(repeat);
  for(int r=0;r<repeat;++r){ CU(cudaDeviceSynchronize()); auto a=clk::now(); DS(cudssExecute(h,CUDSS_PHASE_REFACTORIZATION,cfg,dat,Am,Xm,Bm)); CU(cudaDeviceSynchronize()); ff[r]=ms(a,clk::now()); }
  for(int r=0;r<repeat;++r){ CU(cudaDeviceSynchronize()); auto a=clk::now(); DS(cudssExecute(h,CUDSS_PHASE_SOLVE,cfg,dat,Am,Xm,Bm)); CU(cudaDeviceSynchronize()); ss[r]=ms(a,clk::now()); }
  double fac=median(ff)/B, sol=median(ss)/B;
  double rn=0,bn=0; std::vector<double>x(n);
  if(fp32){std::vector<float>xf(n);CU(cudaMemcpy(xf.data(),d_x,n*es,cudaMemcpyDeviceToHost));for(int i=0;i<n;++i)x[i]=xf[i];}
  else CU(cudaMemcpy(x.data(),d_x,n*es,cudaMemcpyDeviceToHost));
  for(int i=0;i<n;++i){double ax=0;for(int k=A.rp[i];k<A.rp[i+1];++k)ax+=A.v[k]*x[A.ci[k]];double rr=1.0-ax;rn+=rr*rr;bn+=1.0;}
  std::printf("solver=cuDSS prec=%s B=%d W=%d\nfactor_per_sys_ms=%.6f\nsolve_per_sys_ms=%.6f\nrelres=%.3e\n",fp32?"fp32":"fp64",B,W,fac,sol,std::sqrt(rn)/std::sqrt(bn));
  return 0;
}
