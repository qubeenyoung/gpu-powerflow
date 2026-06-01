// Probe: is cuDSS execute (factorize/solve) capturable into a CUDA graph via
// stream capture? Decides the cuGraph design (whether cuDSS can live inside the
// captured iteration graph or must stay as a normal launch outside it).
//
// Sets up a tiny 3x3 SPD-ish system, runs analyze+factorize normally, then
// attempts to stream-capture a SOLVE-phase cudssExecute. Reports capture status.
#include <cudss.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); } }while(0)
#define DK(x) do{ cudssStatus_t s=(x); if(s!=CUDSS_STATUS_SUCCESS){printf("cuDSS err %s:%d status=%d\n",__FILE__,__LINE__,(int)s);} }while(0)

int main()
{
    int n = 3, nnz = 3;  // diagonal matrix [2,3,4]
    int h_rp[4] = {0,1,2,3}, h_ci[3] = {0,1,2};
    double h_v[3] = {2,3,4}, h_b[3] = {2,6,12};  // x = [1,2,3]
    int *d_rp,*d_ci; double *d_v,*d_b,*d_x;
    CK(cudaMalloc(&d_rp,sizeof h_rp)); CK(cudaMalloc(&d_ci,sizeof h_ci));
    CK(cudaMalloc(&d_v,sizeof h_v)); CK(cudaMalloc(&d_b,sizeof h_b)); CK(cudaMalloc(&d_x,sizeof h_b));
    CK(cudaMemcpy(d_rp,h_rp,sizeof h_rp,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci,h_ci,sizeof h_ci,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v,h_v,sizeof h_v,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b,h_b,sizeof h_b,cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudssHandle_t h; DK(cudssCreate(&h)); DK(cudssSetStream(h,stream));
    cudssConfig_t cfg; DK(cudssConfigCreate(&cfg));
    cudssData_t data; DK(cudssDataCreate(h,&data));

    cudssMatrix_t A,X,B;
    DK(cudssMatrixCreateCsr(&A,n,n,nnz,d_rp,nullptr,d_ci,d_v,CUDA_R_32I,CUDA_R_64F,
                            CUDSS_MTYPE_GENERAL,CUDSS_MVIEW_FULL,CUDSS_BASE_ZERO));
    DK(cudssMatrixCreateDn(&X,n,1,n,d_x,CUDA_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    DK(cudssMatrixCreateDn(&B,n,1,n,d_b,CUDA_R_64F,CUDSS_LAYOUT_COL_MAJOR));

    DK(cudssExecute(h,CUDSS_PHASE_ANALYSIS,cfg,data,A,X,B));
    DK(cudssExecute(h,CUDSS_PHASE_FACTORIZATION,cfg,data,A,X,B));
    CK(cudaStreamSynchronize(stream));

    // Attempt to capture a SOLVE into a graph.
    printf("=== attempting stream capture of cudssExecute(SOLVE) ===\n");
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    cudssStatus_t solve_status = cudssExecute(h,CUDSS_PHASE_SOLVE,cfg,data,A,X,B);
    cudaGraph_t graph = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
    printf("cudssExecute(SOLVE) returned status=%d\n", (int)solve_status);
    printf("cudaStreamEndCapture err=%d (%s)\n", (int)end_err,
           end_err==cudaSuccess?"OK":cudaGetErrorString(end_err));

    if (end_err == cudaSuccess && graph) {
        size_t numNodes=0; CK(cudaGraphGetNodes(graph,nullptr,&numNodes));
        printf("captured graph node count = %zu\n", numNodes);
        cudaGraphExec_t exec; cudaError_t ie = cudaGraphInstantiate(&exec,graph,0);
        printf("cudaGraphInstantiate err=%d\n",(int)ie);
        if (ie==cudaSuccess){
            CK(cudaGraphLaunch(exec,stream)); CK(cudaStreamSynchronize(stream));
            std::vector<double> x(3); CK(cudaMemcpy(x.data(),d_x,sizeof h_b,cudaMemcpyDeviceToHost));
            printf("graph-replayed solve x = [%g %g %g] (expect 1 2 3)\n",x[0],x[1],x[2]);
        }
        printf(">>> RESULT: cuDSS SOLVE is CAPTURABLE\n");
    } else {
        printf(">>> RESULT: cuDSS SOLVE is NOT capturable (capture aborted)\n");
    }
    return 0;
}
