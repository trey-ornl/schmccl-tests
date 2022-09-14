#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <hip/hip_runtime.h>
#include <libgen.h>
#include <mpi.h>
#include <unistd.h>

static void checkHip(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

#define CHECK(X) checkHip(X,__FILE__,__LINE__)

static long getBytes(const char *const s)
{
  char c = 0;
  double value = 0;
  const int n = sscanf(s,"%lf %c",&value,&c);
  double units = 1;
  if (n == 2) {
    switch(c) {
      case 'G':
      case 'g':
        units *= 1024;
      case 'M':
      case 'm':
        units *= 1024;
      case 'K':
      case 'k':
        units *= 1024;
    }
  }
  return long(value*units);
}

__global__ static void init(const int n, const float scale, float *const buf)
{
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  if (i < n) buf[i] = scale*float(i+1);
}

__global__ static void compare(const int n, const float *const buf, float *const diff)
{
  __shared__ float ourDiff;
  if (threadIdx.x == 0) ourDiff = 0;
  const int i = threadIdx.x+blockDim.x*blockIdx.x;
  __syncthreads();
  if (i < n) {
    const float x = float(i+1);
    float myDiff = (buf[i]-x)/x;
    myDiff = (myDiff < 0) ? -myDiff : myDiff;
    atomicMax(&ourDiff,myDiff);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(diff,ourDiff);
}

void run(const bool inPlace, const float *const sendBufD, float *const recvBufD, const int count, const int warmupIters, const int iters)
{
  static float *diffH = NULL;
  static float *diffD = NULL;
  if (diffH == NULL) CHECK(hipHostMalloc(&diffH,sizeof(float)));
  if (diffD == NULL) CHECK(hipMalloc(&diffD,sizeof(float)));

  const size_t bytes = count*sizeof(float);
  const void *const bufD = (inPlace) ? MPI_IN_PLACE : sendBufD;
  for (int i = 0; i < warmupIters; i++) {
    if (inPlace) CHECK(hipMemcpy(recvBufD,sendBufD,bytes,hipMemcpyDeviceToDevice));
    MPI_Allreduce(bufD,recvBufD,count,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  }

  constexpr int nt = 1024;
  const int nb = (count+nt-1)/nt;
  float sendDiff = 0;
  double sendTime = 0;
  for (int i = 0; i < iters; i++) {
    if (inPlace) CHECK(hipMemcpy(recvBufD,sendBufD,bytes,hipMemcpyDeviceToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    const double before = MPI_Wtime();
    MPI_Allreduce(bufD,recvBufD,count,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    const double after = MPI_Wtime();
    sendTime += after-before;
    CHECK(hipMemset(diffD,0,sizeof(float)));
    compare<<<nb,nt>>>(count,recvBufD,diffD);
    CHECK(hipMemcpy(diffH,diffD,sizeof(float),hipMemcpyDeviceToHost));
    sendDiff = (sendDiff < *diffH) ? *diffH : sendDiff;
  }
  sendTime *= 1.0e6/double(iters);
  float recvDiff = 0;
  MPI_Reduce(&sendDiff,&recvDiff,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
  double recvTime = 0;
  MPI_Reduce(&sendTime,&recvTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    const double algBW = double(count*sizeof(float))/(recvTime*1e3);
    const double busBW = algBW*double(2*(size-1))/double(size);
    printf(" %8.2f %7.2f %7.2f %6.0e",recvTime,algBW,busBW,recvDiff);
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc,&argv);
  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  enum { MAX_BYTES, MIN_BYTES, STEP_FACTOR, WARMUP_ITERS, ITERS, NUM_LARGS };
  long largs[NUM_LARGS];
  largs[MIN_BYTES] = 8;
  largs[MAX_BYTES] = sizeof(float)*16; //1024*1024;
  largs[STEP_FACTOR] = 2;
  largs[WARMUP_ITERS] = 5;
  largs[ITERS] = 20;

  if (rank == 0) {

    const struct option longopts[] = {
      {"maxbytes",required_argument,0,'e'},
      {"minbytes",required_argument,0,'b'},
      {"stepfactor",required_argument,0,'f'},
      {}};

    int longindex = 0;
    while(true) {
      const int i = getopt_long(argc,argv,"e:b:f",longopts,&longindex);
      if (i == -1) break;
      const char c = i;
      switch(c) {
        case 'b':
          largs[MIN_BYTES] = getBytes(optarg); break;
        case 'e':
          largs[MAX_BYTES] = getBytes(optarg); break;
        case 'f':
          largs[STEP_FACTOR] = strtol(optarg,NULL,0); break;
        default:
          if (c != 'h') printf("invalid option '%c'\n",c);
          printf("USAGE %s \n\t"
                 "[-b,--minbytes <min size in bytes>]\n\t"
                 "[-e,--maxbytes <max size in bytes>]\n\t"
                 "[-f,--stepfactor <increment factor>]\n\t"
                 ,basename(argv[0]));
          fflush(stdout);
          MPI_Abort(MPI_COMM_WORLD,0);
      }
    }

    printf("# nRanks: %d minBytes: %ld maxBytes: %ld step: %ld(factor) warmupIters: %ld iters: %ld\n",size,largs[MIN_BYTES],largs[MAX_BYTES],largs[STEP_FACTOR],largs[WARMUP_ITERS],largs[ITERS]);
    printf("#\n# Using devices\n");
    fflush(stdout);
  }
  MPI_Bcast(largs,NUM_LARGS,MPI_LONG,0,MPI_COMM_WORLD);

  long pid = getpid();
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(name,&len);
  int id = -1;
  CHECK(hipGetDevice(&id));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop,id));
  char pci[] = "0000:00:00.0";
  snprintf(pci,sizeof(pci),"%04x:%02x:%02x.0",prop.pciDomainID,prop.pciBusID,prop.pciDeviceID);
  char flow;

  if (rank == 0) {

    for (int i = 0; i < size; i++) {
      if (i > 0) {
        MPI_Send(&flow,1,MPI_CHAR,i,0,MPI_COMM_WORLD);
        MPI_Recv(&pid,1,MPI_LONG,i,i,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,i,i,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&id,1,MPI_INT,i,i,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(pci,strlen(pci),MPI_CHAR,i,i,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      printf("#   Rank %d Pid %ld on %s device %d [%s]\n",rank,pid,name,id,pci);
    }
    printf("#\n");
    printf("# %66s %27s\n","out-of-place","in-place");
    printf("# %10s %13s %9s %7s %8s %7s %7s %6s %8s %7s %7s %7s\n","size","count","type","redop","time","algbw","busbw","error","time","algbw","busbw","error");

  } else {

    MPI_Recv(&flow,1,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&pid,1,MPI_LONG,0,rank,MPI_COMM_WORLD);
    MPI_Send(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,rank,MPI_COMM_WORLD);
    MPI_Send(&id,1,MPI_INT,0,rank,MPI_COMM_WORLD);
    MPI_Send(pci,strlen(pci),MPI_CHAR,0,rank,MPI_COMM_WORLD);

  }

  assert(largs[MAX_BYTES] >= largs[MIN_BYTES]);
  assert(largs[MAX_BYTES] < long(INT_MAX)*sizeof(float));

  float *sendBufD = NULL;
  CHECK(hipMalloc(&sendBufD,largs[MAX_BYTES]));
  CHECK(hipMemset(sendBufD,0,largs[MAX_BYTES]));
  float *recvBufD = NULL;
  CHECK(hipMalloc(&recvBufD,largs[MAX_BYTES]));
  CHECK(hipMemset(recvBufD,0,largs[MAX_BYTES]));
 
  const int maxCount = largs[MAX_BYTES]/sizeof(float);
  constexpr int nt = 64;
  const int nb = (maxCount+nt-1)/nt;
  const float scale = float(rank+1)*float(2)/(float(size)*float(size+1));
  init<<<nb,nt>>>(maxCount,scale,sendBufD);
  CHECK(hipDeviceSynchronize());

  for (int count = largs[MIN_BYTES]/sizeof(float); count <= maxCount; count *= largs[STEP_FACTOR]) {
    if (rank == 0) printf("%12ld %13d %9s %7s",count*sizeof(float),count,"float","sum");
    run(false,sendBufD,recvBufD,count,largs[WARMUP_ITERS],largs[ITERS]);
    run(true,sendBufD,recvBufD,count,largs[WARMUP_ITERS],largs[ITERS]);
    if (rank == 0) {
      printf("\n");
      fflush(stdout);

    }
  }

  MPI_Finalize();
  return 0;
}

