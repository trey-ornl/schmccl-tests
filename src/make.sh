module load rocm
export CXX=hipcc
export CXXFLAGS="-g -O3 --offload-arch=gfx90a -I${MPICH_DIR}/include"
export LDFLAGS="-L${MPICH_DIR}/lib ${PE_MPICH_GTL_DIR_amd_gfx90a}"
export LIBS="-lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
make
