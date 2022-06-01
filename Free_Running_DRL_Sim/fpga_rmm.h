#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include <cstdio>
#include <cstring>
// #include <xcl2.hpp>

// =================rmm.h for kernel========================
#include "hls_stream.h"
#include "ap_fixed.h"
// #include "hls_math.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <iomanip>


extern "C"{
using namespace std;

const int prefix_s=1;

//#define K 128 //fanout 
//#define Lev1_Width 79
//#define Lev2_Width 10000 //78*128+16
#define K 4 //fanout 
#define D 3 //depth without root
// #define N_learner 8 //learning batch size
#define BS 256 //largest supported learning batch size,
// #define N_actor 32 //inf insert batch size

// #define N_learner 128 //learning batch size
// #define insert_batch 128 //inf batch size
#define Lev1_Width 3
#define Lev2_Width 192
#define Lev3_Width 12288

//K=4=2^2, so each level integer precision = its parent level-2
//K=8=2^3, so each level integer precision = its parent level-3
//fixed point: <total bits, integer bits>
typedef ap_fixed<32,26> fixed_root;
typedef ap_fixed<32,24> fixed_l1;
typedef ap_fixed<32,22> fixed_l2;
typedef ap_fixed<32,20> fixed_l3;

typedef ap_fixed<22,16> fixed_upd;
typedef ap_fixed<12,6> fixed_insrt;
//typedef struct {
//	int TLev0;
//	int TLev1[Lev1_Width];
//	int TLev2[Lev2_Width];
//} Tree;
typedef ap_ufixed<32,0> rng_type;

typedef struct {
	fixed_root TLev0;
	fixed_l1 TLev1[Lev1_Width];
	fixed_l2 TLev2[Lev2_Width];
	fixed_l3 TLev3[Lev3_Width];
} Tree;

typedef struct {
	int start;
	fixed_root newx;
} sibit_io;
// =================End rmm.h for kernel========================


// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 16
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15)};

template <typename T1>
struct aligned_allocator
{
  using value_type = T1;
  T1* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T1)))
      throw std::bad_alloc();
    return reinterpret_cast<T1*>(ptr);
  }
  void deallocate(T1* p, std::size_t num)
  {
    free(p);
  }
};


// ------------------------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------------------------
std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++)
    {
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx")
        {
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size())
    {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if (access(xclbin_file_name.c_str(), R_OK) != 0)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);
    return buf;
}