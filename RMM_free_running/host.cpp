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
#include "./block.h"


// #define N 4

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
// Main program
// ------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
    cl_int err;
    std::string binaryFile = (argc != 2) ? "top.xclbin" : argv[1];
    unsigned fileBufSize;
    std::vector<cl::Device> devices = get_xilinx_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    // cl::Kernel krnl_k2r(program, "top", &err); //comment out===========
    cl::Kernel top_tree(program, "Top_tree", &err);

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------

    // std::vector<int, aligned_allocator<int>> insert_ind_in;
    // insert_ind_in.resize(N);
    std::vector<float, aligned_allocator<float>> pn_in;
    pn_in.resize(N_actor);
    std::vector<int, aligned_allocator<int>> ind_o_out;
    ind_o_out.resize(N_learner);

    printf("here 1\n");
    
    
    int k;
    std::cout << "init..." << std::endl;

    for (k = 0; k < N_actor; k++) {
        pn_in[k] = 1;
    }
    for (k = 0; k < N_learner; k++) {
        ind_o_out[k]=0;
    }

    int insert_signal_in = 0;
    int insert_ind_in=8;
    int upd_in = 0;

  printf("inied\n");
    
    cl_mem_ext_ptr_t InrExt1;
    cl_mem_ext_ptr_t CrExt; //for output
    

    InrExt1.obj = pn_in.data();
    InrExt1.param = 0;
    InrExt1.flags = 0|XCL_MEM_TOPOLOGY;


    CrExt.obj = ind_o_out.data();
    CrExt.param = 0;
    CrExt.flags = 0|XCL_MEM_TOPOLOGY;

  printf("flags set\n");

    // Create the buffers and allocate memory

    cl::Buffer inpn_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * N_actor, &InrExt1, &err);
    printf("buf2\n");

    cl::Buffer out_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(int) * N_learner, &CrExt, &err);

    
    
      printf("setArg finished\n");


    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------
    top_tree.setArg(0, insert_signal_in);
    top_tree.setArg(1, insert_ind_in);
    top_tree.setArg(2, upd_in);
    top_tree.setArg(3, inpn_buf);
    top_tree.setArg(4, out_buf);

    // krnl_k2r[k].setArg(7, out_buf[k]);

    printf("setArg\n");
    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory

    // q.enqueueMigrateMemObjects(insert_signal_in, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({inpn_buf}, 0 /* 0 means from host*/);
    // q.enqueueMigrateMemObjects(upd_in, 0 /* 0 means from host*/);

    // q.finish();

    // // printf("sent data\n");
    // q.enqueueTask(krnl_k2r[k]);
    // q.finish();
    // // printf("executed kernel\n");
    
    // printf("data back\n");
    
    // Wait for all scheduled operations to finish

    q.enqueueTask(top_tree);
    q.finish();

    q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);

    // q.finish();
    // for (k = 0; k < 2; k++) {
    //     q.enqueueMigrateMemObjects({out_buf[k]}, CL_MIGRATE_MEM_OBJECT_HOST);
    // }
    q.finish();
    printf("q.finish\n");
    // ------------------------------------------------------------------------------------
    // Step 4: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    printf("ind_o_out content: ");
    for (k = 0; k < N_learner; k++) {
        printf("%d ",ind_o_out[k]);
    }
    bool match = true;
 
    delete[] fileBuf;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

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