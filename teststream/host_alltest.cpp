#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define In_dim 8 //L1
// #define L1 8 //L1
// #define L2 64 //L2
// #define L3 4 //L3

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include "./learners_test.h"
#include "./rmm.h"

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
    // cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE , &err);

    // string testinput;

    // cin >> testinput;
    

    // ------------------------------------------------------------------------------------
    // Create buffers and initialize test values: Learners
    // ------------------------------------------------------------------------------------
    //In_dim=BATCH SIZE
    // learners_top(blockvec *S, blockvec *Snt, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],int wsync)
    std::vector<blockvec, aligned_allocator<blockvec>> In_rows;
    In_rows.resize(In_dim*BATCHS);
    // std::vector<blockvec> In_rows_snt(In_dim);
    std::vector<blockvec, aligned_allocator<blockvec>> In_rows_snt;
    In_rows_snt.resize(In_dim*BATCHS);
    // std::vector<w1blockvec> Out_w1bram(L1);
    std::vector<w1blockvec, aligned_allocator<w1blockvec>> Out_w1bram;
    Out_w1bram.resize(L1);
    // std::vector<w3blockvec> Out_w2bram(L2);
    std::vector<w3blockvec, aligned_allocator<w3blockvec>> Out_w2bram;
    Out_w2bram.resize(L2);
    // std::vector<blockvec> C_rows(In_dim);

    std::vector<actvec, aligned_allocator<actvec>> In_actions;
    In_actions.resize(BATCHS);

    std::vector<blockvec, aligned_allocator<blockvec>> In_rewards;
    In_rewards.resize(BATCHS);

    std::vector<bsbit, aligned_allocator<bsbit>> In_dones;
    In_dones.resize(BATCHS);

    std::vector<float, aligned_allocator<float>> Out_bias1;
    Out_bias1.resize(L2);
    // std::vector<w3blockvec> Out_w2bram(L2);
    std::vector<float, aligned_allocator<float>> Out_bias2;
    Out_bias2.resize(L3);

    std::vector<float, aligned_allocator<float>> Out_Q;
    Out_Q.resize(BATCHS*BSIZE);

    std::vector<float, aligned_allocator<float>> Out_Loss;
    Out_Loss.resize(BATCHS*BSIZE);

    
    int i, j, jj;
    std::cout << "init input states..." << std::endl;
    printf("\ninput s content:\n");
    
    for (jj = 0; jj < BATCHS; jj++) {   
        for (j = 0; j < L1; j++) {
            for (i = 0; i < BSIZE; i++) {
            In_rows[L1*jj+j].a[i] = float(-j)/float(4.0);
            In_rows_snt[L1*jj+j].a[i] = float(j)/float(4.0);
            // printf("%f ",In_rows[j].a[i]);
            // printf("%f ",In_rows_snt[j].a[i]);
            }
        }
    }

    printf("\nInit input reward/action/done content...\n");

    for (jj = 0; jj < BATCHS; jj++) {   
        for (i = 0; i < BSIZE; i++) {
        // printf("\njj,i:%d,%d\n",jj,i);
        In_actions[jj].a[i] = int(2);
        In_rewards[jj].a[i] = float(1);
        In_dones[jj].a[i] = int(0);
        // printf("%f ",In_actions[jj].a[i]);
        }
    }
    // printf("inied learners\n");


    // ------------------------------------------------------------------------------------
    // Create buffers and initialize test values: Replay
    // ------------------------------------------------------------------------------------
    // Top_tree(int insert_signal,int *insert_ind,float *init_priority, int update_signal, hls::stream<ap_axiu<32,0,0,0>> &pn_in,int sample_signal,int *ind_o)
    // std::vector<float, aligned_allocator<float>> pn_in;
    // pn_in.resize(insert_batch);
    std::vector<int, aligned_allocator<int>> insert_ind;
    insert_ind.resize(insert_batch);
    std::vector<float, aligned_allocator<float>> init_priority;
    init_priority.resize(insert_batch);
    std::vector<int, aligned_allocator<int>> ind_o_out;
    ind_o_out.resize(N_learner);

    printf("\nInit replay insert inputs...\n");
    for (i = 0; i < insert_batch; i++) {
        // printf("\njj,i:%d,%d\n",jj,i);
        insert_ind[i] = i;
        init_priority[i] = i+1;
        // printf("%f ",In_actions[jj].a[i]);
    }
    
    
    // =================learners=================
    cl_mem_ext_ptr_t InrExt;
    cl_mem_ext_ptr_t InrExt2;
    cl_mem_ext_ptr_t InrExt3;
    cl_mem_ext_ptr_t InrExt4;
    cl_mem_ext_ptr_t InrExt5;
    cl_mem_ext_ptr_t OutExt;
    cl_mem_ext_ptr_t OutExt2;
    cl_mem_ext_ptr_t OutExt3;
    cl_mem_ext_ptr_t OutExt4;
    cl_mem_ext_ptr_t OutExt5;
    cl_mem_ext_ptr_t OutExt6;
    

    InrExt.obj = In_rows.data();
    InrExt.param = 0;
    InrExt.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    InrExt2.obj = In_rows_snt.data();
    InrExt2.param = 0;
    InrExt2.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    InrExt3.obj = In_actions.data();
    InrExt3.param = 0;
    InrExt3.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    InrExt4.obj = In_rewards.data();
    InrExt4.param = 0;
    InrExt4.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    InrExt5.obj = In_dones.data();
    InrExt5.param = 0;
    InrExt5.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    OutExt.obj = Out_w1bram.data();
    OutExt.param = 0;
    OutExt.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    OutExt2.obj = Out_w2bram.data();
    OutExt2.param = 0;
    OutExt2.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;


    OutExt3.obj = Out_bias1.data();
    OutExt3.param = 0;
    OutExt3.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    OutExt4.obj = Out_bias2.data();
    OutExt4.param = 0;
    OutExt4.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;


    OutExt5.obj = Out_Q.data();
    OutExt5.param = 0;
    OutExt5.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;

    OutExt6.obj = Out_Loss.data();
    OutExt6.param = 0;
    OutExt6.flags = XCL_MEM_DDR_BANK0|XCL_MEM_TOPOLOGY;
    
    // ========================replay=====================
    
    cl_mem_ext_ptr_t RepInExt1;
    cl_mem_ext_ptr_t RepInExt2;
    cl_mem_ext_ptr_t RepoutExt; //for output
    

    RepInExt1.obj = insert_ind.data();
    RepInExt1.param = 0;
    RepInExt1.flags = 1|XCL_MEM_TOPOLOGY;

    RepInExt2.obj = init_priority.data();
    RepInExt2.param = 0;
    RepInExt2.flags = 1|XCL_MEM_TOPOLOGY;

    RepoutExt.obj = ind_o_out.data();
    RepoutExt.param = 0;
    RepoutExt.flags = 1|XCL_MEM_TOPOLOGY;

    printf("flags set\n");

    // Create the buffers and allocate memory


    // actvec acts={2}; //consistent with tb. Moved to Aligned allocator
    // blockvec r={1}; //consistent with tb. Moved to Aligned allocator
    float gamma=0.5;
    float alpha=0.1;
    int wsync = 0;

  
    // Create the buffers and allocate memory
    cl::Buffer in1_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * In_dim * BATCHS, &InrExt, &err);
    cl::Buffer in2_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * In_dim * BATCHS, &InrExt2, &err);
    cl::Buffer in3_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(actvec) * BATCHS, &InrExt3, &err);
    // std::cout << sizeof(actvec) * BATCHS << std::endl;
    cl::Buffer in4_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * BATCHS, &InrExt4, &err);
    cl::Buffer in5_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(bsbit) * BATCHS, &InrExt5, &err);
    cl::Buffer out1_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(w1blockvec) * L1, &OutExt, &err);
    cl::Buffer out2_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(w3blockvec) * L2, &OutExt2, &err);
    cl::Buffer out3_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * L2, &OutExt3, &err);
    cl::Buffer out4_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * L3, &OutExt4, &err);
    cl::Buffer out5_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * BATCHS*BSIZE, &OutExt5, &err);
    cl::Buffer out6_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * BATCHS*BSIZE, &OutExt6, &err);
    printf("Learners buffers allocated\n");

    // Top_tree(int insert_signal,int *insert_ind,float *init_priority, int update_signal, hls::stream<ap_axiu<32,0,0,0>> &pn_in,int sample_signal,int *ind_o)
    int insert_signal_in = 0;
    int update_signal=1;
    int sample_signal = 1;
    cl::Buffer insind_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(int) * insert_batch, &RepInExt1, &err);
    cl::Buffer inpn_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * insert_batch, &RepInExt2, &err);
    cl::Buffer out_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(int) * N_learner, &RepoutExt, &err);
    printf("Replay buffers allocated\n");


    // Set kernel arguments

    // learners_top(blockvec *S, blockvec *Snt, actvec *acts,blockvec *r,float gamma, float alpha, bsbit *done, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],float *bias1_out,float *bias2_out,int wsync, /*Learners args*/
    // krnl_top.setArg(0, in1_buf);
    // krnl_top.setArg(1, in2_buf);
    // krnl_top.setArg(2, in3_buf);
    // krnl_top.setArg(3, in4_buf);
    // krnl_top.setArg(4, gamma);
    // krnl_top.setArg(5, alpha);
    // krnl_top.setArg(6, in5_buf);
    // krnl_top.setArg(7, out1_buf);
    // krnl_top.setArg(8, out2_buf);
    // krnl_top.setArg(9, out3_buf);
    // krnl_top.setArg(10, out4_buf); //bias2, float*L3
    // krnl_top.setArg(11, wsync);
    // krnl_top.setArg(12, out5_buf); //Logging Qs  float*BATCHS*BSIZE
    // krnl_top.setArg(13, out6_buf); //Logging Loss  float*BATCHS*BSIZE

    
    // Top_tree(int insert_signal,int *insert_ind,float *init_priority, int update_signal, hls::stream<ap_axiu<32,0,0,0>> &pn_in,int sample_signal,int *ind_o)
    // krnl_tree.setArg(0, insert_signal_in);
    // krnl_tree.setArg(1, insind_buf);
    // krnl_tree.setArg(2, inpn_buf);
    // krnl_tree.setArg(3, update_signal);
    // krnl_tree.setArg(5, sample_signal);
    // krnl_tree.setArg(6, out_buf);

    // printf("setArg finished\n");
    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------

    // ===================Replay Insert Initialze:
    // ===================and Replay sampling: (insert only needed once at first. In all later insertion, use insert_signal_in = 2;)
    cl::Kernel krnl_tree(program, "Top_tree", &err);
    insert_signal_in = 1;
    update_signal=0;
    sample_signal = 0;
    krnl_tree.setArg(0, insert_signal_in);
    krnl_tree.setArg(1, insind_buf);
    krnl_tree.setArg(2, inpn_buf);
    krnl_tree.setArg(3, update_signal);
    krnl_tree.setArg(5, sample_signal);
    krnl_tree.setArg(6, out_buf);
    // q.enqueueMigrateMemObjects({insind_buf}, 0 /* 0 means from host*/);
    // q.enqueueMigrateMemObjects({inpn_buf}, 0 /* 0 means from host*/);
    q.enqueueTask(krnl_tree);
    // q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();
    printf("\nFrom Host: Tree init done.\n");
    

    // cl::Kernel krnl_top(program, "learners_top", &err);
    // for (i=0;i<3;i++){
    //     cl::Kernel krnl_tree1(program, "Top_tree", &err);
    //     insert_signal_in = 2;
    //     update_signal=0;
    //     sample_signal = 1;
    //     krnl_tree1.setArg(0, insert_signal_in);
    //     krnl_tree1.setArg(1, insind_buf);
    //     krnl_tree1.setArg(2, inpn_buf);
    //     krnl_tree1.setArg(3, update_signal);
    //     krnl_tree1.setArg(5, sample_signal);
    //     krnl_tree1.setArg(6, out_buf);
    //     q.enqueueMigrateMemObjects({insind_buf,inpn_buf}, 0  0 means from host);
    //     q.enqueueTask(krnl_tree1);
    //     q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    //     q.finish();
    //     // printf("===============================================\n");
    //     // printf("========= Sampling: DATA BACK TO HOST =========\n");
    //     // printf("===============================================\n");
    //     printf("\nSample indices content:\n");
    //     for(int i = 0; i < N_learner; i++) {
    //         printf("%d ",ind_o_out[i]);
    //     }
    // }


    cl::Kernel krnl_tree2(program, "Top_tree", &err);
    insert_signal_in = 2;
    update_signal=0;
    sample_signal = 1;

        printf("\n Host: priorities to insert:\n");
        for (int i=0;i<insert_batch;i++){
            printf("%f ",init_priority[i]);
        }
        printf("\n");
        printf("\n Host: indices to insert:\n");
        for (int i=0;i<insert_batch;i++){
            printf("%d: ",insert_ind[i]);
        }
        printf("\n");

    krnl_tree2.setArg(0, insert_signal_in);
    krnl_tree2.setArg(1, insind_buf);
    krnl_tree2.setArg(2, inpn_buf);
    krnl_tree2.setArg(3, update_signal);
    krnl_tree2.setArg(5, sample_signal);
    krnl_tree2.setArg(6, out_buf);
    q.enqueueMigrateMemObjects({insind_buf,inpn_buf}, 0 /* 0 means from host*/);
    q.enqueueTask(krnl_tree2);
    q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();
    // printf("===============================================\n");
    // printf("========= Sampling: DATA BACK TO HOST =========\n");
    // printf("===============================================\n");
    printf("\nSample indices content:\n");
    for(int i = 0; i < N_learner; i++) {
        printf("%d ",ind_o_out[i]);
    }
    printf("\n");
    q.finish(); //(??? Sequential after Insert or need barrier ???):
    printf("q.finish\n");
    //

    // ===================Learner train with weights init:
    gamma=0.5;
    alpha=0.1;
    wsync = 0;

    cl::Kernel krnl_top(program, "learners_top", &err);
    krnl_top.setArg(0, in1_buf);
    krnl_top.setArg(1, in2_buf);
    krnl_top.setArg(2, in3_buf);
    krnl_top.setArg(3, in4_buf);
    krnl_top.setArg(4, gamma);
    krnl_top.setArg(5, alpha);
    krnl_top.setArg(6, in5_buf);
    krnl_top.setArg(7, out1_buf);
    krnl_top.setArg(8, out2_buf);
    krnl_top.setArg(9, out3_buf);
    krnl_top.setArg(10, out4_buf); //bias2, float*L3
    krnl_top.setArg(11, wsync);
    krnl_top.setArg(12, out5_buf); //Logging Qs  float*BATCHS*BSIZE
    krnl_top.setArg(13, out6_buf); //Logging Loss  float*BATCHS*BSIZE

    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory
    q.enqueueMigrateMemObjects({in1_buf,in2_buf,in3_buf,in4_buf,in5_buf}, 0 /* 0 means from host*/);
    // q.finish();
    printf("sent data\n");
    q.enqueueTask(krnl_top);
    // q.finish();
    printf("enqueue\n");
    q.enqueueMigrateMemObjects({out1_buf,out2_buf,out3_buf,out4_buf,out5_buf,out6_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    printf("executed learner kernel with weight init\n");
    // printf("data back\n");

    // No need to Wait for all scheduled operations to finish, update and learner happens concurrently
    // q.finish();
    
    // ===================Replay Update (??? Parallel with train ???):
    cl::Kernel krnl_tree3(program, "Top_tree", &err);
    insert_signal_in = 0;
    update_signal=1;
    sample_signal = 0;
    krnl_tree3.setArg(0, insert_signal_in);
    krnl_tree3.setArg(1, insind_buf);
    krnl_tree3.setArg(2, inpn_buf);
    krnl_tree3.setArg(3, update_signal);
    krnl_tree3.setArg(5, sample_signal);
    krnl_tree3.setArg(6, out_buf);
    q.enqueueMigrateMemObjects({insind_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({inpn_buf}, 0 /* 0 means from host*/);
    q.enqueueTask(krnl_tree3);
    q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);

    q.finish(); //(??? Sequential after Insert or need barrier ???):
    printf("q.finish\n");
    printf("============================================================================\n");
    printf("================= DATA BACK TO HOST: Learners Train round 1 ================\n");
    printf("============================================================================\n");
    // Print weights, bias

    printf("\nW1 content:\n");
    for(int i = 0; i < L1; i++) {
        for(int j = 0; j < L2; j++) {
                printf("%.8f ",Out_w1bram[j].a[i]);  //L1 rows, L2 cols                
        }
        printf("\n");        
    }
    printf("\nBias1 content:\n");
    for(int i = 0; i < L2; i++) {
        printf("%.8f ",Out_bias1[i]);                 
    }
    printf("\n"); 

    printf("\nW2 content:\n");
    for(int i = 0; i < L2; i++) {
        for(int j = 0; j < L3; j++) {
                printf("%.8f ",Out_w2bram[j].a[i]);  //L2 rows, L3 cols
        }
        printf("\n");
    }
    printf("\nBias2 content:\n");
    for(int i = 0; i < L3; i++) {
        printf("%.8f ",Out_bias2[i]);           
    }
    printf("\n");  

    //Print Qs, Loss

    printf("\nQs content:\n");
    for(int i = 0; i < BATCHS*BSIZE; i++) {
        printf("%.8f ",Out_Q[i]);
    }
    printf("\n"); 
    printf("\nLoss content:\n");
    for(int i = 0; i < BATCHS*BSIZE; i++) {
        printf("%.8f ",Out_Loss[i]);
    }
    printf("\n"); 
    // ===================Learner train with static weight:


    cl::Kernel krnl_top1(program, "learners_top", &err);
    gamma=0.5;
    alpha=0.1;
    wsync = 1;
    krnl_top1.setArg(0, in1_buf);
    krnl_top1.setArg(1, in2_buf);
    krnl_top1.setArg(2, in3_buf);
    krnl_top1.setArg(3, in4_buf);
    krnl_top1.setArg(4, gamma);
    krnl_top1.setArg(5, alpha);
    krnl_top1.setArg(6, in5_buf);
    krnl_top1.setArg(7, out1_buf);
    krnl_top1.setArg(8, out2_buf);
    krnl_top1.setArg(9, out3_buf);
    krnl_top1.setArg(10, out4_buf); //bias2, float*L3
    krnl_top1.setArg(11, wsync);
    krnl_top1.setArg(12, out5_buf); //Logging Qs  float*BATCHS*BSIZE
    krnl_top1.setArg(13, out6_buf); //Logging Loss  float*BATCHS*BSIZE
    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory
    q.enqueueMigrateMemObjects({in1_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in2_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in3_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in4_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in5_buf}, 0 /* 0 means from host*/);
    // q.finish();
    printf("sent data\n");
    q.enqueueTask(krnl_top1);
    // q.finish();
    printf("enqueue\n");
    q.enqueueMigrateMemObjects({out1_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({out2_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({out3_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({out4_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({out5_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({out6_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    printf("executed learner kernel with weight init\n");
    // ===================Replay Update (??? Parallel with train ???):
    insert_signal_in = 0;
    update_signal=1;
    sample_signal = 0;
    q.enqueueMigrateMemObjects({insind_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({inpn_buf}, 0 /* 0 means from host*/);
    q.enqueueTask(krnl_tree);
    q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);

    cl::Kernel krnl_tree4(program, "Top_tree", &err);
    insert_signal_in = 0;
    update_signal=1;
    sample_signal = 0;
    krnl_tree4.setArg(0, insert_signal_in);
    krnl_tree4.setArg(1, insind_buf);
    krnl_tree4.setArg(2, inpn_buf);
    krnl_tree4.setArg(3, update_signal);
    krnl_tree4.setArg(5, sample_signal);
    krnl_tree4.setArg(6, out_buf);
    q.enqueueMigrateMemObjects({insind_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({inpn_buf}, 0 /* 0 means from host*/);
    q.enqueueTask(krnl_tree4);
    q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);

    q.finish(); //(??? Sequential after Insert or need barrier ???):
    printf("q.finish\n");
// q.finish();
    printf("======================================================================================================\n");
    printf("================================== DATA BACK TO HOST: Train round 2 ==================================\n");
    printf("======================================================================================================\n");
    // Print weights, bias

    printf("\nW1 content:\n");
    for(int i = 0; i < L1; i++) {
        for(int j = 0; j < L2; j++) {
                printf("%.8f ",Out_w1bram[j].a[i]);  //L1 rows, L2 cols                
        }
        printf("\n");        
    }
    printf("\nBias1 content:\n");
    for(int i = 0; i < L2; i++) {
        printf("%.8f ",Out_bias1[i]);                 
    }
    printf("\n"); 

    printf("\nW2 content:\n");
    for(int i = 0; i < L2; i++) {
        for(int j = 0; j < L3; j++) {
                printf("%.8f ",Out_w2bram[j].a[i]);  //L2 rows, L3 cols
        }
        printf("\n");
    }
    printf("\nBias2 content:\n");
    for(int i = 0; i < L3; i++) {
        printf("%.8f ",Out_bias2[i]);           
    }
    printf("\n");  

    //Print Qs, Loss

    printf("\nQs content:\n");
    for(int i = 0; i < BATCHS*BSIZE; i++) {
        printf("%.8f ",Out_Q[i]);
    }
    printf("\n"); 
    printf("\nLoss content:\n");
    for(int i = 0; i < BATCHS*BSIZE; i++) {
        printf("%.8f ",Out_Loss[i]);
    }
    printf("\n"); 


    delete[] fileBuf;

    // ------------------------------------------------------------------------------------
    // Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    // bool match = true;
    std::cout << "TEST PASSED" << std::endl;
    return EXIT_SUCCESS;
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