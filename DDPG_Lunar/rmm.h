
#include "hls_stream.h"
#include "ap_fixed.h"
#include "ap_axi_sdata.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{
using namespace std;

const int prefix_s=1;

//#define K 128 //fanout 
//#define Lev1_Width 79
//#define Lev2_Width 10000 //78*128+16
#define K 64 //fanout 
// #define K 16
//#define K 16
#define D 3 //depth without root

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
//typedef struct {
//	int a[L4];
//} w3blockvec;
//typedef struct {
//	int out[BLOCK_SIZE][BLOCK_SIZE];
//} blockmat;

void pseudo_random(unsigned int seed, int load, rng_type ret[], int indrng);
int sampler(int prefix_s);
// void Sib_Iterator_l1(fixed_l1 TLev1[Lev1_Width],fixed_root x,hls::stream<sibit_io> &ind_newx, int parent[N_learner][D-1],int index_learners);
// void Sib_Iterator_l2(fixed_l2 TLev2[Lev2_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int parent[N_learner][D-1],int index_learners);
// void Sib_Iterator_l3(fixed_l3 TLev3[Lev3_Width],hls::stream<sibit_io> &ind_newxin, int ind_arr[N_learner],int index_learners);
void Sib_Iterator_l1(fixed_l1 TLev1[Lev1_Width],fixed_root x,hls::stream<sibit_io> &ind_newx, int parent[128][D-1],int index_learners);
void Sib_Iterator_l2(fixed_l2 TLev2[Lev2_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int parent[128][D-1],int index_learners);
void Sib_Iterator_l3(fixed_l3 TLev3[Lev3_Width],hls::stream<sibit_io> &ind_newxin, int ind_arr[128],int index_learners);
//void process_indices(hls::stream<sibit_io> &ind_newxin, int output_indices[],int index_learners);
//insert signal from cpu, update number from train module
// void Top_tree(int insert_signal,int *insert_ind,float *init_priority, int update_signal, hls::stream<ap_axiu<32,0,0,0>> &pn_in,int sample_signal,int load_seed,int *ind_o);
void Top_tree(int insert_signal,int *insert_ind,float *init_priority, int update_signal, hls::stream<ap_axiu<32,0,0,0>> &pn_in,int sample_signal,int load_seed,int *ind_o,
	int insert_batch, int N_learner);
}
