#include "./rmm.h"
// #include "./learners_test.h"

extern "C"{

void pseudo_random(unsigned int seed, int load, rng_type ret[], int indrng) {
  static ap_uint<32> lfsr;
  // ap_uint<32> lfsr=lfsr_transfer;
  if (load ==1 )
    lfsr = seed;
  bool b_32 = lfsr.get_bit(32-32);
  bool b_22 = lfsr.get_bit(32-22);
  bool b_2 = lfsr.get_bit(32-2);
  bool b_1 = lfsr.get_bit(32-1);
  bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
  lfsr = lfsr >> 1;
  lfsr.set_bit(31, new_bit);

  // u u1;
  // u1.funit=lfsr;
  // ret[indrng]=lfsr;
  rng_type *p = reinterpret_cast<rng_type * >(&lfsr);
  ret[indrng] = *p;

  #ifndef __SYNTHESIS__
    printf("\nLSFR sequence: %s ",lfsr.to_string().c_str());  
    printf("\nlfsr to float: %.12f ",float(lfsr)); 
    printf("\nFixed pt number: %.12f ",float(ret[indrng]));  
  #endif
  // return lfsr.to_uint();
  // return ret;
}

// void Sib_Iterator_l1(fixed_l1 TLev1[Lev1_Width],fixed_root x,hls::stream<sibit_io> &ind_newx, int parent[N_learner][D-1],int index_learners){
void Sib_Iterator_l1(fixed_l1 TLev1[Lev1_Width],fixed_root x,hls::stream<sibit_io> &ind_newx, int parent[128][D-1],int index_learners){
	fixed_root local_pref=0;
	fixed_root prev_pref=0;
	sibit_io data_out;
	for (int i=0; i<Lev1_Width; i++){
		#pragma pipeline 
		// #ifndef __SYNTHESIS__
		// printf("sibiter l1: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
		// #endif
		prev_pref=local_pref;
		local_pref+=TLev1[i];
		if (local_pref >= x){
			//calculating the start addr of next layer sibling iteration
			data_out.start=(i)*K;
			//calculating the new x to be found in the next layer
			data_out.newx=x-prev_pref;
			ind_newx.write(data_out);	
			parent[index_learners][0]=i;
			// #ifndef __SYNTHESIS__
			// printf("Hit l1 par with(newx,parent) = (%f,%d)\n",x-prev_pref,i);
			// #endif 
			break;		
		}

	}
}

// void Sib_Iterator_l2(fixed_l2 TLev2[Lev2_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int parent[N_learner][D-1],int index_learners){
void Sib_Iterator_l2(fixed_l2 TLev2[Lev2_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int parent[128][D-1],int index_learners){
	fixed_l1 local_pref=0;
	fixed_l1 prev_pref=0;
	sibit_io data_in=ind_newxin.read();
	sibit_io data_out;
	int pipstart=data_in.start;
	int pipbound=data_in.start+K;
	fixed_root x=data_in.newx;
	// for (int i=pipstart; i<pipbound; i++){
	int i=pipstart;
	for (i=pipstart; i<pipbound; i++){
		#pragma pipeline
		fixed_l2 tmpload = TLev2[i];
		// #ifndef __SYNTHESIS__
		// printf("sibiter l2: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
		// #endif
		if (local_pref>=x){
			break;
		}
		prev_pref=local_pref;
		local_pref+=tmpload;
		

	}
	i = i -1;
	//calculating the start addr of next layer sibling iteration
	data_out.start=(i)*K;
	//calculating the new x to be found in the next layer
	data_out.newx=x-prev_pref;
	ind_newxout.write(data_out);
	parent[index_learners][1]=i;
	// #ifndef __SYNTHESIS__
	// printf("Hit l2 par with (newx,parent) = (%f,%d)\n",data_in.newx-prev_pref,i);
	// #endif
}

void Sib_Iterator_l3(fixed_l3 TLev3[Lev3_Width],hls::stream<sibit_io> &ind_newxin, int ind_arr[128],int index_learners){
	fixed_l2 local_pref=0;
	fixed_l2 prev_pref=0;
	sibit_io data_in=ind_newxin.read();
	sibit_io data_out;
	int pipstart=data_in.start;
	int pipbound=data_in.start+K;
	fixed_root x=data_in.newx;
	int i=pipstart;
	for (i=pipstart; i<pipbound; i++){
		#pragma pipeline
		// #ifndef __SYNTHESIS__
		// printf("sibiter l3: in for loop i=%d, local_pref=%f\n",i,float(local_pref));
		// #endif
		if (local_pref>=x){
			break;
		}
		prev_pref=local_pref;
		local_pref+=TLev3[i];
	}
	i=i-1;
	// //Last layer: output the current index (min index whose prefixsum>=x)
	// data_out.start=pipstart+i;
	// //No need to calculate the new x to be found in the next layer
	// data_out.newx=data_in.newx-prev_pref;
	// ind_newxout.write(data_out);
	ind_arr[index_learners]=i;
}



//insert ind,upd,pn: from top module logic/train (pn are the values to be substracted from all tree locations, size of N_learners)
//insert signal: from cpu- 1: initialize all priorities to be 0 (one-time-cost); 2: insert for the updated; 0: no insert
//sample signal: from cpu- : 1: sample; 0: no sample
//update signal: from fpga learners- : 1: update; 0: no update
// void Top_tree(int insert_signal,int *insert_ind,int *init_priority, int update_signal, float *pn,int sample_signal,int *ind_o){

void Top_tree(int insert_signal,int *insert_ind,float *init_priority, int update_signal, hls::stream<ap_axiu<32,0,0,0>> &pn_in,int sample_signal,int load_seed,int *ind_o,
	int insert_batch, int N_learner){
	#pragma HLS INTERFACE m_axi port=insert_ind bundle=gmem10 offset=slave
	#pragma HLS INTERFACE m_axi port=init_priority bundle=gmem10 offset=slave
	#pragma HLS INTERFACE m_axi port=ind_o bundle=gmem11 offset=slave

	
	#pragma HLS INTERFACE s_axilite port=insert_ind bundle=control
	#pragma HLS INTERFACE s_axilite port=init_priority bundle=control
	#pragma HLS INTERFACE s_axilite port=ind_o bundle=control

	#pragma HLS INTERFACE s_axilite port=insert_signal bundle=control
	#pragma HLS INTERFACE s_axilite port=update_signal bundle=control
	#pragma HLS INTERFACE s_axilite port=sample_signal bundle=control
	#pragma HLS INTERFACE s_axilite port=load_seed bundle=control
	#pragma HLS INTERFACE s_axilite port=insert_batch bundle=control
	#pragma HLS INTERFACE s_axilite port=N_learner bundle=control

	#pragma HLS INTERFACE axis port=pn_in 

	#pragma HLS INTERFACE s_axilite port=return bundle=control


	static Tree init_fan4;
	#pragma HLS bind_storage variable=init_fan4 type=RAM_2P impl=uram
	
	// printf("Initializing tree for testbench 1\n");
	
//=====================populate tree with 0: initialization=============
	if (insert_signal==1){
		#ifndef __SYNTHESIS__
		printf("\nTree init done.\n");
		#endif
		// for (int i=0;i<Lev2_Width;i++){ //16
		// 	init_fan4.TLev2[i]=0;
		// 	for (int j=0;j<K;j++){ //0*4 ~16*4-1 (0123, 0123....64 times)
		// 		init_fan4.TLev3[i*K+j]=0;
		// 	}
		// }
		// for (int i=0;i<K;i++){ //4
		// 	init_fan4.TLev1[i]=0; //24
		// }
		// init_fan4.TLev0=0; //root
		for (int j=0;j<Lev3_Width;j++){ //0*4 ~16*4-1 (0123, 0123....64 times)
			init_fan4.TLev3[j]=0;
		}
		for (int i=0;i<Lev2_Width;i++){ //16
			init_fan4.TLev2[i]=0;
		}
		for (int i=0;i<Lev1_Width;i++){ //4
			init_fan4.TLev1[i]=0; //24
		}
		init_fan4.TLev0=0; //root
	}


	//===========================Insert+Update for populating replay =======finish this later=========

	if (insert_signal==2){ //Uncomment afterwards as needed
		// fixed_insrt p_ini[insert_batch];
		// int insert_ind_local[insert_batch];
		fixed_insrt p_ini[128];
		int insert_ind_local[128]; //insert_batch cannot exceed 128
		// fixed_insrt p_ini
		for (int i=0;i<insert_batch;i++){
			fixed_insrt p1;
			p1=fixed_insrt(init_priority[i]);
			p_ini[i]=p1;
			insert_ind_local[i]=insert_ind[i]; 
		}

		#ifndef __SYNTHESIS__
		// printf("\n priorities to insert - port:\n");
		// for (int i=0;i<insert_batch;i++){
		// 	printf("%f ",init_priority[i]);
		// }
		printf("\n priorities to insert - fixed pt local buffer:\n");
		for (int i=0;i<insert_batch;i++){
			printf("%s ",p_ini[i].to_string().c_str());
		}
		printf("\n");
		printf("\n indices to insert:\n");
		for (int i=0;i<insert_batch;i++){
			printf("%d: ",insert_ind_local[i]);
		}
		printf("\n");


		printf("\nTree content before insertion\n");
		for (int i=0;i<0+insert_batch;i++){
			int ind_curr=insert_ind_local[i];
			printf("Actor at index %d: \n",i);
			printf("TLev3 index %d priority %f\n",ind_curr,float(init_fan4.TLev3[ind_curr]));
			printf("TLev2 index %d priority %f\n",ind_curr/K,float(init_fan4.TLev2[ind_curr/K])); //parent0
			printf("TLev1 index %d priority %f\n",(ind_curr/K)/K,float(init_fan4.TLev1[(ind_curr/K)/K])); //parent1
			printf("TLev0 priority %f\n\n",float(init_fan4.TLev0)); //parent2
		}
		#endif

		for (int i=0;i<0+insert_batch;i++){
			int ind_curr=insert_ind_local[i];
			int diff=p_ini[i]-init_fan4.TLev3[ind_curr];
			init_fan4.TLev3[ind_curr] = p_ini[i];
			init_fan4.TLev2[ind_curr/K] += diff; //parent0
			init_fan4.TLev1[(ind_curr/K)/K] += diff; //parent1
			init_fan4.TLev0 += diff; //parent2
		#ifndef __SYNTHESIS__
		printf("insert %d",ind_curr);
		#endif
		}

		#ifndef __SYNTHESIS__
		printf("\nTree content after insertion\n");
		for (int i=0;i<0+insert_batch;i++){
			int ind_curr=insert_ind_local[i];
			printf("Actor at index %d: \n",i);
			printf("TLev3 index %d priority %f\n",ind_curr,float(init_fan4.TLev3[ind_curr]));
			printf("TLev2 index %d priority %f\n",ind_curr/K,float(init_fan4.TLev2[ind_curr/K])); //parent0
			printf("TLev1 index %d priority %f\n",(ind_curr/K)/K,float(init_fan4.TLev1[(ind_curr/K)/K])); //parent1
			printf("TLev0 priority %f\n\n",float(init_fan4.TLev0)); //parent2
		}
		#endif
	}



	// //=====================populate tree for testbench purposes=============
	// #ifndef __SYNTHESIS__
	// printf("Initializing tree for testbench\n");
	// for (int i=0;i<Lev2_Width;i++){ //16
	// 	init_fan4.TLev2[i]=0+1+2+3;
	// 	for (int j=0;j<K;j++){ //0*4 ~16*4-1 (0123, 0123....64 times)
	// 		init_fan4.TLev3[i*K+j]=j;
	// 	}
	// }
	// for (int i=0;i<K;i++){ //4
	// 	init_fan4.TLev1[i]=4*(0+1+2+3); //24
	// }
	// init_fan4.TLev0=(0+1+2+3)*16; //root
	// #endif 

	// int ind_o_local[N_learner];
	// //===========================RNG and sampling=============================
	// static int par_m[N_learner][D-1]; //memoinization array for parent nodes, useful in later update process
	int ind_o_local[128];
	//===========================RNG and sampling=============================
	static int par_m[128][D-1]; //memoinization array for parent nodes, useful in later update process
	if (sample_signal==1){
		hls::stream<sibit_io> l1_out_pipe;	
		hls::stream<sibit_io> l2_out_pipe;
		// hls::stream<sibit_io> l3_out_pipe;	
		#pragma HLS STREAM variable=l1_out_pipe depth=8 
		#pragma HLS STREAM variable=l2_out_pipe depth=8 
		// #pragma HLS STREAM variable=l3_out_pipe depth=8 
		#pragma HLS bind_storage variable=l1_out_pipe type=FIFO impl=bram
		#pragma HLS bind_storage variable=l2_out_pipe type=FIFO impl=bram
		// #pragma HLS bind_storage variable=l3_out_pipe type=FIFO impl=bram
		// fixed_root rngs [N_learner]={1.0, 4.9, 6.0, 5.0, 2.0, 17.0, 2.0, 6.1, 1.0, 4.9, 6.0, 5.0, 2.0, 17.0, 2.0, 6.1}; //should return 1,3,51,3,17,25,2,5
		// fixed_root rngs [N_learner];
		// rng_type retz[N_learner]; //betwen 0 and 1
		fixed_root rngs [128];
		rng_type retz[128]; //betwen 0 and 1
		if (load_seed==1) pseudo_random(67,1,retz,0); //seed is 67
		  
		for (int i=0; i<N_learner; i++){
		    // pseudo_random(seed, load, ret[i], lfsr);
		    pseudo_random(67, 0, retz,i);
		}

		for (int i=0; i<N_learner; i++){
		    rngs[i]=retz[i]*init_fan4.TLev0;
		}

		#ifndef __SYNTHESIS__
		printf ("\nAt sampling, root value: %f\n", float(init_fan4.TLev0));
		printf ("\nrandom retz between 0 and 1:\n");
		for (int i=0; i<N_learner; i++){
		    printf ("%f, ", float(retz[i]));
		}
		printf ("\nrngs between 0 and root:\n");
		for (int i=0; i<N_learner; i++){
		    printf ("%f, ", float(rngs[i]));
		}		    
		#endif
		
		samp_main_loop:for (int i=0;i<N_learner;i++){
			#pragma DATAFLOW
	
			Sib_Iterator_l1(init_fan4.TLev1,rngs[i],l1_out_pipe,par_m,i);

			Sib_Iterator_l2(init_fan4.TLev2,l1_out_pipe,l2_out_pipe,par_m,i);

			Sib_Iterator_l3(init_fan4.TLev3,l2_out_pipe,ind_o_local,i);

		}
		for (int i=0;i<N_learner;i++){
			ind_o[i]=ind_o_local[i];
		}
		#ifndef __SYNTHESIS__
		printf("Sampling: parent memoinization array:\n\n");
		for (int i=0;i<N_learner;i++){
			printf("%d %d\n",par_m[i][0],par_m[i][1]); //TLev1,TLev2
		}
		printf("Sampling: output indices array:\n\n");
		for (int i=0;i<N_learner;i++){
			printf("%d ",ind_o[i]); //TLev3
		}
		#endif
	}
	//===========================Update=============================


	if (update_signal==1){ //Uncomment later as needed
	#ifndef __SYNTHESIS__

		printf("Entered update: parent memoinization array:\n\n");
		for (int i=0;i<N_learner;i++){
			printf("%d %d\n",par_m[i][0],par_m[i][1]); //TLev1,TLev2
		}
	printf("Tree content before update\n");
	for (int i=0;i<N_learner;i++){
		printf("Learner at index %d: \n",i);
		printf("TLev0 with priority %f\n",float(init_fan4.TLev0));
		printf("TLev1 index %d with priority %f\n",par_m[i][0],float(init_fan4.TLev1[par_m[i][0]]));
		printf("TLev2 index %d with priority %f\n",par_m[i][1],float(init_fan4.TLev2[par_m[i][1]]));
		printf("TLev3 index %d with priority %f\n\n",ind_o[i],float(init_fan4.TLev3[ind_o[i]]));
	}
	#endif

		// fixed_upd pnint [N_learner];
		fixed_upd pnint [128];
		for (int i=0;i<N_learner;i++){
			ap_axiu<32,0,0,0> pntmp=pn_in.read();
			#ifndef __SYNTHESIS__
			printf("\n===================================pn_in receive from Tree=======================================:%f",float(pntmp.data));
			#endif
			pnint[i]=pntmp.data;
		}
		upd_main_loop:for (int i=0;i<N_learner;i++){
			#pragma pipeline
			fixed_l3 diff=init_fan4.TLev3[ind_o[i]]-pnint[i];

			init_fan4.TLev0-=diff;
			init_fan4.TLev1[par_m[i][0]]-=diff;
			init_fan4.TLev2[par_m[i][1]]-=diff;
			init_fan4.TLev3[ind_o[i]]-=diff;
		}

	#ifndef __SYNTHESIS__
	printf("Tree content after update\n");
	for (int i=0;i<N_learner;i++){
		printf("Learner at index %d: \n",i);
		printf("TLev0 with priority %f\n",float(init_fan4.TLev0));
		printf("TLev1 index %d with priority %f\n",par_m[i][0],float(init_fan4.TLev1[par_m[i][0]]));
		printf("TLev2 index %d with priority %f\n",par_m[i][1],float(init_fan4.TLev2[par_m[i][1]]));
		printf("TLev3 index %d with priority %f\n\n",ind_o[i],float(init_fan4.TLev3[ind_o[i]]));
	}
	#endif 

	}

}

}



