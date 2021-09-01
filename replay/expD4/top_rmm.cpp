#include "./block.h"

extern "C"{


void Sib_Iterator_l1(int TLev1[Lev1_Width],int x,hls::stream<sibit_io> &ind_newx, int parent[N_learner][D-1],int index_learners){
	int local_pref=0;
	int prev_pref=0;
	sibit_io data_out;
	for (int i=0; i<Lev1_Width; i++){
		#pragma pipeline 
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

void Sib_Iterator_l2(int TLev2[Lev2_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int parent[N_learner][D-1],int index_learners){
	int local_pref=0;
	int prev_pref=0;
	sibit_io data_in=ind_newxin.read();
	sibit_io data_out;
	int pipstart=data_in.start;
	int pipbound=data_in.start+K;
	int x=data_in.newx;
	// for (int i=pipstart; i<pipbound; i++){
	int i=pipstart;
	for (i=pipstart; i<pipbound; i++){
		#pragma pipeline
		int tmpload = TLev2[i];
		
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

void Sib_Iterator_l3(int TLev3[Lev3_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int parent[N_learner][D-1],int index_learners){
	int local_pref=0;
	int prev_pref=0;
	sibit_io data_in=ind_newxin.read();
	sibit_io data_out;
	int pipstart=data_in.start;
	int pipbound=data_in.start+K;
	int x=data_in.newx;
	// for (int i=pipstart; i<pipbound; i++){
	int i=pipstart;
	for (i=pipstart; i<pipbound; i++){
		#pragma pipeline
		int tmpload = TLev3[i];
		
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

// void Sib_Iterator_l3(int TLev3[Lev3_Width],hls::stream<sibit_io> &ind_newxin,hls::stream<sibit_io> &ind_newxout, int ind_arr[N_learner],int index_learners){
void Sib_Iterator_l4(int TLev4[Lev4_Width],hls::stream<sibit_io> &ind_newxin, int ind_arr[N_learner],int index_learners){
	int local_pref=0;
	int prev_pref=0;
	sibit_io data_in=ind_newxin.read();
	sibit_io data_out;
	int pipstart=data_in.start;
	int pipbound=data_in.start+K;
	int x=data_in.newx;
	int i=pipstart;
	for (i=pipstart; i<pipbound; i++){
		#pragma pipeline
		// #ifndef __SYNTHESIS__
		// printf("in for loop i=%d, local_pref=%f\n",i,local_pref);
		// #endif
		if (local_pref>=x){
			break;
		}
		prev_pref=local_pref;
		local_pref+=TLev4[i];
	}
	i=i-1;
	// //Last layer: output the current index (min index whose prefixsum>=x)
	// data_out.start=pipstart+i;
	// //No need to calculate the new x to be found in the next layer
	// data_out.newx=data_in.newx-prev_pref;
	// ind_newxout.write(data_out);
	ind_arr[index_learners]=i;
}

//top: declare output_indices[#learners];
// for index_learners=0 to #learners: call above 3+below function in a dataflow{}
// void process_indices(hls::stream<sibit_io> &ind_newxin, int output_indices[],int index_learners){
// 	sibit_io data_in=ind_newxin.read();
// 	output_indices[index_learners]=data_in.newx;
// }

//insert ind,upd,pn: from top module logic/train
//insert signal: from cpu
void Top_tree(int insert_signal,int insert_ind,int upd, float pn[],int ind_o[],int rngs [N_learner]){

	#pragma HLS INTERFACE m_axi port=pn bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=ind_o bundle=gmem4 offset=slave
	#pragma HLS INTERFACE m_axi port=rngs bundle=gmem4 offset=slave

	#pragma HLS INTERFACE s_axilite port=pn bundle=control
	#pragma HLS INTERFACE s_axilite port=ind_o bundle=control
	#pragma HLS INTERFACE s_axilite port=rngs bundle=control


	#pragma HLS INTERFACE s_axilite port=insert_signal bundle=control
	#pragma HLS INTERFACE s_axilite port=insert_ind bundle=control
	#pragma HLS INTERFACE s_axilite port=upd bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	Tree init_fan4;
	#pragma HLS bind_storage variable=init_fan4 type=RAM_1P impl=URAM
	//===========================Insert+Update for populating replay =======finish this later=========
	int p_ini=1;
	#ifndef __SYNTHESIS__
	printf("Tree content before insertion\n");
	for (int i=insert_ind;i<insert_ind+N_actor;i++){
		printf("Actor at index %d: \n",i);
		printf("TLev3 index %d priority %d\n",i,init_fan4.TLev3[i]);
		printf("TLev2 index %d priority %d\n",i/K,init_fan4.TLev2[i/K]); //parent0
		printf("TLev1 index %d priority %d\n",(i/K)/K,init_fan4.TLev1[(i/K)/K]); //parent1
		printf("TLev0 priority %d\n\n",init_fan4.TLev0); //parent2
	}
	#endif 

	// if (insert_signal==1){ //Uncomment afterwards as needed
		for (int i=insert_ind;i<insert_ind+N_actor;i++){
			init_fan4.TLev3[i]=p_ini;
			init_fan4.TLev2[i/K]+=p_ini; //parent0
			init_fan4.TLev1[(i/K)/K]+=p_ini; //parent1
			init_fan4.TLev0+=p_ini; //parent2
		}
	// }

	#ifndef __SYNTHESIS__
	printf("Tree content after insertion\n");
	for (int i=insert_ind;i<insert_ind+N_actor;i++){
		printf("Actor at index %d: \n",i);
		printf("TLev3 index %d priority %d\n",i,init_fan4.TLev3[i]);
		printf("TLev2 index %d priority %d\n",i/K,init_fan4.TLev2[i/K]); //parent0
		printf("TLev1 index %d priority %d\n",(i/K)/K,init_fan4.TLev1[(i/K)/K]); //parent1
		printf("TLev0 priority %d\n\n",init_fan4.TLev0); //parent2
	}
	#endif 

	//=====================populate tree for testbench purposes=============
	#ifndef __SYNTHESIS__
	printf("Initializing tree for testbench\n");
	for (int i=0;i<Lev2_Width;i++){ //16
		init_fan4.TLev2[i]=0+1+2+3;
		for (int j=0;j<K;j++){ //0*4 ~16*4-1 (0123, 0123....64 times)
			init_fan4.TLev3[i*K+j]=j;
		}
	}

	for (int i=0;i<K;i++){ //4
		init_fan4.TLev1[i]=4*(0+1+2+3); //24
	}

	init_fan4.TLev0=(0+1+2+3)*16; //root
	#endif 

	//===========================RNG and sampling=============================
	hls::stream<sibit_io> l1_out_pipe;	
	hls::stream<sibit_io> l2_out_pipe;
	hls::stream<sibit_io> l3_out_pipe;
	// hls::stream<sibit_io> l3_out_pipe;	
	#pragma HLS STREAM variable=l1_out_pipe depth=8 
	#pragma HLS STREAM variable=l2_out_pipe depth=8 
	#pragma HLS STREAM variable=l3_out_pipe depth=8 
	// #pragma HLS STREAM variable=l3_out_pipe depth=8 
	#pragma HLS bind_storage variable=l1_out_pipe type=FIFO impl=bram
	#pragma HLS bind_storage variable=l2_out_pipe type=FIFO impl=bram
	#pragma HLS bind_storage variable=l3_out_pipe type=FIFO impl=bram
	// #pragma HLS bind_storage variable=l3_out_pipe type=FIFO impl=bram

	//The following only used for simulation
	// int rngs [N_learner]={1,4,76,5,25,37,1000000,6}; //should return 1,3,51,3,?,?,2,3

	// int ind_o[N_learner]; //on the top port
	int par_m[N_learner][D-1]; //memoinization array for parent nodes, useful in later update process
	samp_main_loop:for (int i=0;i<N_learner;i++){
		#pragma DATAFLOW
		// #ifndef __SYNTHESIS__
		// printf("Entered Dataflow\n");
		// #endif 
		Sib_Iterator_l1(init_fan4.TLev1,rngs[i],l1_out_pipe,par_m,i);
		// #ifndef __SYNTHESIS__
		// printf("l1\n");
		// #endif 
		Sib_Iterator_l2(init_fan4.TLev2,l1_out_pipe,l2_out_pipe,par_m,i);
		Sib_Iterator_l3(init_fan4.TLev3,l2_out_pipe,l3_out_pipe,par_m,i);
		// #ifndef __SYNTHESIS__
		// printf("l2\n");
		// #endif
		// Sib_Iterator_l3(init_fan4.TLev3,l2_out_pipe,l3_out_pipe,ind_o,i);
		Sib_Iterator_l4(init_fan4.TLev4,l3_out_pipe,ind_o,i);
		// #ifndef __SYNTHESIS__
		// printf("l3\n");
		// #endif
		// process_indices(l3_out_pipe, ind_o,i);
		// #ifndef __SYNTHESIS__
		// printf("Proc\n");
		// #endif
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

	//===========================Update=============================
	#ifndef __SYNTHESIS__
	printf("Tree content before update\n");
	for (int i=0;i<N_learner;i++){
		printf("Learner at index %d: \n",i);
		printf("TLev0 with priority %d\n",init_fan4.TLev0);
		printf("TLev1 index %d with priority %d\n",par_m[i][0],init_fan4.TLev1[par_m[i][0]]);
		printf("TLev2 index %d with priority %d\n",par_m[i][1],init_fan4.TLev2[par_m[i][1]]);
		printf("TLev3 index %d with priority %d\n\n",ind_o[i],init_fan4.TLev3[ind_o[i]]);
	}
	#endif 
	// if (upd==1){ //Uncomment later as needed
		int pnint [N_learner];
		for (int i=0;i<N_learner;i++){
			pnint[i]=pn[i];
		}
		upd_main_loop:for (int i=0;i<N_learner;i++){
			#pragma pipeline
			init_fan4.TLev0-=pnint[i];
			init_fan4.TLev1[par_m[i][0]]-=pnint[i];
			init_fan4.TLev2[par_m[i][1]]-=pnint[i];
			init_fan4.TLev3[ind_o[i]]-=pnint[i];
		}
	// }
	#ifndef __SYNTHESIS__
	printf("Tree content after update\n");
	for (int i=0;i<N_learner;i++){
		printf("Learner at index %d: \n",i);
		printf("TLev0 with priority %d\n",init_fan4.TLev0);
		printf("TLev1 index %d with priority %d\n",par_m[i][0],init_fan4.TLev1[par_m[i][0]]);
		printf("TLev2 index %d with priority %d\n",par_m[i][1],init_fan4.TLev2[par_m[i][1]]);
		printf("TLev3 index %d with priority %d\n\n",ind_o[i],init_fan4.TLev3[ind_o[i]]);
	}
	#endif 
}

}



