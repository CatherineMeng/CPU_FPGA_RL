#include "./learners_test.h"
// #include "./rmm.h"

extern "C"{

// ind: increment from 0 to BATCHS/BSIZE
//in total (from outer loop in top): need to read (BATCHS/BSIZE)*LL time blockvec = BATCHS*LL numbers
void loadIn(blockvec In[],  hls::stream<blockvec> &Inrows,const int LL,int ind){
	for (int i = 0; i < L1; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[ind*L1+i]);
	}

}



void loadSn(blockvec In[], hls::stream<blockvec> &Inrows,const int LL,int ind){
	for (int i = 0; i < L1; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[ind*L1+i]);
	}
}

// r,a:BSIZE floats;
//Qrows, Qtrows: L3*BSIZE z2, aggregate BSIZE
//delt2_buf_fifo:L3*BSIZE, same content as outs, aggreegate L3 to be used in wu-gradient_compute
// pn_out depth=BATCHS (Assume BSIZE=1!), in general BSIZE*BATCHS (objctv will be called BATCHS times
// Qs: length BATCHS*BSIZE
void objctv(blockvec *r, actvec *action, float gamma, bsbit *done, hls::stream<blockvec> &Qrows,hls::stream<blockvec> &Qtrows, 
	hls::stream<w3blockvec> &delt2_buf_fifo, int ind, hls::stream<ap_axiu<32,0,0,0>> &pn_out, float *Qs,float *Loss_sqrt){
	#pragma HLS aggregate variable=Qrows
	#pragma HLS aggregate variable=Qtrows

	blockvec r_local=r[ind];
	actvec action_local=action[ind];
	bsbit done_local=done[ind];
	// Get argmax target Q vals of size BSIZE
	blockvec argmax_tq={0};
	for (int i=0;i<L1;i++){//used to be L3, just use L1 for test
		blockvec tmpqt=Qtrows.read();
		if(i<L3){ //original loop bound
		for (int j=0;j<BSIZE;j++){
			// #pragma HLS UNROLL
			if (tmpqt.a[j]>argmax_tq.a[j])
				argmax_tq.a[j]=tmpqt.a[j];
		}
		}

	}
	#ifndef __SYNTHESIS__
	printf("argmax_tq:");
	for (int j=0;j<BSIZE;j++){
		printf("%f ",argmax_tq.a[j]);}
	#endif

	// actderiv(Qrows, hls::stream<blockvec> &Outrows,L3);
	// Get Q vals, calc obj
	w3blockvec d2tmp [BSIZE];
	float Qtransfer[BSIZE] ={0};
	for (int i=0;i<L1;i++){ //used to be L3, just use L1 for test
		// #pragma HLS PIPELINE
		blockvec tmpq=Qrows.read();
		// blockvec tmpout;
		blockvec tmpobj;
		if(i<L3){ //original loop bound
		for (int j=0;j<BSIZE;j++){	
			// #pragma HLS PIPELINE
			if (i==action_local.a[j])
			{
				float actdertmp=(tmpq.a[j]>0)? 1:0; //relu derivative
				// #ifndef __SYNTHESIS__
				// printf("\ntmpq.a[%d]:%f",j,tmpq.a[j]);
				// #endif
				// tmpobj.a[j]=2*(tmpq.a[j]-r.a[j]*argmax_tq.a[j])*actdertmp; 
				float oneb=1-done_local.a[j]; //cast fixed point to float
				tmpobj.a[j]=2*(r_local.a[j]+oneb*gamma*argmax_tq.a[j]-tmpq.a[j])*actdertmp; 
				Qtransfer[j]=tmpq.a[j];
				// float td=(r_local.a[j]+oneb*gamma*argmax_tq.a[j]-tmpq.a[j])*actdertmp;
				// v.data=td; 
				// pn_out.write(v);
				// #ifndef __SYNTHESIS__
				// printf("\nnode %d, sample in batch-tmpobj.a[%d]:%f",i,j,tmpobj.a[j]);
				// #endif
			}
			else
				{tmpobj.a[j]=0;}
			d2tmp[j].a[i]=tmpobj.a[j];
		}
		// outs[i]=(tmpobj);
		// outs.write(tmpobj);
		}

	}

	for (int j=0;j<BSIZE;j++){
		delt2_buf_fifo.write(d2tmp[j]);
		// ===============write TD to other kernel=================
		ap_axiu<32,0,0,0> v;
		// for (int i=0;i<L3;i++){
			// if (i==action_local.a[j])
			// {
				float actdertmp=(Qtransfer[j]>0)? 1:0; //relu derivative
		
				// tmpobj.a[j]=2*(tmpq.a[j]-r.a[j]*argmax_tq.a[j])*actdertmp; 
				float oneb=1-done_local.a[j]; //cast fixed point to float
				float td=(r_local.a[j]+oneb*gamma*argmax_tq.a[j]-Qtransfer[j])*actdertmp;
				
				v.data=td; 
				
			// }
			pn_out.write(v);
			
		// }
		Qs[ind*BSIZE+j]=Qtransfer[j];
		Loss_sqrt[ind*BSIZE+j]=td;
	}
	#ifndef __SYNTHESIS__
	for (int j=0;j<BSIZE;j++){
		printf("\n(index,Qs and Loss_sqrt) out of all BATCHS*BSIZE: (%d, %F, %F)\n",ind*BSIZE+j,Qs[ind*BSIZE+j],Loss_sqrt[ind*BSIZE+j]);
	}
	// for (int i=0;i<L3;i++){
	// 	for (int j=0;j<BSIZE;j++){
	// 		// printf("%f ",delt2_buf_fifo[j].a[i]);
	// 		printf("%f ",d2tmp[j].a[i]);
	// 	}
	// }

	#endif
}



void wa1(float wa1_buf[L1/P3][L2/T3][P3][T3]){
	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4
	WA1partialsum: for(int k=0; k < BSIZE; k++) {
		// a0blockvec a0tmp = a0_buf_fifo.read();
		// w1blockvec d1tmp = delt1_buf_fifo.read();
		// #pragma HLS DATAFLOW
		for(int i = 0; i < L1/P3; i++) {
			for(int j = 0; j < L2/T3; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=wa1_buf inter false
				for(int ii = 0; ii < P3; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T3; jj++) { //3
						// #pragma HLS UNROLL
						wa1_buf[i][j][ii][jj] = i*j/4+jj-ii;
					}
				}
			}
		}
	}
}

void wa2(hls::stream<w3blockvec> &delt2_buf_fifo,float wa2_buf[L2/P4][L3/T4][P4][T4]){
	#pragma HLS array_partition variable=wa2_buf complete  dim=3
	#pragma HLS array_partition variable=wa2_buf complete  dim=4
	#pragma HLS aggregate variable=delt2_buf_fifo

	WA2partialsum: for(int k=0; k < BSIZE; k++) {
		// w1blockvec a1tmp = a1_buf_fifo.read();
		w3blockvec d2tmp = delt2_buf_fifo.read();
		for(int i = 0; i < L2/P4; i++) {
			for(int j = 0; j < L3/T4; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=wa2_buf inter false
				for(int ii = 0; ii < P4; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T4; jj++) { //3
	
						wa2_buf[i][j][ii][jj] = i*j/4+jj-d2tmp.a[j];
					}
				}
			}
		}
	}
}




// void fw_bw(blockvec *A,w1blockvec w1bram[],w3blockvec w2bram[],float bias1[],float bias2[],a0blockvec a0_buf_fifo[BSIZE],float a1_buf_fifo[L2][BSIZE],float delt2_buf_fifo[BSIZE][L3],float delt1_buf[BSIZE][L2]){
// void fw_bw(blockvec *A,w1blockvec w1bram[],w3blockvec w2bram[],float bias1[],float bias2[],float wa1_global[L1/P3][L2/T3][P3][T3],float wa2_global[L2/P4][L3/T4][P4][T4]){
void fw_bw(blockvec *A,blockvec *Atarg,actvec *acts,blockvec *r,bsbit *done,
	float gamma, hls::stream<ap_axiu<32,0,0,0>> &pn_out,
	float wa1_global[L1/P3][L2/T3][P3][T3],float wa2_global[L2/P4][L3/T4][P4][T4],
	float *Qs,float *Loss_sqrt){

	#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave
	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	// #pragma HLS array_partition variable=a0_buf_fifo type=block  factor=8  dim=1
	#pragma HLS array_partition variable=wa1_global complete  dim=3
	#pragma HLS array_partition variable=wa1_global complete  dim=4
	#pragma HLS array_partition variable=wa2_global complete  dim=3
	#pragma HLS array_partition variable=wa2_global complete  dim=4


	// #pragma HLS array_partition variable=w2bram type=cyclic  factor=8

	hls::stream<blockvec> inpipe;
	hls::stream<blockvec> inpipe0;

	// #pragma HLS array_partition variable=outpipe complete
	#pragma HLS STREAM variable=inpipe depth=8 //L1
	#pragma HLS STREAM variable=inpipe0 depth=8 //L1

 
	hls::stream<w3blockvec> delt2_buf_fifo; //delta2 for wu, produced by obj, parallel access on L3 dimension
	// #pragma HLS aggregate variable=delt2_buf_fifo
	#pragma HLS STREAM variable=delt2_buf_fifo depth=1 //BSIZE
	#pragma HLS bind_storage variable=delt2_buf_fifo type=fifo impl=BRAM


	float wa1_buf[L1/P3][L2/T3][P3][T3]={0};
	float wa2_buf[L2/P4][L3/T4][P4][T4]={0};


	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4
	#pragma HLS array_partition variable=wa2_buf complete  dim=3
	#pragma HLS array_partition variable=wa2_buf complete  dim=4



	for(int ind=0; ind<BATCHS; ind++){
		// #pragma HLS DATAFLOW

		loadIn(A, inpipe, L1, ind);
		// test_loadIn(a0_buf_fifo, inpipe);
		loadSn(Atarg, inpipe0, L1, ind);

		objctv(r, acts, gamma, done, inpipe,inpipe0,delt2_buf_fifo, ind,pn_out, Qs,Loss_sqrt);

		wa1(wa1_buf);
		wa2(delt2_buf_fifo,wa2_buf);


	}

// write to global WA
	for(int i = 0; i < L1/P3; i++) {
		for(int j = 0; j < L2/T3; j++) {
		#pragma HLS PIPELINE
		// #pragma HLS dependence variable=wa1_buf inter false
			for(int ii = 0; ii < P3; ii++) {
				for(int jj = 0; jj < T3; jj++) { 
					wa1_global[i][j][ii][jj] = wa1_buf[i][j][ii][jj];
				}
			}
		}
	}
	for(int i = 0; i < L2/P4; i++) {
		for(int j = 0; j < L3/T4; j++) {
		#pragma HLS PIPELINE
		// #pragma HLS dependence variable=wa2_buf inter false
			for(int ii = 0; ii < P4; ii++) {
				for(int jj = 0; jj < T4; jj++) { 
					wa2_global[i][j][ii][jj] = wa2_buf[i][j][ii][jj];
				}
			}
		}
	}
	// WA2partialsum: for(int k=0; k < BSIZE; k++) {}

}

void learners_top(blockvec *S, blockvec *Snt, actvec *acts, blockvec *r, float gamma, float alpha, bsbit *done, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],float *bias1_out,float *bias2_out,int wsync, /*Learners args*/
	float *Qs,float *Loss_sqrt,/*Logging args*/
	hls::stream<ap_axiu<32,0,0,0>> &pn_out/*Replay args*/){
	#pragma HLS INTERFACE m_axi port=S bundle=gmem1 offset=slave
	#pragma HLS INTERFACE m_axi port=Snt bundle=gmem2 offset=slave
	#pragma HLS INTERFACE m_axi port=acts bundle=gmem5 offset=slave
	#pragma HLS INTERFACE m_axi port=r bundle=gmem6 offset=slave
	#pragma HLS INTERFACE m_axi port=done bundle=gmem7 offset=slave
	#pragma HLS INTERFACE m_axi port=w1bram_out bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=w2bram_out bundle=gmem4 offset=slave
	#pragma HLS INTERFACE m_axi port=bias1_out bundle=gmem8 offset=slave
	#pragma HLS INTERFACE m_axi port=bias2_out bundle=gmem8 offset=slave
	#pragma HLS INTERFACE m_axi port=Qs bundle=gmem9 offset=slave
	#pragma HLS INTERFACE m_axi port=Loss_sqrt bundle=gmem9 offset=slave

	#pragma HLS INTERFACE s_axilite port=S bundle=control
	#pragma HLS INTERFACE s_axilite port=Snt bundle=control
	#pragma HLS INTERFACE s_axilite port=acts bundle=control
	#pragma HLS INTERFACE s_axilite port=r bundle=control
	#pragma HLS INTERFACE s_axilite port=done bundle=control
	#pragma HLS INTERFACE s_axilite port=w1bram_out bundle=control
	#pragma HLS INTERFACE s_axilite port=w2bram_out bundle=control
	#pragma HLS INTERFACE s_axilite port=bias1_out bundle=control
	#pragma HLS INTERFACE s_axilite port=bias2_out bundle=control
	#pragma HLS INTERFACE s_axilite port=Qs bundle=control
	#pragma HLS INTERFACE s_axilite port=Loss_sqrt bundle=control


	#pragma HLS INTERFACE s_axilite port=gamma bundle=control
	#pragma HLS INTERFACE s_axilite port=alpha bundle=control
	#pragma HLS INTERFACE s_axilite port=wsync bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control


	#pragma HLS INTERFACE axis port=pn_out 

	static w1blockvec w1bram[L1]; //w1
	#pragma HLS aggregate variable=w1bram
	#pragma HLS aggregate variable=w1bram_out
	static w3blockvec w2bram[L2]; //w2
	#pragma HLS aggregate variable=w2bram
	#pragma HLS aggregate variable=w2bram_out
	#pragma HLS bind_storage variable=w2bram type=RAM_2P impl=bram

	static w1blockvec w1bram_t[L1]; //w1_target
	#pragma HLS aggregate variable=w1bram_t
	static w3blockvec w2bram_t[L2]; //w2
	#pragma HLS aggregate variable=w2bram_t
	#pragma HLS bind_storage variable=w2bram_t type=RAM_2P impl=bram 


	#ifndef __SYNTHESIS__
	printf("\nacts:\n");
	for(int j = 0; j < BATCHS; j++) {
		for(int i = 0; i < BSIZE; i++) {
			printf("%d ",acts[j].a[i]);  //BS cols
		}
	}
	printf("\nr:\n");
	for(int j = 0; j < BATCHS; j++) {
		for(int i = 0; i < BSIZE; i++) {
			printf("%f ",r[j].a[i]);  //BS cols
		}
	}
	printf("\ndone:\n");
	for(int j = 0; j < BATCHS; j++) {
		for(int i = 0; i < BSIZE; i++) 	{
		// printf("%s ",done.a[i].to_string(10).c_str());  //BS cols
			printf("%d ",done[j].a[i]);  //BS cols
		}
	}
	#endif


//	Init on-chip memory

	static w1blockvec bias1;
	static w3blockvec bias2;

	#pragma HLS aggregate variable=bias1
	#pragma HLS aggregate variable=bias2

	if (wsync==0){ //Init. Q network & target network (only executed exactly once in all iterations!)

	#ifndef __SYNTHESIS__
	printf("\nWeight init.\n");
	#endif
		for (int i=0; i<L1;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				w1bram[i].a[j]=w1list[i][j];
			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
			for  (int j=0; j<L3;j++){
				if (j<2) {w2bram[i].a[j]=w2list_or[i][j];}
				else {w2bram[i].a[j]=w2list_or[i][j-2];}
			}
		}
		for (int i=0; i<L2;i++){
			bias1.a[i]=bias1_list[i];
		}
		for (int i=0; i<L3;i++){
			bias2.a[i]=bias2_list[i];
		}

	}


	float wa1_global[L1/P3][L2/T3][P3][T3]={0};
	float wa2_global[L2/P4][L3/T4][P4][T4]={0}; 

	#pragma HLS array_partition variable=wa1_global complete  dim=3
	#pragma HLS array_partition variable=wa1_global complete  dim=4
	#pragma HLS array_partition variable=wa2_global complete  dim=3
	#pragma HLS array_partition variable=wa2_global complete  dim=4

	float Qs_local[BATCHS*BSIZE];
	float Loss_sqrt_local[BATCHS*BSIZE];
	fw_bw(S,Snt,acts,r,done, gamma, pn_out, wa1_global,wa2_global,Qs_local,Loss_sqrt_local);


	#pragma HLS array_partition variable=w1bram type=cyclic  factor=2
	#pragma HLS array_partition variable=w2bram type=cyclic  factor=8
	// WU: Substract -SGD (Add if SGA) WA from wbrams
	for(int i = 0; i < L1/P3; i++) {
		for(int j = 0; j < L2/T3; j++) {
			for(int ii = 0; ii < P3; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w1bram inter false
				for(int jj = 0; jj < T3; jj++) { 
					w1bram[i*P3+ii].a[j*T3+jj] -=wa1_global[i][j][ii][jj];
					// w1bram_out[i*P3+ii].a[j*T3+jj]=w1bram[i*P3+ii].a[j*T3+jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nw1bram updated.\n");
	#endif

	for(int i = 0; i < L2/P4; i++) {
		for(int j = 0; j < L3/T4; j++) {
		// #pragma HLS PIPELINE
		// #pragma HLS dependence variable=w2bram inter false
			for(int ii = 0; ii < P4; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w2bram inter false
				for(int jj = 0; jj < T4; jj++) { 
					w2bram[i*P4+ii].a[j*T4+jj] -=wa2_global[i][j][ii][jj];
					// w2bram_out[i*P4+ii].a[j*T4+jj]=w2bram[i*P4+ii].a[j*T4+jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nw2bram updated.\n");
	#endif

	for (int i=0; i<L2;i++){
		bias1.a[i]+=0.2;
		// bias1_out[i]=bias1.a[i];
	}
	for (int i=0; i<L3;i++){
		bias2.a[i]-=0.12;
		// bias2_out[i]=bias2.a[i];
	}

	#ifndef __SYNTHESIS__
	printf("\nbiases updated.\n");
	#endif
	//sync weights to cpu
	// {
	// #pragma HLS DATAFLOW
	wb1wb:for(int i = 0; i < L1; i++) {
		w1blockvec tmpw1b;
		#pragma HLS PIPELINE
		for(int jj = 0; jj < L2; jj++) { 
			tmpw1b.a[jj]=w1bram[i].a[jj];
		}
		w1bram_out[i]=tmpw1b;
	}
	wb2wb:for(int i = 0; i < L2; i++) {
		#pragma HLS PIPELINE
		for(int jj = 0; jj < L3; jj++) {
			w2bram_out[i].a[jj]=w2bram[i].a[jj];
		}
	}
	for (int i=0; i<L2;i++){
		bias1_out[i]=bias1.a[i];
	}
	for (int i=0; i<L3;i++){
		bias2_out[i]=bias2.a[i];
	}
	// }	
	for (int i=0; i<BATCHS*BSIZE;i++){
		Qs[i]=Qs_local[i];
		Loss_sqrt[i]=Loss_sqrt_local[i];
	}

	#ifndef __SYNTHESIS__
	printf("\nQs and Loss updated.\n");
	#endif
	#ifndef __SYNTHESIS__
	printf("\nTransfer finished.\n");
	#endif

}

}