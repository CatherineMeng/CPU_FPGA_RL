#include "./topl_new.h"

extern "C"{

// ind: increment from 0 to BATCHS/BSIZE
//in total (from outer loop in top): need to read (BATCHS/BSIZE)*LL time blockvec = BATCHS*LL numbers
void loadIn(blockvec In[],  hls::stream<a0blockvec> &a0_buf_fifo,hls::stream<blockvec> &Inrows,const int LL,int ind){
	blockvec Inrows_local[LL];
	#ifndef __SYNTHESIS__
	printf("\nInput S read:\n");
	#endif
	for (int i = 0; i < LL; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[ind*LL+i]);
		Inrows_local[i]=In[ind*LL+i];
		#ifndef __SYNTHESIS__
		for (int j = 0; j < BSIZE; j++){printf("%.8f ",In[ind*LL+i].a[j]);}
		#endif
	}
	// get a0_buf_fifo for WA
	for (int j = 0; j < BSIZE; j++){
		a0blockvec a0tmp;
		for (int i = 0; i < L1; i++){
			#pragma HLS PIPELINE
			// a0tmp.a[i]=In[ind*LL+i].a[j];
			a0tmp.a[i]=Inrows_local[i].a[j];
		}
		a0_buf_fifo.write(a0tmp);

	}
}



void loadSn(blockvec In[], hls::stream<blockvec> &Inrows,const int LL,int ind){
	#ifndef __SYNTHESIS__
	printf("\nInput Snt read:\n");
	#endif
	for (int i = 0; i < LL; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[ind*LL+i]);
		#ifndef __SYNTHESIS__
		for (int j = 0; j < BSIZE; j++){printf("%.8f ",In[ind*LL+i].a[j]);}
		#endif
	}
}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// void fw_l1(hls::stream<blockvec> &Inrows, float C[BSIZE/P][64/T][P][T],w1blockvec Wcols[], hls::stream<blockvec> &Crows, float a1_buf_fifo[L2][BSIZE], const int LL,const int LN) {
// void fw_l1(hls::stream<blockvec> &Inrows, float z1_buf[BSIZE/P][L2/T][P][T], float a1_buf_fifo[L2][BSIZE],float bias[], w1blockvec Wcols[], hls::stream<blockvec> &Crows, bsbit actder[L2],const int LL,const int LN) {
void fw_l1(hls::stream<blockvec> &Inrows, hls::stream<w1blockvec> &a1_buf_fifo, w1blockvec bias, w1blockvec Wcols[], hls::stream<blockvec> &Crows, hls::stream<bsbit> &actder_fifo,const int LL,const int LN) {

	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=actder_fifo
	#pragma HLS aggregate variable=bias
	// #pragma HLS ARRAY_PARTITION variable=z1_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z1_buf dim=4 complete
    #pragma HLS dependence class=array variable=z1_buf_local type=inter dependent=false
    #pragma HLS dependence class=array variable=z1_buf_local type=intra dependent=false
	float z1_buf_local[BSIZE/P][L2/T][P][T];
	#pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=4 complete
	partialsum: for(int k=0; k < LL; k++) {
		blockvec tempA = Inrows.read();
		w1blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P; i++) {
			for(int j = 0; j < LN/T; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=z1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					#pragma HLS UNROLL
					for(int jj = 0; jj < T; jj++) {
						#pragma HLS UNROLL
						//#pragma HLS dependence variable=C inter false
						if (k==0) z1_buf_local[i][j][ii][jj]=tempA.a[i*P+ii] * tempB.a[j*T+jj];
						else if (k==LL-1) z1_buf_local[i][j][ii][jj] += (tempA.a[i*P+ii] * tempB.a[j*T+jj] + bias.a[j*T+jj]);
						else z1_buf_local[i][j][ii][jj] += tempA.a[i*P+ii] * tempB.a[j*T+jj];
					}
				}
			}
		}
	}
	
	//write out to stream: next fw

	#ifndef __SYNTHESIS__
	printf("\nz1_buf content:\n");//should be L2 rows, BSIZE columns
	#endif
	// get a1_buf_fifo for WA, find actder, write activatpon out to stream
	writeout: for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf_fifo partition
		for(int jj = 0; jj < T; jj++) {
			blockvec tempC;
			bsbit actdertmp;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P; i++) {
				#pragma HLS PIPELINE
				for(int ii = 0; ii < P; ii++) {
					 //activation
					float tmpz=(z1_buf_local[i][j][ii][jj]>0)? z1_buf_local[i][j][ii][jj]:0;
					// a1_buf_fifo[i*P+ii].a[j*T+jj]=tmpz;
					actdertmp.a[i*P+ii]=(z1_buf_local[i][j][ii][jj]>0)? 1:0; //activation derivative

					tempC.a[i*P+ii]=tmpz; //activation
					// #ifndef __SYNTHESIS__
					// printf("%.8f ",z1_buf_local[i][j][ii][jj]);
					// #endif
				}
			}
			actder_fifo.write(actdertmp);
			Crows.write(tempC);
			// #ifndef __SYNTHESIS__
			// printf("\n");
			// #endif
		}
	}
	//write out to stream: a1_buf

	writeouta1fifo: for(int i = 0; i < BSIZE/P; i++) {
		for(int ii = 0; ii < P; ii++) {
			w1blockvec a1buftmp;
			for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf_fifo partition
				#pragma HLS PIPELINE
				for(int jj = 0; jj < T; jj++) {
					 //activation
					// float tmpz=(z1_buf_local[i][j][ii][jj]>0)? z1_buf_local[i][j][ii][jj]:0;
					a1buftmp.a[j*T+jj]=(z1_buf_local[i][j][ii][jj]>0)? z1_buf_local[i][j][ii][jj]:0;
				}
			}

			a1_buf_fifo.write(a1buftmp);

		}
	}
}


void fw_l1_targ(hls::stream<blockvec> &Inrows,  w1blockvec bias, w1blockvec Wcols[], hls::stream<blockvec> &Crows, const int LL,const int LN) {

	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=bias
	// #pragma HLS ARRAY_PARTITION variable=z1_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z1_buf dim=4 complete

	float z1_buf_local[BSIZE/P][L2/T][P][T];
	#pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=4 complete
	partialsum: for(int k=0; k < LL; k++) {
		blockvec tempA = Inrows.read();
		w1blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P; i++) {
			for(int j = 0; j < LN/T; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=z1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					#pragma HLS UNROLL
					for(int jj = 0; jj < T; jj++) {
						#pragma HLS UNROLL
						//#pragma HLS dependence variable=C inter false
						if (k==0) z1_buf_local[i][j][ii][jj]=tempA.a[i*P+ii] * tempB.a[j*T+jj];
						else if (k==LL-1) z1_buf_local[i][j][ii][jj] += (tempA.a[i*P+ii] * tempB.a[j*T+jj] + bias.a[j*T+jj]);
						else z1_buf_local[i][j][ii][jj] += tempA.a[i*P+ii] * tempB.a[j*T+jj];
					}
				}
			}
		}
	}
	
	//write out to stream: next fw

	#ifndef __SYNTHESIS__
	printf("\n targ z1_buf content:\n");//should be L2 rows, BSIZE columns
	#endif
	// get a1_buf_fifo for WA, find actder, write activatpon out to stream
	writeout: for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf_fifo partition
		for(int jj = 0; jj < T; jj++) {
			blockvec tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P; i++) {
				#pragma HLS PIPELINE
				for(int ii = 0; ii < P; ii++) {
					 //activation
					tempC.a[i*P+ii]=(z1_buf_local[i][j][ii][jj]>0)? z1_buf_local[i][j][ii][jj]:0; //activation
					// #ifndef __SYNTHESIS__
					// printf("%.8f ",z1_buf_local[i][j][ii][jj]);
					// #endif
				}
			}
			Crows.write(tempC);
			// #ifndef __SYNTHESIS__
			// printf("\n");
			// #endif
		}
	}
}


// wu(C)

//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// void fw_l2(hls::stream<blockvec> &Inrows, float z2_buf[BSIZE/P2][L3/T2][P2][T2], float bias[],w3blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN) {
void fw_l2(hls::stream<blockvec> &Inrows, w3blockvec bias,w3blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN) {
	// #pragma HLS INLINE
	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=bias
	// float C[BSIZE/P2][3/T2][P2][T2]={0};
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=4 complete

	float z2_buf_local[BSIZE/P2][L3/T2][P2][T2];
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=4 complete
	#pragma HLS bind_storage variable=z2_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < LL; k++) {
	blockvec tempA = Inrows.read();
	w3blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P2; i++) {
			partialsum_l2: for(int j = 0; j < LN/T2; j++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=z2_buf_local type=inter dependent=false
				for(int ii = 0; ii < P2; ii++) {
					#pragma HLS UNROLL
					for(int jj = 0; jj < T2; jj++) { //3
						#pragma HLS UNROLL
						// z2_buf_local[i][j][ii][jj] = z2_buf_local[i][j][ii][jj] + tempA.a[i*P2+ii] * tempB.a[j*T2+jj];
						if (k==0) z2_buf_local[i][j][ii][jj]=tempA.a[i*P2+ii] * tempB.a[j*T2+jj];
						else if (k==LL-1) z2_buf_local[i][j][ii][jj] += (tempA.a[i*P2+ii] * tempB.a[j*T2+jj] + bias.a[j*T2+jj]);
						else z2_buf_local[i][j][ii][jj] += tempA.a[i*P2+ii] * tempB.a[j*T2+jj];
					}
				}
			}
		}
	}


	#ifndef __SYNTHESIS__
	printf("\nz2_buf content:\n");

	for(int j = 0; j < LN/T2; j++) { //this factor consistent with a1_buf_fifo partition
		for(int jj = 0; jj < T2; jj++) {
			for(int i = 0; i < BSIZE/P2; i++) {
				for(int ii = 0; ii < P2; ii++) {
					printf("%.8f ",z2_buf_local[i][j][ii][jj]);
				}
			}
			printf("\n");
		}
	}
	#endif
	// write out to stream
	for(int j = 0; j < LN/T2; j++) {
		for(int jj = 0; jj < T2; jj++) {
			blockvec tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P2; i++) {
				#pragma HLS PIPELINE
				for(int ii = 0; ii < P2; ii++) {
					tempC.a[i*P2+ii]=z2_buf_local[i][j][ii][jj];
				}
			}

			Crows.write(tempC);
		}
	}

}


// r,a:BSIZE floats;
//Qrows, Qtrows: L3*BSIZE z2, aggregate BSIZE
//act_deriv(Qrows) hadamard* should be delt 2
//outs:L3*BSIZE, should be delt2 (aggregate BSIZE, used by bw)
//delt2_buf_fifo:L3*BSIZE, same content as outs, aggreegate L3 to be used in wu-gradient_compute
// void objctv(blockvec r, actvec action, hls::stream<blockvec> &Qrows,hls::stream<blockvec> &Qtrows,
// 	blockvec outs[],float delt2_buf_fifo[BSIZE][L3]){
// void objctv(blockvec *r, actvec *action, float gamma, bsbit *done, hls::stream<blockvec> &Qrows,hls::stream<blockvec> &Qtrows, hls::stream<blockvec> &outs,hls::stream<w3blockvec> &delt2_buf_fifo, int ind){
void objctv(blockvec *r, actvec *action, float gamma, bsbit *done, hls::stream<blockvec> &Qrows,hls::stream<blockvec> &Qtrows, hls::stream<blockvec> &outs,
	hls::stream<w3blockvec> &delt2_buf_fifo, int ind, hls::stream<ap_axiu<32,0,0,0>> &pn_out, float *Qs,float *Loss_sqrt){
	#pragma HLS aggregate variable=Qrows
	#pragma HLS aggregate variable=Qtrows
	#pragma HLS aggregate variable=r
	#pragma HLS aggregate variable=action
	#pragma HLS aggregate variable=done
	#pragma HLS aggregate variable=delt2_buf_fifo
	#pragma HLS aggregate variable=pn_out

	blockvec r_local=r[ind];
	actvec action_local=action[ind];
	bsbit done_local=done[ind];
	// Get argmax target Q vals of size BSIZE

	// #ifndef __SYNTHESIS__
	// printf("\nobjctv read\n");
	// #endif
	blockvec argmax_tq={0};
	for (int i=0;i<L3;i++){
		#pragma HLS PIPELINE II=2
		blockvec tmpqt=Qtrows.read();
		for (int j=0;j<BSIZE;j++){
			#pragma HLS UNROLL
			if (tmpqt.a[j]>argmax_tq.a[j])
				argmax_tq.a[j]=tmpqt.a[j];
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
	float Qtransfer[BSIZE];
	float TDtransfer[BSIZE];
	for (int i=0;i<L3;i++){
		// #pragma HLS PIPELINE
		blockvec tmpq=Qrows.read();
		// blockvec tmpout;
		blockvec tmpobj;
		
		for (int j=0;j<BSIZE;j++){
			
			#pragma HLS PIPELINE
			if (i==action_local.a[j]) //only happens BSIZE times
			{
				float actdertmp=(tmpq.a[j]>0)? 1:0; //relu derivative
				#ifndef __SYNTHESIS__
				printf("\ntmpq.a[%d]:%f",j,tmpq.a[j]);
				#endif
				// tmpobj.a[j]=2*(tmpq.a[j]-r.a[j]*argmax_tq.a[j])*actdertmp; 
				float oneb=1-done_local.a[j]; //cast fixed point to float
				tmpobj.a[j]=2*(r_local.a[j]+oneb*gamma*argmax_tq.a[j]-tmpq.a[j])*actdertmp; 
				Qtransfer[j]=tmpq.a[j];
				TDtransfer[j]=tmpobj.a[j]/2;
				#ifndef __SYNTHESIS__
				printf("\nnode %d, sample in batch-tmpobj.a[%d]:%f",i,j,tmpobj.a[j]);
				#endif
			}
			else
				tmpobj.a[j]=0;
			//write to delt2_buf_fifo
			d2tmp[j].a[i]=tmpobj.a[j];
		}
		// outs[i]=(tmpobj);
		outs.write(tmpobj);
	}


	for (int i=0;i<BSIZE;i++){
		delt2_buf_fifo.write(d2tmp[i]);
		ap_axiu<32,0,0,0> v;
		float td=TDtransfer[i];
		v.data=td; 
		pn_out.write(v);
		Qs[ind*BSIZE+i]=Qtransfer[i];
		Loss_sqrt[ind*BSIZE+i]=td;
	}

	#ifndef __SYNTHESIS__
	for (int j=0;j<BSIZE;j++){
		printf("\n(index,Qs and Loss_sqrt) out of all BATCHS*BSIZE: (%d, %F, %F)\n",ind*BSIZE+j,Qs[ind*BSIZE+j],Loss_sqrt[ind*BSIZE+j]);
	}

	printf("\ndelt2_buf_fifo content:\n");//should be L3 rows, BSIZE columns
	for (int i=0;i<L3;i++){
		for (int j=0;j<BSIZE;j++){
			// printf("%f ",delt2_buf_fifo[j].a[i]);
			printf("%f ",d2tmp[j].a[i]);
		}
	}
	#endif
}


//Inrows: LN blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LL blockvecs (each batchsize)
// void sub_backmm2(hls::stream<blockvec> &Inrows, 
// 	w3blockvec Wcols0, w3blockvec Wcols1, w3blockvec Wcols2, w3blockvec Wcols3,
// 	w3blockvec Wcols4,w3blockvec Wcols5,w3blockvec Wcols6,w3blockvec Wcols7, hls::stream<blockvec> &Crows, 
// 	float delt1_buf[BSIZE/Pb][L3/Tb][Pb][Tb], const int LL,const int LN,int ind) {
// void sub_backmm2(blockvec Inrows[], w1blockvec Wcols[], bsbit actder[L2],w1blockvec delt1_buf[BSIZE], const int LL,const int LN){
void sub_backmm2(hls::stream<blockvec> &Inrows, w1blockvec Wcols[], hls::stream<bsbit> &actder_fifo, hls::stream<w1blockvec> &delt1_buf_fifo, const int LL,const int LN){

	#pragma HLS aggregate variable=Inrows
	// #pragma HLS aggregate variable=Wcols1s
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=delt1_buf_fifo
	#pragma HLS aggregate variable=actder_fifo


	float delt1_buf_local[BSIZE/P][L2/T][P][T]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=4 complete
	
	#pragma HLS bind_storage variable=delt1_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < L3; k++) { //LL is L3
		blockvec tempA = Inrows.read();
		w1blockvec tempB = Wcols[k]; //tempB size L2
    #pragma HLS aggregate variable=tempA
     // #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P; i++) {
			for(int j = 0; j < L2/T; j++) { //LN is L2
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=delt1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T; jj++) { 
						// #pragma HLS UNROLL
						// delt1_buf[i][j][ii][jj] = delt1_buf[i][j][ii][jj] + tempA.a[i*Pb+ii] * (*arraywq[j*Tb+jj]).a[k]; //*arraywq: because wcols partitioned in cyclic manner, adjacent indices are in different banks
						// delt1_buf_local[i][j][ii][jj] = delt1_buf_local[i][j][ii][jj] + tempA.a[i*P+ii] * (Wcols[j*T+jj]).a[k];
						delt1_buf_local[i][j][ii][jj] = delt1_buf_local[i][j][ii][jj] + tempA.a[i*P+ii] * tempB.a[j*T+jj];
					}
				}
			}
		}

	}
	// #ifndef __SYNTHESIS__
	// printf("\ndelt1_buf content before z1 :\n\n");//should be L3 rows, BSIZE columns
	// for(int j = 0; j < L2/T; j++) { //this factor consistent with a1_buf_fifo partition
	// 	for(int jj = 0; jj < T; jj++) {
	// 		for(int i = 0; i < BSIZE/P; i++) {
	// 			for(int ii = 0; ii < P; ii++) {
	// 				printf("%.8f ",delt1_buf_local[i][j][ii][jj]);
	// 			}
	// 		}
	// 		printf("\n");

	// 	}
	// }
	// #endif

	multactder:for(int j = 0; j < L2/T; j++) { 
		for(int jj = 0; jj < T; jj++) {
			bsbit actdertmp = actder_fifo.read();
			for(int i = 0; i < BSIZE/P; i++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=delt1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					// delt times z1 relu derivative
					float tmpdelt=delt1_buf_local[i][j][ii][jj];
					// delt1_buf_local[i][j][ii][jj] = (actder_fifo[j*T+jj].a[i*P+ii]!=0)? tmpdelt:0;
					delt1_buf_local[i][j][ii][jj] = (actdertmp.a[i*P+ii]!=0)? tmpdelt:0;
				
				}
			}
		}
	}

	for(int i = 0; i < BSIZE/P; i++) {
		for(int ii = 0; ii < P; ii++) {
			w1blockvec d1tmp;
			#pragma HLS aggregate variable=d1tmp
			writeout:for(int j = 0; j < L2/T; j++) { 
				#pragma HLS PIPELINE
				for(int jj = 0; jj < T; jj++) {
					// delt1_buf_fifo[i*P+ii].a[j*T+jj] = delt1_buf_local[i][j][ii][jj];
					d1tmp.a[j*T+jj] = delt1_buf_local[i][j][ii][jj];
				}
			}
			delt1_buf_fifo.write(d1tmp);
		}
	}

	// #ifndef __SYNTHESIS__
	// printf("\ndelt1_buf content after z1:\n\n");//should be L3 rows, BSIZE columns
	// for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf_fifo partition
	// 	for(int jj = 0; jj < T; jj++) {
	// 		for(int i = 0; i < BSIZE/P; i++) {
	// 			for(int ii = 0; ii < P; ii++) {
	// 				printf("%.8f ",delt1_buf[i*P+ii].a[j*T+jj]);
	// 			}
	// 		}
	// 		printf("\n");

	// 	}
	// }
	// #endif

}



void wa1(hls::stream<a0blockvec> &a0_buf_fifo, hls::stream<w1blockvec> &delt1_buf_fifo, float wa1_buf[L1/P3][L2/T3][P3][T3], w1blockvec gr_bias1){
	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4

	// #pragma HLS aggregate variable=a0_buf_fifo
	// #pragma HLS aggregate variable=delt1_buf_fifo
	// #pragma HLS aggregate variable=bias1
	#pragma HLS aggregate variable=gr_bias1

	WA1partialsum: for(int k=0; k < BSIZE; k++) {
		a0blockvec a0tmp = a0_buf_fifo.read();
		w1blockvec d1tmp = delt1_buf_fifo.read();
		for(int i = 0; i < L1/P3; i++) {
			for(int j = 0; j < L2/T3; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=wa1_buf inter false
				for(int ii = 0; ii < P3; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T3; jj++) { //3
						// #pragma HLS UNROLL
						wa1_buf[i][j][ii][jj] = wa1_buf[i][j][ii][jj] + a0tmp.a[i*P3+ii] * d1tmp.a[j*T3+jj];
					}
				}
			}
		}
		for (int i=0; i<L2;i++){
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=gr_bias1 inter false
			gr_bias1.a[i] += d1tmp.a[i];
		}		
	}
}

void wa2(hls::stream<w1blockvec> &a1_buf_fifo, hls::stream<w3blockvec> &delt2_buf_fifo, float wa2_buf[L2/P4][L3/T4][P4][T4], w3blockvec gr_bias2){
	#pragma HLS array_partition variable=wa2_buf complete  dim=3
	#pragma HLS array_partition variable=wa2_buf complete  dim=4

	// #pragma HLS aggregate variable=a1_buf_fifo
	// #pragma HLS aggregate variable=delt2_buf_fifo
	// #pragma HLS aggregate variable=bias2
	#pragma HLS aggregate variable=gr_bias2

	WA2partialsum: for(int k=0; k < BSIZE; k++) {
		w1blockvec a1tmp = a1_buf_fifo.read();
		w3blockvec d2tmp = delt2_buf_fifo.read();
		for(int i = 0; i < L2/P4; i++) {
			for(int j = 0; j < L3/T4; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=wa2_buf inter false
				for(int ii = 0; ii < P4; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T4; jj++) { //3
						// #pragma HLS UNROLL
						// delt1_buf[i][j][ii][jj] = delt1_buf[i][j][ii][jj] + tempA.a[i*Pb+ii] * (*arraywq[j*Tb+jj]).a[k]; //*arraywq: because wcols partitioned in cyclic manner, adjacent indices are in different banks
						// wa2_buf[i][j][ii][jj] = wa2_buf[i][j][ii][jj] + a1_buf_fifo[k].a[i*P4+ii] * delt2_buf_fifo[k].a[j*T4+jj];
						wa2_buf[i][j][ii][jj] = wa2_buf[i][j][ii][jj] + a1tmp.a[i*P4+ii] * d2tmp.a[j*T4+jj];
					}
				}
			}
		}
		for (int i=0; i<L3;i++){
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=gr_bias2 inter false
			gr_bias2.a[i] += d2tmp.a[i];
		}
	}
}






//add learners input interfaces: blockvec *R,actvec *Acts,  blockvec *Snt,actvec *Dn,
//add learners output interfaces(back to cpu): w1blockvec *w1bram,w3blockvec *w2bram
//add replay inputs: int insert_signal,int insert_ind,
//add replay outputs: int ind_o[]
//add qt weight sync signal: if wsync==0: init q & qt params; if wsync==1: let qt params=q params; else: keep updating q param
// void learners_top(blockvec *S, blockvec *Snt, actvec *acts,blockvec *r,float gamma, float alpha, bsbit *done, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],int wsync){
void learners_top(blockvec *S, blockvec *Snt, actvec *acts, blockvec *r, float gamma, float alpha, bsbit *done, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],float *bias1_out,float *bias2_out,int wsync, /*Learners args*/
float *Qs,float *Loss_sqrt,/*Logging args*/
hls::stream<ap_axiu<32,0,0,0>> &pn_out/*Replay args*/){
	#pragma HLS INTERFACE m_axi port=S bundle=gmem1 offset=slave
	#pragma HLS INTERFACE m_axi port=Snt bundle=gmem2 offset=slave
	#pragma HLS INTERFACE m_axi port=acts bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=r bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=done bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=w1bram_out bundle=gmem4 offset=slave
	#pragma HLS INTERFACE m_axi port=w2bram_out bundle=gmem5 offset=slave
	#pragma HLS INTERFACE m_axi port=bias1_out bundle=gmem6 offset=slave
	#pragma HLS INTERFACE m_axi port=bias2_out bundle=gmem7 offset=slave
	#pragma HLS INTERFACE m_axi port=Qs bundle=gmem8 offset=slave
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

	#pragma HLS INTERFACE s_axilite port=wsync bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS aggregate variable=S
	#pragma HLS aggregate variable=Snt
	#pragma HLS aggregate variable=r
	#pragma HLS aggregate variable=acts
	#pragma HLS aggregate variable=done
	#pragma HLS aggregate variable=w1bram_out
	#pragma HLS aggregate variable=w2bram_out
	#pragma HLS aggregate variable=bias1_out
	#pragma HLS aggregate variable=bias2_out
	#pragma HLS aggregate variable=pn_out


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

	static w1blockvec bias1;
	static w3blockvec bias2;
	static w1blockvec bias1_t;
	static w3blockvec bias2_t;
	#pragma HLS aggregate variable=bias1
	#pragma HLS aggregate variable=bias2
	#pragma HLS aggregate variable=bias1_t
	#pragma HLS aggregate variable=bias2_t
	

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
			printf("%d ",done[j].a[i]);  //BS cols
		}
	}
	#endif


//	Init on-chip memory
	// float bias1[L2];
	// float bias2[L3];

	if (wsync==0){ //Init. Q network & target network (only executed exactly once in all iterations!)

	#ifndef __SYNTHESIS__
	printf("\nWeight init.\n");
	#endif
		for (int i=0; i<L1;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				w1bram[i].a[j]=w1list[i][j];
				w1bram_t[i].a[j]=w1list[i][j];
			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
			for  (int j=0; j<L3;j++){
				if (j<2) {w2bram[i].a[j]=w2list_or[i][j];w2bram_t[i].a[j]=w2list_or[i][j];}
				else {w2bram[i].a[j]=w2list_or[i][j-2];w2bram_t[i].a[j]=w2list_or[i][j-2];}
			}
		}
		for (int i=0; i<L2;i++){
			bias1.a[i]=bias1_list[i];
			bias1_t.a[i]=bias1_list[i];
		}
		for (int i=0; i<L3;i++){
			bias2.a[i]=bias2_list[i];
			bias2_t.a[i]=bias2_list[i];
		}

	}

	else if (wsync==1){ //sync target network with Q network
	#ifndef __SYNTHESIS__
	printf("\nTarget Weight sync.\n");
	#endif
		for (int i=0; i<L1;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				w1bram_t[i].a[j]=w1bram[i].a[j];
			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
			for  (int j=0; j<L3;j++){
				w2bram_t[i].a[j]=w2bram[i].a[j];
			}
		}
		for (int i=0; i<L2;i++){
			// #pragma HLS PIPELINE
			bias1_t.a[i]=bias1.a[i];
		}
		for (int i=0; i<L3;i++){
			// #pragma HLS PIPELINE
			bias2_t.a[i]=bias2.a[i];
		}	
	}



	// fw_bw(S,Snt,acts,r,done,w1bram,w2bram,w1bram_t,w2bram_t,bias1, bias2,bias1_t, bias2_t,gamma,  wa1_global,wa2_global);

	//================================================================================================


	hls::stream<blockvec> inpipe("inpipe");
	hls::stream<blockvec> inpipe0("inpipe0");
	hls::stream<blockvec> outpipe0("outpipe0");
	hls::stream<blockvec> outpipe1("outpipe1");
	hls::stream<blockvec> outpipe2("outpipe2");
	hls::stream<blockvec> outpipe3("outpipe3");
	// #pragma HLS array_partition variable=outpipe complete
	#pragma HLS STREAM variable=inpipe depth=8 //L1
	#pragma HLS STREAM variable=inpipe0 depth=8 //L1
	#pragma HLS STREAM variable=outpipe0 depth=64 //L2
	#pragma HLS STREAM variable=outpipe1 depth=4 //L3
	#pragma HLS STREAM variable=outpipe2 depth=64 //L2
	#pragma HLS STREAM variable=outpipe3 depth=4 //L3
 	#pragma HLS bind_storage variable=inpipe type=fifo impl=BRAM 
 	#pragma HLS bind_storage variable=inpipe0 type=fifo impl=BRAM 
 	#pragma HLS bind_storage variable=outpipe0 type=fifo impl=BRAM 
 	#pragma HLS bind_storage variable=outpipe1 type=fifo impl=MEMORY 
 	#pragma HLS bind_storage variable=outpipe2 type=fifo impl=BRAM 
 	#pragma HLS bind_storage variable=outpipe3 type=fifo impl=MEMORY 


	hls::stream<a0blockvec> a0_buf_fifo("a0_buf_fifo");
	#pragma HLS STREAM variable=a0_buf_fifo depth=5 //a0depth
	#pragma HLS bind_storage variable=a0_buf_fifo type=fifo impl=BRAM 
	// bsbit actder[L2];

	hls::stream<bsbit> actder_fifo("actder_fifo");
	#pragma HLS STREAM variable=actder_fifo depth=192 //actderdepth
	#pragma HLS bind_storage variable=actder_fifo type=fifo impl=BRAM 
	// w1blockvec delt1_buf[BSIZE];

	hls::stream<w1blockvec> delt1_buf_fifo("delt1_buf_fifo");
	#pragma HLS STREAM variable=delt1_buf_fifo depth=1 //BSIZE
	#pragma HLS bind_storage variable=delt1_buf_fifo type=fifo impl=MEMORY 


	// w1blockvec a1_buf_fifo[BSIZE];
	hls::stream<w1blockvec> a1_buf_fifo("a1_buf_fifo");
	#pragma HLS STREAM variable=a1_buf_fifo depth=3 //a1depth
	#pragma HLS bind_storage variable=a1_buf_fifo type=fifo impl=MEMORY 


	hls::stream<w3blockvec> delt2_buf_fifo("delt2_buf_fifo"); //delta2 for wu, produced by obj, parallel access on L3 dimension
	#pragma HLS STREAM variable=delt2_buf_fifo depth=1 //BSIZE
	#pragma HLS bind_storage variable=delt2_buf_fifo type=fifo impl=BRAM

	// float delt1_buf[BSIZE][L2]; 
	// w1blockvec delt1_buf[BSIZE]={0}; //delta1 for wu, produced by sub_backmm2, parallel access on L2 dimension
	float wa1_buf[L1/P3][L2/T3][P3][T3]={0};
	float wa2_buf[L2/P4][L3/T4][P4][T4]={0};
	w1blockvec gr_bias1={0};
	w3blockvec gr_bias2={0};

	// #pragma HLS array_partition variable=w2bram type=cyclic  factor=8 

	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4
	#pragma HLS array_partition variable=wa2_buf complete  dim=3
	#pragma HLS array_partition variable=wa2_buf complete  dim=4


	w1blockvec w2bram_copy[L3]; //w2 for BW, aggregate dim L2
	#pragma HLS aggregate variable=w2bram_copy
	for (int i=0; i<L3; i++){
		for (int j = 0; j < L2; j++){
			#pragma HLS PIPELINE
			w2bram_copy[i].a[j]=w2bram[j].a[i];
		}
	}


	hls::stream<blockvec> outpipe6("outpipe6");
	#pragma HLS STREAM variable=outpipe6 depth=4 //L3

	float Qs_local[BATCHS*BSIZE];
	float Loss_sqrt_local[BATCHS*BSIZE];
	for(int ind=0; ind<BATCHS; ind++){
		#pragma HLS DATAFLOW

		loadIn(S, a0_buf_fifo, inpipe, L1, ind);

		loadSn(Snt, inpipe0, L1, ind);

		fw_l1(inpipe,a1_buf_fifo,bias1, w1bram, outpipe0, actder_fifo,L1,L2);

		fw_l2(outpipe0, bias2,w2bram, outpipe1,L2,L3);

		fw_l1_targ(inpipe0,bias1_t, w1bram_t, outpipe2,L1,L2);
		fw_l2(outpipe2, bias2_t,w2bram_t, outpipe3,L2,L3);

		// test_target(outpipe2,outpipe3);
		objctv(r, acts, gamma, done, outpipe1,outpipe3, outpipe6, delt2_buf_fifo, ind,pn_out, Qs_local, Loss_sqrt_local);
		
		// test_objctv(outpipe1,outpipe3,outpipe6, delt2_buf_fifo);
		sub_backmm2(outpipe6, w2bram_copy, actder_fifo, delt1_buf_fifo, L3,L2);
		// test_sub_backmm2(outpipe6, actder_fifo, delt1_buf_fifo);
		
		wa1(a0_buf_fifo, delt1_buf_fifo, wa1_buf, gr_bias1);
		wa2(a1_buf_fifo, delt2_buf_fifo, wa2_buf, gr_bias2);
	}
	//================================================================================================

	float alpha_local=alpha;
	#pragma HLS array_partition variable=w1bram type=cyclic  factor=2
	#pragma HLS array_partition variable=w2bram type=cyclic  factor=8
	// WU: Substract -SGD (Add if SGA) WA from wbrams
	for(int i = 0; i < L1/P3; i++) {
		for(int j = 0; j < L2/T3; j++) {
			for(int ii = 0; ii < P3; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w1bram inter false
				for(int jj = 0; jj < T3; jj++) { 
					w1bram[i*P3+ii].a[j*T3+jj] -= alpha_local * wa1_buf[i][j][ii][jj];
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
			for(int ii = 0; ii < P4; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w2bram inter false
				for(int jj = 0; jj < T4; jj++) { 
					w2bram[i*P4+ii].a[j*T4+jj] -= alpha_local * wa2_buf[i][j][ii][jj];
					// w2bram_out[i*P4+ii].a[j*T4+jj]=w2bram[i*P4+ii].a[j*T4+jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nw2bram updated.\n");
	#endif

	for (int i=0; i<L2;i++){
		bias1.a[i] -= alpha_local * gr_bias1.a[i];
		// bias1_out[i]=bias1.a[i];
	}
	for (int i=0; i<L3;i++){
		bias2.a[i] -= alpha_local * gr_bias2.a[i];
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
		#pragma HLS PIPELINE
		bias1_out[i]=bias1.a[i];
	}
	for (int i=0; i<L3;i++){
		#pragma HLS PIPELINE
		bias2_out[i]=bias2.a[i];
	}
	// }	
	for (int i=0; i<BATCHS*BSIZE;i++){
		#pragma HLS PIPELINE
		Qs[i]=Qs_local[i];
		Loss_sqrt[i]=Loss_sqrt_local[i];
	}
	// }	
	#ifndef __SYNTHESIS__
	printf("\nAll Transfer Finished.\n");
	#endif

}

}