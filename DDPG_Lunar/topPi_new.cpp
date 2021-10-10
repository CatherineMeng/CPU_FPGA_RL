// #include "./topPi_new.h"
#include "./topQ_new.h"
// #include "./common.h"

extern "C"{



// ind: increment from 0 to BATCHS/BSIZE
// loadIn(S,  Acts, LSApipe, Snt, LSNpipe0,  Qa0_buf_fifo, ind)

void PiloadIn(float S_In[],  hls::stream<float> &LSpipe0, hls::stream<float> &LSpipe1, hls::stream<Pia0blockvec> &Pia0_buf_fifo,int ind){
	// float Inrows_local[LL];
	for (int i = 0; i < L1_pi; i++){ //s space
		#pragma HLS PIPELINE
		LSpipe0.write(S_In[ind*L1_pi+i]);
		LSpipe1.write(S_In[ind*L1_pi+i]);
	}
	Pia0blockvec a0tmp;
	#ifndef __SYNTHESIS__
	printf("\n\nPiloadIn: Input SPiin read:\n");
	#endif
	for (int i = 0; i < L1_pi; i++){ //s space
		#pragma HLS PIPELINE
		a0tmp.a[i]=S_In[ind*L1_pi+i];
		#ifndef __SYNTHESIS__
		{printf("%.8f ",S_In[ind*L1_pi+i]);}
		#endif
	}
	// get a0_buf_fifo for WA
	Pia0_buf_fifo.write(a0tmp);
	
}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// void fw_l1(hls::stream<blockvec> &Inrows, float C[BSIZE/P][64/T][P][T],w1blockvec Wcols[], hls::stream<blockvec> &Crows, float Qa1_buf_fifo[L2][BSIZE], const int LL,const int LN) {
// void fw_l1(hls::stream<blockvec> &Inrows, float z1_buf[BSIZE/P][L2/T][P][T], float Qa1_buf_fifo[L2][BSIZE],float bias[], w1blockvec Wcols[], hls::stream<blockvec> &Crows, int actder[L2],const int LL,const int LN) {
void topPifw_l1(hls::stream<float> &Inrows, hls::stream<w1blockvec> &Qa1_buf_fifo, w1blockvec bias, w1blockvec Wcols[], hls::stream<float> &Crows, hls::stream<int> &Qactder_fifo,const int LL,const int LN) {


	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=Qactder_fifo
	#pragma HLS aggregate variable=bias

    #pragma HLS dependence class=array variable=z1_buf_local type=inter dependent=false
    #pragma HLS dependence class=array variable=z1_buf_local type=intra dependent=false
	float z1_buf_local[L2/T][T];

	#pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=2 complete
	partialsum: for(int k=0; k < LL; k++) {
		float tempA = Inrows.read();
		w1blockvec tempB = Wcols[k];
    // #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		// for(int i = 0; i < BSIZE/P; i++) {
			for(int j = 0; j < LN/T; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=z1_buf_local inter false
				// for(int ii = 0; ii < P; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T; jj++) {
						#pragma HLS UNROLL
						//#pragma HLS dependence variable=C inter false
						if (k==0) z1_buf_local[j][jj]=tempA * tempB.a[j*T+jj];
						else if (k==LL-1) z1_buf_local[j][jj] += (tempA * tempB.a[j*T+jj] + bias.a[j*T+jj]);
						else z1_buf_local[j][jj] += tempA * tempB.a[j*T+jj];
					}
				// }
			}
		// }
	}
	
	//write out to stream: next fw

	#ifndef __SYNTHESIS__
	printf("\n topPifw_l1: z1_buf content:\n");//should be L2 rows, BSIZE columns
	#endif
	// get Qa1_buf_fifo for WA, find actder, write activatpon out to stream
	writeout: for(int j = 0; j < LN/T; j++) { //this factor consistent with Qa1_buf_fifo partition
		for(int jj = 0; jj < T; jj++) {
			float tempC;
			int actdertmp;

			#pragma HLS PIPELINE

			float tmpz=(z1_buf_local[j][jj]>0)? z1_buf_local[j][jj]:0;
			// Qa1_buf_fifo[i*P+ii].a[j*T+jj]=tmpz;
			actdertmp=(z1_buf_local[j][jj]>0)? 1:0; //activation derivative

			tempC=tmpz; //activation
			#ifndef __SYNTHESIS__
			printf("%.8f ",z1_buf_local[j][jj]);
			#endif

			Qactder_fifo.write(actdertmp);
			
		}
	}
	//write out to stream: a1_buf

		w1blockvec a1buftmp;
		for(int j = 0; j < LN/T; j++) { //this factor consistent with Qa1_buf_fifo partition
			#pragma HLS PIPELINE
			for(int jj = 0; jj < T; jj++) {
				 //activation
				// float tmpz=(z1_buf_local[i][j][ii][jj]>0)? z1_buf_local[i][j][ii][jj]:0;
				a1buftmp.a[j*T+jj]=(z1_buf_local[j][jj]>0)? z1_buf_local[j][jj]:0;
				Crows.write(a1buftmp.a[j*T+jj]);
			}
		}

		Qa1_buf_fifo.write(a1buftmp);


}


void topPifw_l2(hls::stream<float> &Inrows, Piw2blockvec bias,Piw2blockvec Wcols[], hls::stream<float> &Crows,  hls::stream<float> &pitanh_actder, const int LL,const int LN) {
	// #pragma HLS INLINE
	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=bias
	// float C[BSIZE/P2][3/T2][P2][T2]={0};
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=4 complete

	float z2_buf_local[L3_pi/T2][T2];
	// #pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=2 complete
	#pragma HLS bind_storage variable=z2_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < LL; k++) {
	float tempA = Inrows.read();
	Piw2blockvec tempB = Wcols[k];
    // #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		partialsum_l2: for(int j = 0; j < LN/T2; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=z2_buf_local type=inter dependent=false
			for(int jj = 0; jj < T2; jj++) { //3
				#pragma HLS UNROLL
				// z2_buf_local[i][j][ii][jj] = z2_buf_local[i][j][ii][jj] + tempA.a[i*P2+ii] * tempB.a[j*T2+jj];
				if (k==0) z2_buf_local[j][jj]=tempA * tempB.a[j*T2+jj];
				else if (k==LL-1) z2_buf_local[j][jj] += (tempA * tempB.a[j*T2+jj] + bias.a[j*T2+jj]);
				else z2_buf_local[j][jj] += tempA * tempB.a[j*T2+jj];
			}
		}
	}


	#ifndef __SYNTHESIS__
	printf("\ntopPifw_l2: z2_buf content:\n");
	for(int j = 0; j < LN/T2; j++) { //this factor consistent with Qa1_buf_fifo partition
		for(int jj = 0; jj < T2; jj++) {
			printf("%.8f ",z2_buf_local[j][jj]);
		}
	}
	#endif
	// write out to stream
	for(int j = 0; j < LN/T2; j++) {
		for(int jj = 0; jj < T2; jj++) {
			#pragma HLS PIPELINE
			float tempC;
			if (z2_buf_local[j][jj]>1)tempC=1;
			else if (z2_buf_local[j][jj]<-1)tempC=-1;
			else tempC= z2_buf_local[j][jj]; //=======approx tanh==========
			Crows.write(tempC);
			pitanh_actder.write(tempC);
		}
	}

}


void topPiconcatpipe(hls::stream<float> &LSpipe1, hls::stream<float> &Pil2_pipe, hls::stream<float> &LSApipe){ //assumes s first, a next
#ifndef __SYNTHESIS__
	printf("\nQ_Input after concatenation:\n");
#endif
	for (int i = 0; i < L1_pi; i++){
		#pragma HLS PIPELINE
		float tmp=LSpipe1.read();
		LSApipe.write(tmp); //fisrt s
		#ifndef __SYNTHESIS__
		printf("%f ", tmp);
		#endif

	}
	for (int i = 0; i < L3_pi; i++){
		#pragma HLS PIPELINE
		float tmp=Pil2_pipe.read();
		LSApipe.write(tmp); //then a
		#ifndef __SYNTHESIS__
		printf("%f ", tmp);
		#endif	
	}
}

void fw_l1_Q(hls::stream<float> &Inrows,  w1blockvec bias, w1blockvec Wcols[], hls::stream<float> &Crows, hls::stream<int> &Ql1actder_fifo,const int LL,const int LN) {

	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=bias

	float z1_buf_local[L2/T][T]={0};
	#pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=2 complete
	partialsum: for(int k=0; k < LL; k++) {
	float tempA = Inrows.read();
	w1blockvec tempB = Wcols[k];
     #pragma HLS aggregate variable=tempB
		for(int j = 0; j < LN/T; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=z1_buf_local inter false
			for(int jj = 0; jj < T; jj++) {
				#pragma HLS UNROLL
				//#pragma HLS dependence variable=C inter false
				if (k==0) z1_buf_local[j][jj]=tempA * tempB.a[j*T+jj];
				else if (k==LL-1) z1_buf_local[j][jj] += (tempA * tempB.a[j*T+jj] + bias.a[j*T+jj]);
				else z1_buf_local[j][jj] += tempA * tempB.a[j*T+jj];
			}
		}	
	}
	
	//write out to stream: next fw

	#ifndef __SYNTHESIS__
	printf("\n Pi fw_l1_Q: z1_buf content:\n");//should be L2 rows, BSIZE columns
	#endif
	// get Qa1_buf_fifo for WA, find actder, write activatpon out to stream
	writeout: for(int j = 0; j < LN/T; j++) { //this factor consistent with Qa1_buf_fifo partition
		for(int jj = 0; jj < T; jj++) {
			float tempC;
			int actdertmp;
			#pragma HLS aggregate variable=tempC
			#pragma HLS PIPELINE
			tempC=(z1_buf_local[j][jj]>0)? z1_buf_local[j][jj]:0; //activation
			actdertmp=(z1_buf_local[j][jj]>0)? 1:0; //activation derivative

			#ifndef __SYNTHESIS__
			printf("%.8f ",z1_buf_local[j][jj]);
			#endif

			Crows.write(tempC);
			Ql1actder_fifo.write(actdertmp);
		}
	}
}




//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// topPifw_l2_q(Ql1_pipe, bias2_Q,w2bram_Q, Ql2_pipe,L2);//l2 inf for Q
void topPifw_l2_q(hls::stream<float> &Inrows, float bias,float Wcols[], hls::stream<float> &Crows,const int LL) {
	float z2_buf_local;

	partialsum1D: for(int k=0; k < L2; k++) {
		float tempA = Inrows.read();
		float tempB = Wcols[k];
		#pragma HLS PIPELINE
		// #pragma HLS dependence variable=z2_buf_local type=inter dependent=false
		if (k==0) z2_buf_local = tempA * tempB;
		else if (k==L2-1) z2_buf_local += (tempA * tempB + bias);
		else z2_buf_local += tempA * tempB;
	}

	#ifndef __SYNTHESIS__
	printf("\ntopPi topPifw_l2_q: z2_buf content:\n");
		printf("%.8f ",z2_buf_local);
	#endif

	Crows.write(z2_buf_local);
}






// void test_sub_backmm2(hls::stream<blockvec> &Inrows, hls::stream<int> &Qactder_fifo, hls::stream<w1blockvec> &delt1_buf_fifo){
// 	multactder:for(int j = 0; j < L2; j++) { 
// 		int actdertmp = Qactder_fifo.read();
// 	}
// 	partialsum: for(int k=0; k < L3; k++) { //LL is L3
// 		blockvec tempA = Inrows.read();
// 	}
// 	// for(int i = 0; i < BSIZE; i++) {
// 		w1blockvec d1tmp;
// 		#pragma HLS aggregate variable=d1tmp
// 		writeout:for(int j = 0; j < L2; j++) { 
// 			#pragma HLS PIPELINE
// 				// delt1_buf_fifo[i*P+ii].a[j*T+jj] = delt1_buf_local[i][j][ii][jj];
// 				d1tmp.a[j] =1;
// 		}
// 		delt1_buf_fifo.write(d1tmp);
// 	// }
// }


//Inrows: LN blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LL blockvecs (each batchsize)
// void sub_backmm2(hls::stream<blockvec> &Inrows, 
// 	Piw2blockvec Wcols0, Piw2blockvec Wcols1, Piw2blockvec Wcols2, Piw2blockvec Wcols3,
// 	Piw2blockvec Wcols4,Piw2blockvec Wcols5,Piw2blockvec Wcols6,Piw2blockvec Wcols7, hls::stream<blockvec> &Crows, 
// 	float delt1_buf[BSIZE/Pb][L3/Tb][Pb][Tb], const int LL,const int LN,int ind) {
// sub_backmm2(loss_pipe, w2bram_Q_copy, Ql1actder_fifo, delt1_buf_fifo, L2);
void sub_backmm2Q(hls::stream<float> &loss_pipe, w1blockvec Wcols, hls::stream<int> &Ql1actder_fifo, hls::stream<float> &Ql2back_pipe, const int LN){
	// #pragma HLS aggregate variable=loss_pipe
	// #pragma HLS aggregate variable=Wcols1s
	#pragma HLS aggregate variable=Wcols
	// #pragma HLS aggregate variable=Ql2back_pipe

	float delt1_buf_local[L2/T][T]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=2 complete
	// #pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=4 complete
	
	#pragma HLS bind_storage variable=delt1_buf_local type=RAM_2P impl=bram

	#ifndef __SYNTHESIS__
	printf("\ntopPi sub_backmm2Q: delt l1 Q content before actder:\n");
	#endif

	float tempA = loss_pipe.read();
	w1blockvec tempB = Wcols; //tempB size L2

     #pragma HLS aggregate variable=tempB
	for(int j = 0; j < L2/T; j++) { //LN is L2
	#pragma HLS PIPELINE
	#pragma HLS dependence variable=delt1_buf_local inter false
		for(int jj = 0; jj < T; jj++) { 
			int actdertmp = Ql1actder_fifo.read();
			delt1_buf_local[j][jj] = delt1_buf_local[j][jj] + tempA * tempB.a[j*T+jj];
			#ifndef __SYNTHESIS__
				printf("%.8f ",delt1_buf_local[j][jj]);
			#endif
			// times actder
			delt1_buf_local[j][jj] = (actdertmp!=0)? delt1_buf_local[j][jj]:0;
		}

	}

	writeout:for(int j = 0; j < L2/T; j++) { 
		#pragma HLS PIPELINE
		for(int jj = 0; jj < T; jj++) {
			Ql2back_pipe.write(delt1_buf_local[j][jj]);
		}
	}
}

void sub_backmm1Q(hls::stream<float> &in_pipe, Qa0blockvec Wcols[], hls::stream<float> &out_pipe, const int LL, const int LN){
	#pragma HLS aggregate variable=Wcols

	float delt1_buf_local[L1_q/T2][T2]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=2 complete
	#ifndef __SYNTHESIS__
	printf("\ntopPi sub_backmm1Q: delt l0 Q content (no actder):\n");
	#endif
	
	#pragma HLS bind_storage variable=delt1_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < L2; k++) { //LL is L3
		float tempA = in_pipe.read();
		Qa0blockvec tempB = Wcols[k]; //tempB size L2
    #pragma HLS aggregate variable=tempA
     // #pragma HLS aggregate variable=tempB
		for(int j = 0; j < L1_q/T2; j++) { //LN is L2
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=delt1_buf_local inter false
			// #pragma HLS UNROLL
			for(int jj = 0; jj < T2; jj++) { 
				delt1_buf_local[j][jj] = delt1_buf_local[j][jj] + tempA * tempB.a[j*T2+jj];
			}
			
		}
		

	}

	writeout:for(int j = 0; j < L1_q/T2; j++) { 
		#pragma HLS PIPELINE
		for(int jj = 0; jj < T2; jj++) {
				#ifndef __SYNTHESIS__
					printf("%.8f ",delt1_buf_local[j][jj]);
				#endif
			out_pipe.write(delt1_buf_local[j][jj]);
		}
	}
}

void sub_backmm2pi(hls::stream<float> &Inrows, w1blockvec Wcols[], hls::stream<int> &actder_fifo, hls::stream<w1blockvec> &delt1_buf_fifo, hls::stream<Piw2blockvec> &delt2_buf_fifo, hls::stream<float> &pitanh_actder, const int LL,const int LN){

	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=delt1_buf_fifo

	#ifndef __SYNTHESIS__
	printf("\ntopPi sub_backmm2pi: Pi backward inputs after actder:\n");
	#endif
	float delt1_buf_local[L2/T][T]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=2 complete
	// #pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=4 complete
	#pragma HLS bind_storage variable=delt1_buf_local type=RAM_2P impl=bram

	drainoutsspace: for(int k=0; k < L1_pi; k++) { //first s, then a. To get a, first drain out s
		Inrows.read();
	}
	Piw2blockvec d2tmp;
	// for(int k=0; k < L1_pi; k++)float tempA = Inrows.read(); //drop s-dim state space outputs

	partialsum: for(int k=0; k < L3_pi; k++) { //LL is L3
		float tempA = Inrows.read();
		float actder=pitanh_actder.read();

		float tempAA=tempA*actder;
				#ifndef __SYNTHESIS__
					printf("%.8f ",tempAA);
				#endif
		w1blockvec tempB = Wcols[k]; //tempB size L2
		d2tmp.a[k]=tempAA;
     // #pragma HLS aggregate variable=tempB
		for(int j = 0; j < L2/T; j++) { //LN is L2
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=delt1_buf_local inter false
			// #pragma HLS UNROLL
			for(int jj = 0; jj < T; jj++) { 
				// #pragma HLS UNROLL
				// delt1_buf[i][j][ii][jj] = delt1_buf[i][j][ii][jj] + tempA.a[i*Pb+ii] * (*arraywq[j*Tb+jj]).a[k]; //*arraywq: because wcols partitioned in cyclic manner, adjacent indices are in different banks
				// delt1_buf_local[i][j][ii][jj] = delt1_buf_local[i][j][ii][jj] + tempA.a[i*P+ii] * (Wcols[j*T+jj]).a[k];
				delt1_buf_local[j][jj] = delt1_buf_local[j][jj] + tempAA * tempB.a[j*T+jj];
			}	
		}
	}
	delt2_buf_fifo.write(d2tmp);
	#ifndef __SYNTHESIS__
	printf("\ntopPi sub_backmm2pi: Delt1 pi before actder:\n");
	#endif
	w1blockvec d1tmp;
	#pragma HLS aggregate variable=d1tmp
	multactder:for(int j = 0; j < L2/T; j++) { 
		for(int jj = 0; jj < T; jj++) {
			float actdertmp = actder_fifo.read();
		
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=delt1_buf_local inter false
			

				#ifndef __SYNTHESIS__
					printf("%.8f ",delt1_buf_local[j][jj]);
				#endif
			// float tmpdelt=delt1_buf_local[j][jj];
			// delt1_buf_local[i][j][ii][jj] = (actder_fifo[j*T+jj].a[i*P+ii]!=0)? tmpdelt:0;
			// d1tmp.a[j*T+jj] = (actdertmp!=0)? tmpdelt:0;
					// delt times z1 relu derivative
			d1tmp.a[j*T+jj] = (actdertmp!=0)? delt1_buf_local[j][jj]:0;

			// d1tmp.a[j*T+jj] = delt1_buf_local[j][jj];
		}
	}
	delt1_buf_fifo.write(d1tmp);

}

void wa1_pi(hls::stream<Pia0blockvec> &a0_buf_fifo, hls::stream<w1blockvec> &delt1_buf_fifo, float wa1_buf[L1_pi/P3][L2/T3][P3][T3], float gr_bias1[L2]){
	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4

	// #pragma HLS aggregate variable=a0_buf_fifo
	// #pragma HLS aggregate variable=delt1_buf_fifo
	// #pragma HLS aggregate variable=bias1
	#pragma HLS aggregate variable=gr_bias1

	 // for(int k=0; k < BSIZE; k++) {
		Pia0blockvec a0tmp = a0_buf_fifo.read();
		w1blockvec d1tmp = delt1_buf_fifo.read();
		WA1partialsum:for(int i = 0; i < L1_pi/P3; i++) {
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
			gr_bias1[i] += d1tmp.a[i];
		}		
	// }
}

void wa2_pi(hls::stream<w1blockvec> &Pia1_buf_fifo, hls::stream<Piw2blockvec> &delt2_buf_fifo, float wa2_buf[L2/P4][L3_pi/T4][P4_pi][T4_pi], float gr_bias2[L3_pi]){
	#pragma HLS array_partition variable=wa2_buf complete  dim=3
	#pragma HLS array_partition variable=wa2_buf complete  dim=4

	// #pragma HLS aggregate variable=Qa1_buf_fifo
	// #pragma HLS aggregate variable=delt2_buf_fifo
	// #pragma HLS aggregate variable=bias2
	#pragma HLS aggregate variable=gr_bias2

	// for(int k=0; k < BSIZE; k++) {
		w1blockvec a1tmp = Pia1_buf_fifo.read();
		Piw2blockvec d2tmp = delt2_buf_fifo.read();
		WA2partialsum: for(int i = 0; i < L2/P4; i++) {
			for(int j = 0; j < L3_pi/T4; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=wa2_buf inter false
				for(int ii = 0; ii < P4; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T4; jj++) { //3
						// #pragma HLS UNROLL
						// delt1_buf[i][j][ii][jj] = delt1_buf[i][j][ii][jj] + tempA.a[i*Pb+ii] * (*arraywq[j*Tb+jj]).a[k]; //*arraywq: because wcols partitioned in cyclic manner, adjacent indices are in different banks
						// wa2_buf[i][j][ii][jj] = wa2_buf[i][j][ii][jj] + Qa1_buf_fifo[k].a[i*P4+ii] * delt2_buf_fifo[k].a[j*T4+jj];
						wa2_buf[i][j][ii][jj] = wa2_buf[i][j][ii][jj] + a1tmp.a[i*P4+ii] * d2tmp.a[j*T4+jj];
					}
				}
			}
		}
		for (int i=0; i<L3_pi;i++){
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=gr_bias2 inter false
			gr_bias2[i] += d2tmp.a[i];
		}
	// }
}






//add learners input interfaces: blockvec *R,actvec *Acts,  blockvec *Snt,actvec *Dn,
//add learners output interfaces(back to cpu): w1blockvec *w1bram_policy,Piw2blockvec *w2bram_policy
//add replay inputs: int insert_signal,int insert_ind,
//add replay outputs: int ind_o[]
//add qt weight sync signal: if wsync==0: init q & qt params; if wsync==1: let qt params=q params; else: keep updating q param
// void learners_top(blockvec *S, blockvec *Snt, actvec *acts,blockvec *r,float gamma, float alpha, int *done, w1blockvec policyw1_out[L1],Piw2blockvec policyw2_out[L2],int wsync){
void learnersPi_top(float *S, float alpha, int Piwsync,
hls::stream<ap_axiu<32,0,0,0>> &Qw1_axiin, hls::stream<ap_axiu<32,0,0,0>> &Qw2_axiin, hls::stream<ap_axiu<32,0,0,0>> &Qbias1_axiin, hls::stream<ap_axiu<32,0,0,0>> &Qbias2_axiin,
hls::stream<ap_axiu<32,0,0,0>> &Pitw1_axiout, hls::stream<ap_axiu<32,0,0,0>> &Pitw2_axiout, hls::stream<ap_axiu<32,0,0,0>> &Pitbias1_axiout, hls::stream<ap_axiu<32,0,0,0>> &Pitbias2_axiout,
w1blockvec policyw1_out[L1_pi], Piw2blockvec policyw2_out[L2], float *bias1_out, float *bias2_out,
int BATCHS
){
	#pragma HLS INTERFACE m_axi port=S bundle=gmem1 offset=slave
	#pragma HLS INTERFACE m_axi port=policyw1_out bundle=gmem2 offset=slave
	#pragma HLS INTERFACE m_axi port=policyw2_out bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=bias1_out bundle=gmem4 offset=slave
	#pragma HLS INTERFACE m_axi port=bias2_out bundle=gmem5 offset=slave

	#pragma HLS INTERFACE s_axilite port=S
	#pragma HLS INTERFACE s_axilite port=policyw1_out
	#pragma HLS INTERFACE s_axilite port=policyw2_out
	#pragma HLS INTERFACE s_axilite port=bias1_out
	#pragma HLS INTERFACE s_axilite port=bias2_out


	#pragma HLS INTERFACE axis port=Qw1_axiin
	#pragma HLS INTERFACE axis port=Qw2_axiin
	#pragma HLS INTERFACE axis port=Qbias1_axiin
	#pragma HLS INTERFACE axis port=Qbias2_axiin
	#pragma HLS INTERFACE axis port=Pitw1_axiout
	#pragma HLS INTERFACE axis port=Pitw2_axiout
	#pragma HLS INTERFACE axis port=Pitbias1_axiout
	#pragma HLS INTERFACE axis port=Pitbias2_axiout

	#pragma HLS INTERFACE s_axilite port=alpha
	#pragma HLS INTERFACE s_axilite port=Piwsync
	#pragma HLS INTERFACE s_axilite port=BATCHS
	#pragma HLS INTERFACE s_axilite port=return


	#pragma HLS aggregate variable=S
	// #pragma HLS aggregate variable=Qw1_axiin
	// #pragma HLS aggregate variable=Qw2_axiin
	// #pragma HLS aggregate variable=Pitw1_axiout
	// #pragma HLS aggregate variable=Pitw2_axiout
	#pragma HLS aggregate variable=policyw1_out
	#pragma HLS aggregate variable=policyw2_out
	#pragma HLS aggregate variable=bias1_out
	#pragma HLS aggregate variable=bias2_out

	static w1blockvec w1bram_Q[L1_q]; //Qw1
	#pragma HLS aggregate variable=w1bram_Q
	#pragma HLS bind_storage variable=w1bram_Q type=RAM_2P impl=bram
	static float w2bram_Q[L2]; //Qw2
	#pragma HLS bind_storage variable=w2bram_Q type=RAM_2P impl=bram
	static w1blockvec bias1_Q;
	#pragma HLS aggregate variable=bias1_Q
	static float bias2_Q;	


	static w1blockvec w1bram_pi[L1_pi]; //Qw1 target
	#pragma HLS aggregate variable=w1bram_pi
	#pragma HLS bind_storage variable=w1bram_pi type=RAM_2P impl=bram
	static Piw2blockvec w2bram_pi[L2]; //Qw2 target
	#pragma HLS bind_storage variable=w2bram_pi type=RAM_2P impl=bram
	static w1blockvec bias1_pi;
	#pragma HLS aggregate variable=bias1_pi
	static Piw2blockvec bias2_pi;
	#pragma HLS aggregate variable=bias2_pi


	// Inference chains to produce Q objctv:
	hls::stream<float> LSApipe("LSApipe");
	hls::stream<float> LSpipe0("LSpipe0");
	hls::stream<float> LSpipe1("LSpipe1");
	hls::stream<float> Pil1_pipe("Pil1_pipe");
	hls::stream<float> Pil2_pipe("Pil2_pipe");
	hls::stream<float> Ql1_pipe("Ql1_pipe");
	hls::stream<float> Ql2_pipe("Ql2_pipe");
	hls::stream<float> Ql1back_pipe("Ql1back_pipe");
	hls::stream<float> Ql2back_pipe("Ql2back_pipe");
	#pragma HLS STREAM variable=LSApipe depth=12 //L1
	#pragma HLS STREAM variable=LSpipe0 depth=8 //L1
	#pragma HLS STREAM variable=LSpipe1 depth=24 //L1*3 stages
	#pragma HLS STREAM variable=Pil1_pipe depth=64 //L2
	#pragma HLS STREAM variable=Pil2_pipe depth=4 //L3
	#pragma HLS STREAM variable=Ql1_pipe depth=64 //L2
	#pragma HLS STREAM variable=Ql2_pipe depth=4 //L3
	#pragma HLS STREAM variable=Ql1back_pipe depth=64 //L2
	#pragma HLS STREAM variable=Ql2back_pipe depth=4 //L3
 	#pragma HLS bind_storage variable=LSApipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=LSpipe0 type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=LSpipe1 type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Pil1_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Pil2_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Ql1_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Ql2_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Ql1back_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Ql2back_pipe type=fifo impl=SRL 


	// Inference chains to produce Q objctv:
	hls::stream<Pia0blockvec> Pia0_buf_fifo("Qa0_buf_fifo");
	#pragma HLS STREAM variable=Pia0_buf_fifo depth=8 //??????????????????????
	#pragma HLS bind_storage variable=Pia0_buf_fifo type=fifo impl=SRL 
	// int actder[L2];

	hls::stream<int> Ql1actder_fifo("Ql1actder_fifo");
	#pragma HLS STREAM variable=Ql1actder_fifo depth=192 //actderdepth??????????????????????
	#pragma HLS bind_storage variable=Ql1actder_fifo type=fifo impl=SRL 
	// // w1blockvec delt1_buf[BSIZE];

	hls::stream<int> Pil1actder_fifo("Pil1actder_fifo");
	#pragma HLS STREAM variable=Pil1actder_fifo depth=192 //actderdepth??????????????????????
	#pragma HLS bind_storage variable=Pil1actder_fifo type=fifo impl=SRL 

	hls::stream<w1blockvec> delt1_buf_fifo("delt1_buf_fifo");
	#pragma HLS STREAM variable=delt1_buf_fifo depth=1 //BSIZE??????????????????????
	#pragma HLS bind_storage variable=delt1_buf_fifo type=fifo impl=SRL 


	// w1blockvec Qa1_buf_fifo[BSIZE];
	hls::stream<w1blockvec> Pia1_buf_fifo("Qa1_buf_fifo");
	#pragma HLS STREAM variable=Pia1_buf_fifo depth=3 //a1depth??????????????????????
	#pragma HLS bind_storage variable=Pia1_buf_fifo type=fifo impl=SRL 


	hls::stream<Piw2blockvec> delt2_buf_fifo("delt2_buf_fifo"); //delta2 for wu, produced by obj, parallel access on L3 dimension
	#pragma HLS STREAM variable=delt2_buf_fifo depth=1 //BSIZE??????????????????????
	#pragma HLS bind_storage variable=delt2_buf_fifo type=fifo impl=SRL

	hls::stream<float> loss_pipe("loss_pipe"); //this is for Q
	#pragma HLS STREAM variable=loss_pipe depth=4 //L3
	#pragma HLS bind_storage variable=loss_pipe type=fifo impl=SRL

	hls::stream<float> pitanh_actder("loss_pipe"); //
	#pragma HLS STREAM variable=pitanh_actder depth=8 //L3
	#pragma HLS bind_storage variable=pitanh_actder type=fifo impl=SRL

	// float delt1_buf[BSIZE][L2]; 
	// w1blockvec delt1_buf[BSIZE]={0}; //delta1 for wu, produced by sub_backmm2, parallel access on L2 dimension
	float wa1_buf[L1_pi/P3][L2/T3][P3][T3]={0};
	float wa2_buf[L2/P4][L3_pi/T4][P4_pi][T4_pi]={0};
	// w1blockvec gr_bias1={0};
	// Piw2blockvec gr_bias2={0};
	float gr_bias1[L2] ={0};
	float gr_bias2[L3_pi] ={0};
	// #pragma HLS array_partition variable=w2bram_policy type=cyclic  factor=8 

	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4
	#pragma HLS array_partition variable=wa2_buf complete  dim=3
	#pragma HLS array_partition variable=wa2_buf complete  dim=4

	#ifndef __SYNTHESIS__
	printf("\nPi_top: Q init done. Q content:\n");
	#endif
		// ==============================================DO THIS==============================================
	// if Piwsync==1: send pitarg to SLR0 (write)
	// must: read Q from SLR0 (read)
	for (int i=0; i<L1_q;i++){
		for  (int j=0; j<L2;j++){
			ap_axiu<32,0,0,0> qwtmp=Qw1_axiin.read();
			w1bram_Q[i].a[j]=qwtmp.data;
			#ifndef __SYNTHESIS__
			printf("%f ", float(qwtmp.data)); //w1bram_Q[i].a[j]
			#endif
		}
		#ifndef __SYNTHESIS__
		printf("\n");
		#endif
	}
	#ifndef __SYNTHESIS__
	printf("\nPi_top: Q sync l1\n");
	#endif
	for (int i=0; i<L2;i++){
		ap_axiu<32,0,0,0> qwtmp=Qw2_axiin.read();
		w2bram_Q[i]=qwtmp.data;	
	}
	#ifndef __SYNTHESIS__
	printf("\nPi_top: Q sync l2\n");
	#endif
	for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
		ap_axiu<32,0,0,0> qwtmp=Qbias1_axiin.read();
		bias1_Q.a[i]=qwtmp.data;
	}
	#ifndef __SYNTHESIS__
	printf("\nPi_top: Q sync l3\n");
	#endif
	ap_axiu<32,0,0,0> qwtmp=Qbias2_axiin.read();
	bias2_Q=qwtmp.data;

	#ifndef __SYNTHESIS__
	printf("\nPi_top: Q Sync done.\n");
	#endif


	if (Piwsync==0){ //Init. Q network & target network (only executed exactly once in all iterations!)
	#ifndef __SYNTHESIS__
	printf("\nWeight init.\n");
	#endif
		for (int i=0; i<L1_pi;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				w1bram_pi[i].a[j]=w1list[i][j];
			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
			for  (int j=0; j<L3_pi;j++){
				w2bram_pi[i].a[j]=w2list_or[i][j%2];
				// else w2bram_pi[i].a[j]=w2list_or[i][j-2];
			}
		}
		for (int i=0; i<L2;i++){
			bias1_pi.a[i]=bias1_list[i];
		}
		for (int i=0; i<L3_pi;i++){
			bias2_pi.a[i]=bias2_list[i];
		}

//===========================================================================================================================
		for (int i=0; i<L1_q;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				if (i<L1_pi){
					w1bram_Q[i].a[j]=w1list[i][j];
				}
				else{
					w1bram_Q[i].a[j]=w1list[i-L1_pi][j];					
				}

			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
			w2bram_Q[i]=w2list_or[i][0];
		}
		for (int i=0; i<L2;i++){
			bias1_Q.a[i]=bias1_list[i];
		}
		bias2_Q=bias2_list[0];

	}

	if (Piwsync==1){
		for (int i=0; i<L1_pi;i++){
			for  (int j=0; j<L2;j++){
				#pragma HLS PIPELINE
					ap_axiu<32,0,0,0> v;
					float ww=w1bram_pi[i].a[j];
					v.data=ww; 
					Pitw1_axiout.write(v);
				}
			}
		// }
		for (int i=0; i<L2;i++){
			for  (int j=0; j<L3_pi;j++){
				#pragma HLS PIPELINE
				ap_axiu<32,0,0,0> v;
				float ww=w2bram_pi[i].a[j];
				v.data=ww; 
				Pitw2_axiout.write(v);

			}
		}
		for (int i=0; i<L2;i++){
			#pragma HLS PIPELINE
			ap_axiu<32,0,0,0> v;
			float ww=bias1_pi.a[i];
			v.data=ww; 
			Pitbias1_axiout.write(v);
		}
		for (int i=0; i<L3_pi;i++){
			#pragma HLS PIPELINE
			ap_axiu<32,0,0,0> v;
			float ww=bias2_pi.a[i];
			v.data=ww; 
			Pitbias2_axiout.write(v);
		}
	#ifndef __SYNTHESIS__
	printf("\nPi Sync done.\n");
	#endif
	}




	// must do: copy transposed w2bram,w1bram Q for bw
	Qa0blockvec w1bram_Q_copy[L2];
	w1blockvec w2bram_Q_copy; //w2 for BW, aggregate dim L2
	
	#pragma HLS aggregate variable=w2bram_Q_copy
	for (int j = 0; j < L2; j++){
		// #pragma HLS PIPELINE
		Qa0blockvec tmpcol;
		w2bram_Q_copy.a[j]=w2bram_Q[j];
		for (int i = 0; i < L1_q; i++){
			#pragma HLS PIPELINE
			tmpcol.a[i]=w1bram_Q[i].a[j];
		}
		w1bram_Q_copy[j]=tmpcol;
	}
	// must do: copy transposed w2bram Pi for bw
	// static Piw2blockvec w2bram_pi[L2]; //Qw2 target
	w1blockvec w2bram_pi_copy[L3_pi];
	for (int j = 0; j < L3_pi; j++){
		w1blockvec tmpcol;
		for (int i = 0; i < L2; i++){
			#pragma HLS PIPELINE
			tmpcol.a[i]=w2bram_pi[i].a[j];
		}
		w2bram_pi_copy[j]=tmpcol;
	}

	// float Qs_local[BATCHS];
	// float Loss_sqrt_local[BATCHS];
	// float Qs_local[128];
	// float Loss_sqrt_local[128];

	for(int ind=0; ind<BATCHS; ind++){
		#pragma HLS DATAFLOW
		PiloadIn(S, LSpipe0, LSpipe1, Pia0_buf_fifo, ind);
		// /*test begin*/
		// for (int i = 0; i < L1_pi; i++){ //s space
		// 	LSpipe0.read();
		// 	LSpipe1.read();
		// }
		// Pia0blockvec a0tmp=Pia0_buf_fifo.read();
		// /*test*/
		topPifw_l1(LSpipe0, Pia1_buf_fifo, bias1_pi, w1bram_pi, Pil1_pipe, Pil1actder_fifo,L1_pi,L2); //l1 inf for Pi
		topPifw_l2(Pil1_pipe, bias2_pi,w2bram_pi, Pil2_pipe, pitanh_actder,L2,L3_pi);//l2 inf for Pi ========================tanh on Pil2pipe===============
		/*test begin*/
		// for(int j = 0; j < L3_pi/T2; j++) {
		// 	for(int jj = 0; jj < T2; jj++) {
		// 		float tempC=Pil2_pipe.read();
		// 	}
		// }
		// for (int i = 0; i < L1_pi; i++){LSpipe1.read();}
		// Pia0blockvec a0tmp=Pia0_buf_fifo.read();
		// for(int j = 0; j < L2/T; j++) { 
		// 	for(int jj = 0; jj < T; jj++) {
		// 		Pil1actder_fifo.read();
		// 	}
		// }
		// Pia1_buf_fifo.read();
		/*test*/
		topPiconcatpipe(LSpipe1,Pil2_pipe,LSApipe);
		/*test begin*/
		// for (int i = 0; i < L1_pi+L3_pi; i++){
		// 	LSApipe.read();
		// }
		/*test*/
		fw_l1_Q(LSApipe, bias1_Q, w1bram_Q, Ql1_pipe, Ql1actder_fifo, L1_q,L2);//l1 inf for Q
		topPifw_l2_q(Ql1_pipe, bias2_Q,w2bram_Q, Ql2_pipe,L2);//l2 inf for Q
		/*test begin*/
		// for(int j = 0; j < L2; j++) { 
		// 	// Ql1_pipe.read();
		// 	Ql1actder_fifo.read();
		// }
		// Ql2_pipe.write(float(1));
		// float tempA = Ql2_pipe.read();
		/*test*/
		sub_backmm2Q(Ql2_pipe, w2bram_Q_copy, Ql1actder_fifo, Ql2back_pipe, L2);
		/*test begin*/
		// for(int j = 0; j < L2/T; j++) { 
		// 	for(int jj = 0; jj < T; jj++) {
		// 		Ql2back_pipe.read();
		// 	}
		// }
		/*test*/
		sub_backmm1Q(Ql2back_pipe, w1bram_Q_copy, Ql1back_pipe, L2, L1_q); 
		sub_backmm2pi(Ql1back_pipe, w2bram_pi_copy, Pil1actder_fifo, delt1_buf_fifo, delt2_buf_fifo, pitanh_actder, L3_pi, L2); //Ql1back_pipe input action space elements need to multiply by actder tanh of z2_pi
		wa1_pi(Pia0_buf_fifo, delt1_buf_fifo, wa1_buf, gr_bias1);
		wa2_pi(Pia1_buf_fifo, delt2_buf_fifo, wa2_buf, gr_bias2);

	}

	//================================================================================================

	#ifndef __SYNTHESIS__
	printf("\nPi wa1 content from fpga:\n");
	for(int i = 0; i < L1_pi/P3; i++) {
		for(int ii = 0; ii < P3; ii++) {
			for(int j = 0; j < L2/T3; j++) {
				for(int jj = 0; jj < T3; jj++) { 
					printf("%f ",wa1_buf[i][j][ii][jj]);
				}
			}
			printf("\n");
		}
	}
	// float wa2_buf[L2/P4][L3_pi/T4][P4_pi][T4_pi]={0};
	printf("\nPi wa2 content from fpga:\n");
	for(int i = 0; i < L2/P4; i++) {
		for(int ii = 0; ii < P4; ii++) {
			for(int j = 0; j < L3_pi/T4; j++) {
				for(int jj = 0; jj < T4; jj++) { 
					printf("%f ",wa2_buf[i][j][ii][jj]);
				}
			}
			printf("\n");
		}
	}
	printf("\nPi_top: Pi gr_bias1 content from fpga:\n");
	for (int i=0; i<L2;i++){
		printf("%f ",gr_bias1[i]);;
	}
	printf("\nPi_top: Pi gr_bias2 content from fpga:\n");
	for (int i=0; i<L3_pi;i++){
		printf("%f ",gr_bias2[i]);;
	}
	#endif

	float alpha_local=alpha;
	#pragma HLS array_partition variable=w1bram_pi type=cyclic  factor=2
	#pragma HLS array_partition variable=w2bram_pi type=cyclic  factor=8
	// WU: Substract -SGD (Add if SGA) WA from wbrams
	for(int i = 0; i < L1_pi/P3; i++) {
		for(int j = 0; j < L2/T3; j++) {
			for(int ii = 0; ii < P3; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w1bram_pi inter false
				for(int jj = 0; jj < T3; jj++) { 
					w1bram_pi[i*P3+ii].a[j*T3+jj] += alpha_local * wa1_buf[i][j][ii][jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nPi_top: w1bram_policy updated.\n");
	#endif

	for(int i = 0; i < L2/P4; i++) {
		for(int j = 0; j < L3_pi/T4; j++) {
			for(int ii = 0; ii < P4; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w2bram_pi inter false
				for(int jj = 0; jj < T4; jj++) { 
					w2bram_pi[i*P4+ii].a[j*T4+jj] += alpha_local * wa2_buf[i][j][ii][jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nPi_top: w2bram_pi updated.\n");
	#endif

	for (int i=0; i<L2;i++){
		bias1_pi.a[i] += alpha_local * gr_bias1[i];
	}
	for (int i=0; i<L3_pi;i++){
		bias2_pi.a[i] += alpha_local * gr_bias2[i];
	}


	#ifndef __SYNTHESIS__
	printf("\nPi_top: biases updated.\n");
	#endif

	//sync weights to cpu
	wb1wb:for(int i = 0; i < L1_pi; i++) {
		w1blockvec tmpw1b;
		#pragma HLS PIPELINE
		for(int jj = 0; jj < L2; jj++) { 
			tmpw1b.a[jj]=w1bram_pi[i].a[jj];
		}
		policyw1_out[i]=tmpw1b;
	}
	wb2wb:for(int i = 0; i < L2; i++) {
		#pragma HLS PIPELINE
		for(int jj = 0; jj < L3_pi; jj++) {
			policyw2_out[i].a[jj]=w2bram_pi[i].a[jj];
		}

	}
	for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
		bias1_out[i]=bias1_pi.a[i];
	}
	for (int i=0; i<L3_pi;i++){
		#pragma HLS PIPELINE
		bias2_out[i]=bias2_pi.a[i];
	}
	
	#ifndef __SYNTHESIS__
	printf("\nAll Transfer Finished.\n");
	#endif

}

}