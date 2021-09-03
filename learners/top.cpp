#include "./block.h"

extern "C"{

// ind: increment from 0 to BATCHS/BSIZE
//in total (from outer loop in top): need to read (BATCHS/BSIZE)*LL time blockvec = BATCHS*LL numbers
void loadIn(blockvec In[],  hls::stream<blockvec> &Inrows,const int LL,int ind){
	for (int i = 0; i < LL; i++){
		#pragma HLS PIPELINE
		Inrows.write(In[ind*LL+i]);
	}
}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// void fw_l1(hls::stream<blockvec> &Inrows, float C[BSIZE/P][64/T][P][T],w1blockvec Wcols[], hls::stream<blockvec> &Crows, float a1_buf[L2][BSIZE], const int LL,const int LN) {
void fw_l1(hls::stream<blockvec> &Inrows, float z1_buf[BSIZE/P][L2/T][P][T], float bias[], w1blockvec Wcols[], hls::stream<blockvec> &Crows, bsbit actder[L2],const int LL,const int LN) {
	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=actder
	#pragma HLS ARRAY_PARTITION variable=z1_buf dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z1_buf dim=4 complete

	float z1_buf_local[BSIZE/P][L2/T][P][T]={0};
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
						z1_buf_local[i][j][ii][jj] += tempA.a[i*P+ii] * tempB.a[j*T+jj];
					}
				}
			}
		}
	}
	
	//add bias, find actder
	
	#ifndef __SYNTHESIS__
	printf("\nz1_buf content:\n");//should be L2 rows, BSIZE columns
	#endif
	for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf partition
		for(int jj = 0; jj < T; jj++) {
			for(int i = 0; i < BSIZE/P; i++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=z1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					z1_buf_local[i][j][ii][jj]+=bias[j*T+jj];
					actder[j*T+jj].a[i*P+ii]=(z1_buf_local[i][j][ii][jj]>0)? 1:0;
					z1_buf[i][j][ii][jj]=z1_buf_local[i][j][ii][jj];
					#ifndef __SYNTHESIS__
					printf("%.8f ",z1_buf_local[i][j][ii][jj]);
					#endif
				}
			}
			#ifndef __SYNTHESIS__
			printf("\n");
			#endif
		}
	}

	//write out to stream
	
	for(int j = 0; j < LN/T; j++) {
		for(int jj = 0; jj < T; jj++) {
			blockvec tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P; i++) {
				#pragma HLS PIPELINE
				for(int ii = 0; ii < P; ii++) {
					tempC.a[i*P+ii]=z1_buf_local[i][j][ii][jj];
				}
			}
			Crows.write(tempC);
		}
	}

}

// wu(C)

//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
void fw_l2(hls::stream<blockvec> &Inrows, float z2_buf[BSIZE/P2][L3/T2][P2][T2], float bias[],w3blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN) {
	#pragma HLS INLINE
	#pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=Crows
	// float C[BSIZE/P2][3/T2][P2][T2]={0};
	#pragma HLS ARRAY_PARTITION variable=z2_buf dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z2_buf dim=4 complete

	float z2_buf_local[BSIZE/P2][L3/T2][P2][T2]={0};
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=4 complete

	#pragma HLS bind_storage variable=z2_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < LL; k++) {
	blockvec tempA = Inrows.read();
	w3blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P2; i++) {
			for(int j = 0; j < LN/T2; j++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=z2_buf_local inter false
				for(int ii = 0; ii < P2; ii++) {
					#pragma HLS UNROLL
					for(int jj = 0; jj < T2; jj++) { //3
						#pragma HLS UNROLL
						z2_buf_local[i][j][ii][jj] = z2_buf_local[i][j][ii][jj] + tempA.a[i*P2+ii] * tempB.a[j*T2+jj];
					}
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nz2_buf content:\n");//should be L3 rows, BSIZE columns
	#endif
	//add bias
	for(int j = 0; j < LN/T2; j++) { //this factor consistent with a1_buf partition
		for(int jj = 0; jj < T2; jj++) {
			for(int i = 0; i < BSIZE/P2; i++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=z2_buf_local inter false
				for(int ii = 0; ii < P2; ii++) {
					z2_buf_local[i][j][ii][jj]+=bias[j*T+jj];
					#ifndef __SYNTHESIS__
					printf("%.8f ",z2_buf_local[i][j][ii][jj]);
					#endif
				}
			}
			#ifndef __SYNTHESIS__
			printf("\n");
			#endif
		}
	}

	//write out to stream
	for(int j = 0; j < LN/T2; j++) {
		for(int jj = 0; jj < T2; jj++) {
			blockvec tempC;
			#pragma HLS aggregate variable=tempC
			for(int i = 0; i < BSIZE/P2; i++) {
				#pragma HLS PIPELINE
				for(int ii = 0; ii < P2; ii++) {
					tempC.a[i*P2+ii]=z2_buf_local[i][j][ii][jj];
					z2_buf[i][j][ii][jj]=z2_buf_local[i][j][ii][jj];
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
//delt2_buf:L3*BSIZE, same content as outs, aggreegate L3 to be used in wu-gradient_compute
void objctv(blockvec r, actvec action, hls::stream<blockvec> &Qrows,hls::stream<blockvec> &Qtrows,
	blockvec outs[],float delt2_buf[BSIZE][L3]){
	#pragma HLS aggregate variable=Qrows
	#pragma HLS aggregate variable=Qtrows

	// Get argmax target Q vals of size BSIZE
	blockvec argmax_tq={0};
	for (int i=0;i<L3;i++){
		#pragma HLS PIPELINE
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
	for (int i=0;i<L3;i++){
		// #pragma HLS PIPELINE
		blockvec tmpq=Qrows.read();
		blockvec tmpobj;
		for (int j=0;j<BSIZE;j++){
			#pragma HLS PIPELINE
			if (i==action.a[j])
			{
				float actder=(tmpq.a[j]>0)? 1:0; //relu derivative
				#ifndef __SYNTHESIS__
				printf("\ntmpq.a[%d]:%f",j,tmpq.a[j]);
				#endif
				tmpobj.a[j]=2*(tmpq.a[j]-r.a[j]*argmax_tq.a[j])*actder;
				#ifndef __SYNTHESIS__
				printf("\nnode %d, tmpobj.a[%d]:%f",i,j,tmpobj.a[j]);
				#endif
			}
			else
				tmpobj.a[j]=0;
			//write to delt2_buf
			delt2_buf[j][i]=tmpobj.a[j];
			#ifndef __SYNTHESIS__
			if(delt2_buf[j][i]!=0)printf("\ndelt2_buf[%d][%d]:%f",j,i,delt2_buf[j][i]);
			#endif
		}
		outs[i]=(tmpobj);
	}
	#ifndef __SYNTHESIS__
	printf("\ndelt2_buf content:\n");//should be L3 rows, BSIZE columns
	for (int i=0;i<L3;i++){
		for (int j=0;j<BSIZE;j++){
			printf("%f ",delt2_buf[j][i]);
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
void sub_backmm2(blockvec Inrows[], w3blockvec Wcols[], bsbit actder[L2],float delt1_buf[BSIZE/P][L2/T][P][T], const int LL,const int LN){
	#pragma HLS aggregate variable=Inrows
	// #pragma HLS aggregate variable=Wcols1s
	#pragma HLS aggregate variable=Crows
	// float z2_buf[BSIZE/Pb][L3/Tb][Pb][Tb]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=delt1_buf dim=4 complete
	// #pragma HLS ARRAY_PARTITION variable=z1_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z1_buf dim=4 complete

	#pragma HLS array_partition variable=Wcols type=cyclic  factor=8 
	// w3blockvec * arraywq[8]; //the size+ #ports is equal to Tb

	float delt1_buf_local[BSIZE/P][L2/T][P][T]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=4 complete
	
	#pragma HLS bind_storage variable=delt1_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < LL; k++) {
		blockvec tempA = Inrows[k];
		// w3blockvec tempB = Wcols[k];
    #pragma HLS aggregate variable=tempA
     // #pragma HLS aggregate variable=tempB
		for(int i = 0; i < BSIZE/P; i++) {
			for(int j = 0; j < LN/T; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=delt1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					// #pragma HLS UNROLL
					for(int jj = 0; jj < T; jj++) { //3
						// #pragma HLS UNROLL
						// delt1_buf[i][j][ii][jj] = delt1_buf[i][j][ii][jj] + tempA.a[i*Pb+ii] * (*arraywq[j*Tb+jj]).a[k]; //*arraywq: because wcols partitioned in cyclic manner, adjacent indices are in different banks
						delt1_buf_local[i][j][ii][jj] = delt1_buf_local[i][j][ii][jj] + tempA.a[i*P+ii] * (Wcols[j*T+jj]).a[k];
					}
				}
			}
		}

	}
	#ifndef __SYNTHESIS__
	printf("\ndelt1_buf content before z1 :\n\n");//should be L3 rows, BSIZE columns
	for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf partition
		for(int jj = 0; jj < T; jj++) {
			for(int i = 0; i < BSIZE/P; i++) {
				for(int ii = 0; ii < P; ii++) {
					printf("%.8f ",delt1_buf_local[i][j][ii][jj]);
				}
			}
			printf("\n");

		}
	}
	#endif

	for(int j = 0; j < LN/T; j++) { 
		for(int jj = 0; jj < T; jj++) {
			for(int i = 0; i < BSIZE/P; i++) {
				#pragma HLS PIPELINE
				// #pragma HLS dependence variable=z1_buf_local inter false
				for(int ii = 0; ii < P; ii++) {
					// delt times z1 relu derivative
					// delt1_buf_local[i][j][ii][jj] = (z1_buf[i][j][ii][jj]>0)? delt1_buf[i][j][ii][jj]:0;
					delt1_buf_local[i][j][ii][jj] = (actder[j*T+jj].a[i*P+ii]!=0)? delt1_buf_local[i][j][ii][jj]:0;
					delt1_buf[i][j][ii][jj] = delt1_buf_local[i][j][ii][jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\ndelt1_buf content after z1:\n\n");//should be L3 rows, BSIZE columns
	for(int j = 0; j < LN/T; j++) { //this factor consistent with a1_buf partition
		for(int jj = 0; jj < T; jj++) {
			for(int i = 0; i < BSIZE/P; i++) {
				for(int ii = 0; ii < P; ii++) {
					printf("%.8f ",delt1_buf[i][j][ii][jj]);
				}
			}
			printf("\n");

		}
	}
	#endif

}



void activation(hls::stream<blockvec> &Inrows, hls::stream<blockvec> &Outrows,const int L){
	#ifndef __SYNTHESIS__
	printf("\na1:\n");
	#endif
	for (int i = 0; i < L; i++){
		#pragma HLS PIPELINE
		blockvec temp = Inrows.read();
		blockvec temp_out;
		//tanh:
		// for (int j = 0; j < BSIZE; j++){
		// 	#pragma HLS UNROLL
		// 	temp_out.a[j]=hls::tanh(temp.a[j]);
		// }
		//relu:
		for (int j = 0; j < BSIZE; j++){
			#pragma HLS UNROLL
			temp_out.a[j]=(temp.a[j]>0) ? temp.a[j]:0;
			#ifndef __SYNTHESIS__
			printf("%.8f ",temp_out.a[j]);
			#endif
		}
		Outrows.write(temp_out);
	}
}


void storeDDR(blockvec C[],  hls::stream<blockvec> &Crows,  const int LN){
	for (int i = 0; i < LN; i++){
		#pragma HLS PIPELINE
   printf("In itr %d\n",i);
		C[i] = Crows.read();
	}
 printf("Yaaassss\n");

}

void fw_bw(blockvec *A,w1blockvec w1bram[],w3blockvec w2bram[],float bias1[],float bias2[],float a0_buf[L1][BSIZE],float a1_buf[L2][BSIZE],float delt2_buf[BSIZE][L3],float delt1_buf[BSIZE/P][L2/T][P][T]){

	#pragma HLS INTERFACE m_axi port=A bundle=gmem0 offset=slave
	#pragma HLS INTERFACE s_axilite port=A bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS array_partition variable=a0_buf type=block  factor=8  dim=1

	hls::stream<blockvec> inpipe;

	hls::stream<blockvec> outpipe[6];

	// #pragma HLS array_partition variable=outpipe complete

	#pragma HLS STREAM variable=inpipe depth=64
	#pragma HLS STREAM variable=outpipe depth=64



	#ifndef __SYNTHESIS__
	printf("\nw2bram sampled content:\n");
	for  (int j=0; j<L3;j++){
		printf("%f ",w2bram[0].a[j]);
	}
	printf("\n");
	for  (int j=0; j<L3;j++){
		printf("%f ",w2bram[2].a[j]);
	}
	printf("\n");
	for  (int j=0; j<L3;j++){
		printf("%f ",w2bram[63].a[j]);
	}
	#endif

	
	blockvec r={1}; 
	actvec acts={2};

	#ifndef __SYNTHESIS__
	for (int j = 0; j < BSIZE; j++){
	#pragma HLS PIPELINE
		acts.a[j]=j+2;
	}
	#endif
	blockvec outpipe6[L3];

		float z1_buf[BSIZE/P][L2/T][P][T]={0};
		float z2_buf[BSIZE/P2][L3/T2][P2][T2]={0};
		bsbit actder[L2]={0};
	for(int ind=0; ind<1; ind++){

		
		// {
			#pragma HLS DATAFLOW
			loadIn(A, inpipe, L1,ind);
			fw_l1(inpipe, z1_buf, bias1, w1bram, outpipe[0], actder,L1,L2);
		  	activation(outpipe[0], outpipe[1],L2);
			fw_l2(outpipe[1], z2_buf, bias2,w2bram, outpipe[2],L2,L3);
			
			// consistent with python golden tb
			for (int i = 0; i < L3; i++){
				blockvec tmpt;
				for (int j = 0; j < BSIZE; j++){
				#pragma HLS PIPELINE
					tmpt.a[j]=i+2;
				}
				outpipe[5].write(tmpt);
			}
			objctv(r, acts, outpipe[2],outpipe[5],outpipe6, delt2_buf);
			
			// sub_backmm2(hls::stream<blockvec> &Inrows, w3blockvec Wcols[], float z1_buf[BSIZE/P][64/T][P][T],float delt1_buf[BSIZE/P][64/T][P][T], const int LL,const int LN) {
			sub_backmm2(outpipe6, w2bram, actder, delt1_buf, L3,L2);
		// }
	}
}

//add learners input interfaces: blockvec *R,actvec *Acts,  blockvec *Snt,actvec *Dn,
//add learners output interfaces(back to cpu): w1blockvec *w1bram,w3blockvec *w2bram
//add replay inputs: int insert_signal,int insert_ind,
//add replay outputs: int ind_o[]
//add qt weight sync signal
void learners_top(blockvec *S){
	w1blockvec w1bram[L1]; //w1
	w3blockvec w2bram[L2]; //w2
	#pragma HLS bind_storage variable=w2bram type=RAM_2P impl=bram

//	Init on-chip memory
	
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
			if (j<2) w2bram[i].a[j]=w2list_or[i][j];
			else w2bram[i].a[j]=w2list_or[i][j-2];
		}
	}

	float a0_buf[L1][BSIZE]; //a0 for wu, parallel access on L1 dimension
	// float z1_buf[BSIZE/P][L2/T][P][T]; //z1 for bw, produced by fw_l1,
	float a1_buf[L2][BSIZE]; //a1 for wu, produced by fw_l1, parallel access on L2 dimension
	float delt2_buf[BSIZE][L3]={0}; //delta2 for wu, produced by obj, parallel access on L3 dimension
	// float delt1_buf[BSIZE][L2]; 
	float delt1_buf[BSIZE/P][L2/T][P][T]={0}; //delta1 for wu, produced by sub_backmm2, parallel access on L2 dimension
	// #pragma HLS array_partition variable=w2bram type=cyclic  factor=8 
	// #pragma HLS array_partition variable=a0_buf type=block  factor=8  dim=1

  	float bias1[L2]={-1.1225467920303345,-0.5253201723098755,0.8014744520187378,-1.078803539276123,0.7526521682739258,0.8947911262512207,1.030880331993103,-0.11566359549760818,0.8868575096130371,0.7529403567314148,0.0815008357167244,-0.7764682173728943,0.7573199272155762,0.700654923915863,0.9816205501556396,-0.7538471221923828,-0.8123699426651001,1.0642855167388916,-0.7657874822616577,-1.0403845310211182,0.27166444063186646,-0.9976863861083984,-0.9416283965110779,-1.0264735221862793,0.5003169775009155,-1.057175874710083,0.8024879693984985,-0.06931311637163162,0.7876960635185242,1.0516828298568726,-1.1551307439804077,-0.9914983510971069,1.115867018699646,-0.8172269463539124,0.751054584980011,-0.5702265501022339,-0.8541625142097473,1.0552011728286743,0.5897875428199768,-0.9063143730163574,-0.0014912269543856382,-0.8715770840644836,-0.2481270581483841,-1.0776419639587402,0.8115789294242859,0.8825179934501648,-0.6865957379341125,0.8269904851913452,0.7347946763038635,0.12292467802762985,0.4563320279121399,0.8172180652618408,-0.058201681822538376,-0.00186146458145231,1.0419872999191284,0.944339394569397,-0.919138491153717,0.6711363196372986,0.930547833442688,0.8000667691230774,-0.5643067955970764,0.45937785506248474,-0.688703179359436,-1.0188298225402832};
  	float bias2[L3]={0.46203604340553284,0.37419500946998596,0.3,0.1};
  	// void fw_bw(blockvec *A,w1blockvec w1bram[],w3blockvec w2bram[],float bias1[],float bias2[],float a0_buf[L1][BSIZE],float a1_buf[L2][BSIZE],float delt2_buf[BSIZE][L3],float delt1_buf[BSIZE][L2]){
	fw_bw(S,w1bram,w2bram,bias1, bias2,a0_buf,a1_buf,delt2_buf,delt1_buf);
	for (int i = 0; i < L2; i++){
		for (int j = 0; j < BSIZE; j++){
		#pragma HLS PIPELINE
			a0_buf[i][j]=S[i].a[j];
		}
	}

}

}

