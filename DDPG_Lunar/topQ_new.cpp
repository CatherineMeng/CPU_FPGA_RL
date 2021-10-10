#include "./topQ_new.h"
// #include "./common.h"

extern "C"{



// ind: increment from 0 to BATCHS/BSIZE
// loadIn(S,  Acts, LSApipe, Snt, LSNpipe0,  Qa0_buf_fifo, ind)
void loadIn(float S_In[],  float a_in[], hls::stream<float> &LSApipe, float Snt_In[], hls::stream<float> &LSNpipe0, hls::stream<float> &LSNpipe1, hls::stream<Qa0blockvec> &Qa0_buf_fifo,int ind){
	// float Inrows_local[LL];
	Qa0blockvec a0tmp;
	// #ifndef __SYNTHESIS__
	// printf("\nInput QSin read:\n");
	// #endif
	for (int i = 0; i < L1_pi; i++){ //s space
		#pragma HLS PIPELINE
		LSNpipe0.write(Snt_In[ind*L1_pi+i]);
		LSNpipe1.write(Snt_In[ind*L1_pi+i]);
		LSApipe.write(S_In[ind*L1_pi+i]); //assume first S then a
		a0tmp.a[i]=S_In[ind*L1_pi+i];//assume first S then a
		// #ifndef __SYNTHESIS__
		// {printf("%.8f ",S_In[ind*L1_pi+i]);}
		// #endif
	}
	

	for (int i = 0; i < L3_pi; i++){ //L1+pi+L3_pi=L1_q=s+a space
		#pragma HLS PIPELINE
		LSApipe.write(a_in[ind*L3_pi+i]);//assume first S then a
		// #ifndef __SYNTHESIS__
		// {printf("%.8f ",a_in[ind*L3_pi+i]);}
		// #endif
		a0tmp.a[L1_pi+i]=a_in[ind*L3_pi+i];//assume first S then a

	}
	// get a0_buf_fifo for WA
	Qa0_buf_fifo.write(a0tmp);
	
}


//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// void fw_l1(hls::stream<blockvec> &Inrows, float C[BSIZE/P][64/T][P][T],w1blockvec Wcols[], hls::stream<blockvec> &Crows, float Qa1_buf_fifo[L2][BSIZE], const int LL,const int LN) {
// void fw_l1(hls::stream<blockvec> &Inrows, float z1_buf[BSIZE/P][L2/T][P][T], float Qa1_buf_fifo[L2][BSIZE],float bias[], w1blockvec Wcols[], hls::stream<blockvec> &Crows, int actder[L2],const int LL,const int LN) {
void fw_l1(hls::stream<float> &Inrows, hls::stream<w1blockvec> &Qa1_buf_fifo, w1blockvec bias, w1blockvec Wcols[], hls::stream<float> &Crows, hls::stream<int> &Qactder_fifo,const int LL,const int LN) {

	// #pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Qa1_buf_fifo
	#pragma HLS aggregate variable=Wcols
	// #pragma HLS aggregate variable=Crows
	// #pragma HLS aggregate variable=Qactder_fifo
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
	printf("\nQz1_buf content:\n");//should be L2 rows, BSIZE columns
	#endif
	// get Qa1_buf_fifo for WA, find actder, write activatpon out to stream
	writeout: for(int j = 0; j < LN/T; j++) { //this factor consistent with Qa1_buf_fifo partition
		for(int jj = 0; jj < T; jj++) {
			float tempC;
			int actdertmp;
			#pragma HLS PIPELINE

			#ifndef __SYNTHESIS__
			printf("%.8f ",z1_buf_local[j][jj]);//should be L2 rows, BSIZE columns
			#endif

			float tmpz=(z1_buf_local[j][jj]>0)? z1_buf_local[j][jj]:0;
			// Qa1_buf_fifo[i*P+ii].a[j*T+jj]=tmpz;
			actdertmp=(z1_buf_local[j][jj]>0)? 1:0; //activation derivative
			tempC=tmpz; //activation

			Qactder_fifo.write(actdertmp);
			Crows.write(tempC);
		}
	}

	w1blockvec a1buftmp;
	for(int j = 0; j < LN/T; j++) { //this factor consistent with Qa1_buf_fifo partition
		#pragma HLS PIPELINE
		for(int jj = 0; jj < T; jj++) {
			 //activation
			a1buftmp.a[j*T+jj]=(z1_buf_local[j][jj]>0)? z1_buf_local[j][jj]:0;
		}
	}

	Qa1_buf_fifo.write(a1buftmp);

}


void fw_l1_targ(hls::stream<float> &Inrows,  w1blockvec bias, w1blockvec Wcols[], hls::stream<float> &Crows, const int LL,const int LN) {

	// #pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	// #pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=bias

	float z1_buf_local[L2/T][T];
	// #pragma HLS ARRAY_PARTITION variable=z1_buf_local dim=3 complete
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
				if (k==0) z1_buf_local[j][jj]=tempA * tempB.a[j*T+jj];
				else if (k==LL-1) z1_buf_local[j][jj] += (tempA * tempB.a[j*T+jj] + bias.a[j*T+jj]);
				else z1_buf_local[j][jj] += tempA * tempB.a[j*T+jj];
			}
		}
	}
	
	//write out to stream: next fw

	// #ifndef __SYNTHESIS__
	// printf("\n Q/Pi targ z1_buf content:\n");//should be L2 rows, BSIZE columns
	// #endif
	// get Qa1_buf_fifo for WA, find actder, write activatpon out to stream
	writeout: for(int j = 0; j < LN/T; j++) { //this factor consistent with Qa1_buf_fifo partition
		for(int jj = 0; jj < T; jj++) {
			float tempC;
			#pragma HLS PIPELINE
			 //activation
			tempC=(z1_buf_local[j][jj]>0)? z1_buf_local[j][jj]:0; //activation
			Crows.write(tempC);
			// #ifndef __SYNTHESIS__
			// printf("%.8f ",z1_buf_local[j][jj]);//should be L2 rows, BSIZE columns
			// #endif
		}
	}
}


// wu(C)

//Inrows: LL blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LN blockvecs (each batchsize)
// void fw_l2(hls::stream<blockvec> &Inrows, float z2_buf[BSIZE/P2][L3/T2][P2][T2], float bias[],Piw2blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN) {
void fw_l2_q(hls::stream<float> &Inrows, float bias,float Wcols[], hls::stream<float> &Crows,const int LL) {
	// #pragma HLS INLINE
	// #pragma HLS aggregate variable=Inrows
	// #pragma HLS aggregate variable=Wcols
	// #pragma HLS aggregate variable=Crows
	// #pragma HLS aggregate variable=bias
	// float C[BSIZE/P2][3/T2][P2][T2]={0};
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=4 complete

	float z2_buf_local;
	// #pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=2 complete
	#pragma HLS bind_storage variable=z2_buf_local type=RAM_2P impl=bram

	partialsum: for(int k=0; k < LL; k++) {
		float tempA = Inrows.read();
		float tempB = Wcols[k];

		#pragma HLS PIPELINE
		#pragma HLS dependence variable=z2_buf_local type=inter dependent=false
		if (k==0) z2_buf_local = tempA * tempB;
		else if (k==LL-1) z2_buf_local += (tempA * tempB + bias);
		else z2_buf_local += tempA * tempB;

	}


	// #ifndef __SYNTHESIS__
	// printf("\nQ/Qtarg z2_buf content:\n");

	// printf("%.8f ",z2_buf_local);

	
	// #endif
	Crows.write(z2_buf_local);


}

void fw_l2(hls::stream<float> &Inrows, Piw2blockvec bias,Piw2blockvec Wcols[], hls::stream<float> &Crows,const int LL,const int LN) {
	// #pragma HLS INLINE
	// #pragma HLS aggregate variable=Inrows
	#pragma HLS aggregate variable=Wcols
	// #pragma HLS aggregate variable=Crows
	#pragma HLS aggregate variable=bias
	// float C[BSIZE/P2][3/T2][P2][T2]={0};
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=3 complete
	// #pragma HLS ARRAY_PARTITION variable=z2_buf dim=4 complete

	float z2_buf_local[L3_pi/T2][T2];
	// #pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=3 complete
	#pragma HLS ARRAY_PARTITION variable=z2_buf_local dim=2 complete
	#pragma HLS bind_storage variable=z2_buf_local type=RAM_2P impl=bram

	// #ifndef __SYNTHESIS__
	// printf("\n\n===============================Input and weights for fw_l2 pi=============================\n ");
	// #endif

	partialsum: for(int k=0; k < LL; k++) {
	float tempA = Inrows.read();
	Piw2blockvec tempB = Wcols[k];
	// #ifndef __SYNTHESIS__
	// // for(int kk=0; kk < L3_pi; kk++)
	// printf("%f %f, ", tempA, tempB.a[0]);
	// printf("\n ");
	// #endif
     #pragma HLS aggregate variable=tempB
		partialsum_l2: for(int j = 0; j < LN/T2; j++) {
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=z2_buf_local type=inter dependent=false
			for(int jj = 0; jj < T2; jj++) { //3
				#pragma HLS UNROLL
				if (k==0) z2_buf_local[j][jj]=tempA * tempB.a[j*T2+jj];
				else if (k==LL-1) z2_buf_local[j][jj] += (tempA * tempB.a[j*T2+jj] + bias.a[j*T2+jj]);
				else z2_buf_local[j][jj] += tempA * tempB.a[j*T2+jj];
			}
		}
	}


	// #ifndef __SYNTHESIS__
	// printf("\nPi targ z2_buf content:\n");

	// for(int j = 0; j < LN/T2; j++) { //this factor consistent with Qa1_buf_fifo partition
	// 	for(int jj = 0; jj < T2; jj++) {
	// 		printf("%.8f ",z2_buf_local[j][jj]);
	// 		// printf("\n");
	// 	}
	// }
	// #endif
	// write out to stream
	for(int j = 0; j < LN/T2; j++) {
		for(int jj = 0; jj < T2; jj++) {
			#pragma HLS PIPELINE
			float tempC;
			if (z2_buf_local[j][jj]>1)tempC=1;
			else if (z2_buf_local[j][jj]<-1)tempC=-1;
			else tempC= z2_buf_local[j][jj]; //=======approx tanh==========
			// tempC=z2_buf_local[j][jj];
			Crows.write(tempC);
		}
	}

}


void concatpipe(hls::stream<float> &LSNpipe1, hls::stream<float> &Pitl2_pipe, hls::stream<float> &LSApipe1){ //assumes s first, a next
// #ifndef __SYNTHESIS__
// 	printf("\n\nInput_Q targ after concatenation:\n");
// #endif
	for (int i = 0; i < L1_pi; i++){
		#pragma HLS PIPELINE
		float tmp=LSNpipe1.read();
		LSApipe1.write(tmp);
	// #ifndef __SYNTHESIS__
	// printf("%f ", tmp);
	// #endif
	}
	// for (int i = 0; i < L1_q - L1_pi; i++){
	for (int i = 0; i < L3_pi; i++){
		#pragma HLS PIPELINE
		float tmp=Pitl2_pipe.read();
		LSApipe1.write(tmp);
	// #ifndef __SYNTHESIS__
	// printf("%f ", tmp);
	// #endif
	}

}


// r,a:BSIZE floats;
//Qrows, Qtrows: L3*BSIZE z2, aggregate BSIZE
//act_deriv(Qrows) hadamard* should be delt 2
//outs:L3*BSIZE, should be delt2 (aggregate BSIZE, used by bw)
//delt2_buf_fifo:L3*BSIZE, same content as outs, aggreegate L3 to be used in wu-gradient_compute
// objctv(r, acts, gamma, done, Q_pipe,Qtl2_pipe, loss_pipe, delt2_buf_fifo, ind,pn_out, Qs_local, Loss_sqrt_local);				
void objctv(float *r, float gamma, int *done, hls::stream<float> &Q_pipe,hls::stream<float> &Qtl2_pipe, hls::stream<float> &loss_pipe,
	hls::stream<float> &delt2_buf_fifo, int ind, hls::stream<ap_axiu<32,0,0,0>> &pn_out, float *Qs,float *Loss_sqrt){
	// #pragma HLS aggregate variable=Q_pipe
	// #pragma HLS aggregate variable=Qtl2_pipe
	// #pragma HLS aggregate variable=r
	// #pragma HLS aggregate variable=action
	// #pragma HLS aggregate variable=done

	#pragma HLS aggregate variable=loss_pipe
	#pragma HLS aggregate variable=pn_out


	float r_local=r[ind];
	int done_local=done[ind];

	float argmax_tq=Qtl2_pipe.read();
	// #ifndef __SYNTHESIS__
	// printf("argmax_tq:");
	// // for (int j=0;j<BSIZE;j++){
	// printf("%f ",argmax_tq);
	// // }
	// #endif


	float Qtransfer;
	float TDtransfer;

	float tmpq=Q_pipe.read();

	float tmpobj;
	#pragma HLS PIPELINE

	float actdertmp=1; //activation derivative is 1 since no relu at output layer
	// #ifndef __SYNTHESIS__
	// printf("\ntmpq:%f",tmpq);
	// #endif
	// tmpobj.a[j]=2*(tmpq.a[j]-r.a[j]*argmax_tq.a[j])*actdertmp; 
	float oneb=1-done_local; //cast fixed point to float
	tmpobj=2*(tmpq - r_local - oneb*gamma*argmax_tq)*actdertmp; 
	Qtransfer=tmpq;
	TDtransfer=tmpobj/2;
	// loss_ave[i]+=tmpobj;
	// #ifndef __SYNTHESIS__
	// printf("\nsample in batch-tmpobj:%f",tmpobj);
	// #endif

	loss_pipe.write(tmpobj);

	delt2_buf_fifo.write(tmpobj);
	ap_axiu<32,0,0,0> v;
	float td=TDtransfer;
	v.data=td; 
	pn_out.write(v);
	#ifndef __SYNTHESIS__
	printf("\n===================================pn_out send from Q=======================================:%f",float(v.data));
	#endif
	Qs[ind]=Qtransfer;
	Loss_sqrt[ind]=td;

	// #ifndef __SYNTHESIS__
	// printf("\n(index,Qs and Loss_sqrt) out of all BATCHS: (%d, %F, %F)\n",ind,Qs[ind],Loss_sqrt[ind]);
	// // }
	// printf("\ndelt2_buf_fifo content:%f\n",tmpobj);

	// #endif
}



//Inrows: LN blcokvecs (each batchsize)
//Wcols: LL wblockvecs (each LN)
//Crows: LL blockvecs (each batchsize)
// void sub_backmm2(hls::stream<blockvec> &Inrows, 
// 	Piw2blockvec Wcols0, Piw2blockvec Wcols1, Piw2blockvec Wcols2, Piw2blockvec Wcols3,
// 	Piw2blockvec Wcols4,Piw2blockvec Wcols5,Piw2blockvec Wcols6,Piw2blockvec Wcols7, hls::stream<blockvec> &Crows, 
// 	float delt1_buf[BSIZE/Pb][L3/Tb][Pb][Tb], const int LL,const int LN,int ind) {
// sub_backmm2(loss_pipe, w2bram_Q_copy, Ql1actder_fifo, delt1_buf_fifo, L2);
void sub_backmm2(hls::stream<float> &loss_pipe, w1blockvec Wcols, hls::stream<int> &Qactder_fifo, hls::stream<w1blockvec> &delt1_buf_fifo, const int LN){

	// #pragma HLS aggregate variable=loss_pipe
	// #pragma HLS aggregate variable=Wcols1s
	#pragma HLS aggregate variable=Wcols
	#pragma HLS aggregate variable=delt1_buf_fifo
	// #pragma HLS aggregate variable=Qactder_fifo


	float delt1_buf_local[L2/T][T]={0};
	#pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=2 complete
	// #pragma HLS ARRAY_PARTITION variable=delt1_buf_local dim=4 complete
	
	#pragma HLS bind_storage variable=delt1_buf_local type=RAM_2P impl=bram

	float tempA = loss_pipe.read();
	w1blockvec tempB = Wcols; //tempB size L2
     #pragma HLS aggregate variable=tempB

	for(int j = 0; j < L2/T; j++) { //LN is L2
	#pragma HLS PIPELINE
	#pragma HLS dependence variable=delt1_buf_local inter false
		for(int jj = 0; jj < T; jj++) { 
			delt1_buf_local[j][jj] = delt1_buf_local[j][jj] + tempA * tempB.a[j*T+jj];
		}
	}



	multactder:for(int j = 0; j < L2/T; j++) { 
		for(int jj = 0; jj < T; jj++) {
			int actdertmp = Qactder_fifo.read();
			#pragma HLS PIPELINE
			#pragma HLS dependence variable=delt1_buf_local inter false
			float tmpdelt=delt1_buf_local[j][jj];
			delt1_buf_local[j][jj] = (actdertmp!=0)? tmpdelt:0;	
		}
	}

	// #ifndef __SYNTHESIS__
	// printf("\ndelt1_buf_fifo content:\n");
	// #endif
	w1blockvec d1tmp;
	#pragma HLS aggregate variable=d1tmp
	writeout:for(int j = 0; j < L2/T; j++) { 
		#pragma HLS PIPELINE
		for(int jj = 0; jj < T; jj++) {
			d1tmp.a[j*T+jj] = delt1_buf_local[j][jj];
		// #ifndef __SYNTHESIS__
		// printf("%.8f ",delt1_buf_local[j][jj]);
		// #endif
		}
	}
	delt1_buf_fifo.write(d1tmp);


}



void wa1(hls::stream<Qa0blockvec> &Qa0_buf_fifo, hls::stream<w1blockvec> &delt1_buf_fifo, float wa1_buf[L1_q/P3][L2/T3][P3][T3], float gr_bias1[L2]){
	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4

	// #pragma HLS aggregate variable=a0_buf_fifo
	// #pragma HLS aggregate variable=delt1_buf_fifo
	// #pragma HLS aggregate variable=bias1
	#pragma HLS aggregate variable=delt1_buf_fifo


	Qa0blockvec a0tmp = Qa0_buf_fifo.read();
	w1blockvec d1tmp = delt1_buf_fifo.read();
	WA1partialsum:for(int i = 0; i < L1_q/P3; i++) {
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

}

void wa2(hls::stream<w1blockvec> &Qa1_buf_fifo, hls::stream<float> &delt2_buf_fifo, float wa2_buf[L2/P4][P4], float &gr_bias2){
	#pragma HLS array_partition variable=wa2_buf complete  dim=2

	// #ifndef __SYNTHESIS__
	// printf("\n===========================================================Qa1_buf_fifo content================================================================:\n");
	// #endif
	// for(int k=0; k < BSIZE; k++) {
	w1blockvec a1tmp = Qa1_buf_fifo.read();
	float d2tmp = delt2_buf_fifo.read();
	// #ifndef __SYNTHESIS__
	// printf("%f %f, ", a1tmp.a[0],d2tmp);
	// for (int i=0;i<L2;i++)printf("%f ",a1tmp.a[i]);
	// #endif
	WA2partialsum: for(int i = 0; i < L2/P4; i++) {
		// for(int i = 0; i < L2/P4; i++) {
		#pragma HLS PIPELINE
		#pragma HLS dependence variable=wa2_buf inter false
		for(int ii = 0; ii < P4; ii++) {
			wa2_buf[i][ii] = wa2_buf[i][ii] + a1tmp.a[i*P4+ii] * d2tmp;
		}
	}
	gr_bias2 += d2tmp;
}




//add qt weight sync signal: if Qwsync==0: init q & qt params; if Qwsync==1: let qt params=q params; else: keep updating q param; ; if Piwsync==1: let Pit params=Pi params; 
void learnersQ_top(float *S, float *Snt, float *acts, float *r, float gamma, float alpha, int *done, 
// w1blockvec Qw1_axiout[L1],Piw2blockvec Qw2_axiout[L2],float *Qbias1_axiout,float *Qbias2_axiout,
// w1blockvec Pitw1_axiin[L1],Piw2blockvec Pitw2_axiin[L2],float *Pitbias1_axiin,float *Pitbias2_axiin,
hls::stream<ap_axiu<32,0,0,0>> &Qw1_axiout, hls::stream<ap_axiu<32,0,0,0>> &Qw2_axiout, hls::stream<ap_axiu<32,0,0,0>> &Qbias1_axiout, hls::stream<ap_axiu<32,0,0,0>> &Qbias2_axiout,
hls::stream<ap_axiu<32,0,0,0>> &Pitw1_axiin, hls::stream<ap_axiu<32,0,0,0>> &Pitw2_axiin, hls::stream<ap_axiu<32,0,0,0>> &Pitbias1_axiin, hls::stream<ap_axiu<32,0,0,0>> &Pitbias2_axiin,
int Qwsync,int Piwsync, /*Learners args*/
float *Qs,float *Loss_sqrt,/*Logging args*/
hls::stream<ap_axiu<32,0,0,0>> &pn_out/*Replay args*/,int BATCHS){
	#pragma HLS INTERFACE m_axi port=S bundle=gmem1 offset=slave
	#pragma HLS INTERFACE m_axi port=Snt bundle=gmem2 offset=slave
	#pragma HLS INTERFACE m_axi port=acts bundle=gmem3 offset=slave
	#pragma HLS INTERFACE m_axi port=r bundle=gmem4 offset=slave
	#pragma HLS INTERFACE m_axi port=done bundle=gmem5 offset=slave
	#pragma HLS INTERFACE m_axi port=Qs bundle=gmem6 offset=slave
	#pragma HLS INTERFACE m_axi port=Loss_sqrt bundle=gmem7 offset=slave


	#pragma HLS INTERFACE s_axilite port=S
	#pragma HLS INTERFACE s_axilite port=Snt
	#pragma HLS INTERFACE s_axilite port=acts
	#pragma HLS INTERFACE s_axilite port=r
	#pragma HLS INTERFACE s_axilite port=done
	#pragma HLS INTERFACE s_axilite port=Qs
	#pragma HLS INTERFACE s_axilite port=Loss_sqrt

	#pragma HLS INTERFACE axis port=pn_out
	#pragma HLS INTERFACE axis port=Qw1_axiout
	#pragma HLS INTERFACE axis port=Qw2_axiout
	#pragma HLS INTERFACE axis port=Qbias1_axiout
	#pragma HLS INTERFACE axis port=Qbias2_axiout
	#pragma HLS INTERFACE axis port=Pitw1_axiin
	#pragma HLS INTERFACE axis port=Pitw2_axiin
	#pragma HLS INTERFACE axis port=Pitbias1_axiin
	#pragma HLS INTERFACE axis port=Pitbias2_axiin

	#pragma HLS INTERFACE s_axilite port=gamma
	#pragma HLS INTERFACE s_axilite port=alpha
	#pragma HLS INTERFACE s_axilite port=Qwsync
	#pragma HLS INTERFACE s_axilite port=Piwsync
	#pragma HLS INTERFACE s_axilite port=BATCHS
	#pragma HLS INTERFACE s_axilite port=return

	#pragma HLS aggregate variable=S
	#pragma HLS aggregate variable=Snt
	// #pragma HLS aggregate variable=Qw1_axiout
	// #pragma HLS aggregate variable=Qw2_axiout
	// #pragma HLS aggregate variable=Pitw1_axiin
	// #pragma HLS aggregate variable=Pitw2_axiin


	static w1blockvec w1bram_Q[L1_q]; //Qw1
	#pragma HLS aggregate variable=w1bram_Q
	#pragma HLS bind_storage variable=w1bram_Q type=RAM_2P impl=bram
	static float w2bram_Q[L2]; //Qw2
	#pragma HLS bind_storage variable=w2bram_Q type=RAM_2P impl=bram
	static w1blockvec bias1_Q;
	#pragma HLS aggregate variable=bias1_Q
	static float bias2_Q;	


	static w1blockvec w1bram_Qt[L1_q]; //Qw1 target
	#pragma HLS aggregate variable=w1bram_Qt
	#pragma HLS bind_storage variable=w1bram_Qt type=RAM_2P impl=bram
	static float w2bram_Qt[L2]; //Qw2 target
	#pragma HLS bind_storage variable=w2bram_Qt type=RAM_2P impl=bram
	static w1blockvec bias1_Qt;
	#pragma HLS aggregate variable=bias1_Qt
	static float bias2_Qt;

	// target networks - on-chip

	static w1blockvec w1bram_Pit[L1_pi]; //Pi w1_target
	#pragma HLS aggregate variable=w1bram_Pit
	#pragma HLS bind_storage variable=w1bram_Pit type=RAM_2P impl=bram 
	static Piw2blockvec w2bram_Pit[L2]; //Pi w2 target
	#pragma HLS aggregate variable=w2bram_Pit
	#pragma HLS bind_storage variable=w2bram_Pit type=RAM_2P impl=bram 
	static w1blockvec bias1_Pit;
	static Piw2blockvec bias2_Pit;
	#pragma HLS aggregate variable=bias1_Pit
	#pragma HLS aggregate variable=bias2_Pit
	
	

	

	// #ifndef __SYNTHESIS__
	// printf("\nacts:\n");
	// for(int j = 0; j < BATCHS*L3_pi; j++) {
	// 	// for(int i = 0; i < BSIZE; i++) {
	// 		printf("%d ",acts[j]);  //BS cols
	// 	// }
	// }
	// printf("\nr:\n");
	// for(int j = 0; j < BATCHS; j++) {
	// 	// for(int i = 0; i < BSIZE; i++) {
	// 		printf("%f ",r[j]);  //BS cols
	// 	// }
	// }
	// printf("\ndone:\n");
	// for(int j = 0; j < BATCHS; j++) {
	// 	// for(int i = 0; i < BSIZE; i++) 	{
	// 		printf("%d ",done[j]);  //BS cols
	// 	// }
	// }
	// #endif


	if (Qwsync==0){ //Init. Q network & target network (only executed exactly once in all iterations!)
	#ifndef __SYNTHESIS__
	printf("\nWeight init.\n");
	#endif
		for (int i=0; i<L1_q;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				if (i<L1_pi){
					w1bram_Q[i].a[j]=w1list[i][j];
					w1bram_Qt[i].a[j]=w1list[i][j];
					w1bram_Pit[i].a[j]=w1list[i][j];
				}
				else{
					w1bram_Q[i].a[j]=w1list[i-L1_pi][j];
					w1bram_Qt[i].a[j]=w1list[i-L1_pi][j];					
				}

			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
			// for  (int j=0; j<L3;j++){
			w2bram_Q[i]=w2list_or[i][0];w2bram_Qt[i]=w2list_or[i][0];
			for(int j=0; j<L3_pi; j++){
				w2bram_Pit[i].a[j]=w2list_or[i][j%2];
			}
			// }
		}
		for (int i=0; i<L2;i++){
			bias1_Q.a[i]=bias1_list[i];
			bias1_Qt.a[i]=bias1_list[i];
			bias1_Pit.a[i]=bias1_list[i];
		}

		bias2_Q=bias2_list[0];
		bias2_Qt=bias2_list[0];
		for (int i=0; i<L3_pi;i++){
			bias2_Pit.a[i]=bias2_list[i];
		}	

	}

	else if (Qwsync==1){ //sync target network with Q network
	#ifndef __SYNTHESIS__
	printf("\nTarget Weight sync.\n");
	#endif
		for (int i=0; i<L1_q;i++){
			#pragma HLS PIPELINE
			for  (int j=0; j<L2;j++){
				#pragma HLS UNROLL
				w1bram_Qt[i].a[j]=w1bram_Q[i].a[j];

			}
		}

		for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
	
				w2bram_Qt[i]=w2bram_Q[i];
			
		}
		for (int i=0; i<L2;i++){
			// #pragma HLS PIPELINE
			bias1_Qt.a[i]=bias1_Q.a[i];
		}
	
			// #pragma HLS PIPELINE
			bias2_Qt=bias2_Q;
		
	}



	// fw_bw(S,Snt,acts,r,done,w1bram_policy,w2bram_policy,w1bram_policy_t,w2bram_policy_t,bias1, bias2,bias1_t, bias2_t,gamma,  wa1_global,wa2_global);

	//================================================================================================

	// Inference chains to produce Q objctv:
	hls::stream<float> LSApipe("LSApipe");
	hls::stream<float> LSApipe1("LSApipe1");
	hls::stream<float> LSNpipe0("LSNpipe0");
	hls::stream<float> LSNpipe1("LSNpipe1");
	hls::stream<float> Ql1_pipe("Ql1pipe0");
	hls::stream<float> Q_pipe("Qpipe0");
	hls::stream<float> Pitl1_pipe("Pitl1_pipe");
	hls::stream<float> Pitl2_pipe("Pitl2_pipe");
	hls::stream<float> Qtl1_pipe("Qtl1_pipe");
	hls::stream<float> Qtl2_pipe("Qtl2_pipe");
	#pragma HLS STREAM variable=LSApipe depth=12 //L1
	#pragma HLS STREAM variable=LSApipe1 depth=12 //L1
	#pragma HLS STREAM variable=LSNpipe0 depth=8 //L1
	#pragma HLS STREAM variable=LSNpipe1 depth=32 //L1*4 stages
	#pragma HLS STREAM variable=Ql1_pipe depth=64 //L2
	#pragma HLS STREAM variable=Q_pipe depth=4 //L3
	#pragma HLS STREAM variable=Pitl1_pipe depth=64 //L2
	#pragma HLS STREAM variable=Pitl2_pipe depth=4 //L3
	#pragma HLS STREAM variable=Qtl1_pipe depth=64 //L2
	#pragma HLS STREAM variable=Qtl2_pipe depth=4 //L3
 	#pragma HLS bind_storage variable=LSApipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=LSApipe1 type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=LSNpipe0 type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=LSNpipe1 type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Ql1_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Q_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Pitl1_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Pitl2_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Qtl1_pipe type=fifo impl=SRL 
 	#pragma HLS bind_storage variable=Qtl2_pipe type=fifo impl=SRL 


	// Inference chains to produce Q objctv:
	hls::stream<Qa0blockvec> Qa0_buf_fifo("Qa0_buf_fifo");
	#pragma HLS STREAM variable=Qa0_buf_fifo depth=8 //??????????????????????
	#pragma HLS bind_storage variable=Qa0_buf_fifo type=fifo impl=SRL 
	// int actder[L2];

	hls::stream<int> Ql1actder_fifo("Ql1actder_fifo");
	#pragma HLS STREAM variable=Ql1actder_fifo depth=192 //actderdepth??????????????????????
	#pragma HLS bind_storage variable=Ql1actder_fifo type=fifo impl=SRL 
	// // w1blockvec delt1_buf[BSIZE];

	hls::stream<w1blockvec> delt1_buf_fifo("delt1_buf_fifo");
	#pragma HLS STREAM variable=delt1_buf_fifo depth=1 //BSIZE??????????????????????
	#pragma HLS bind_storage variable=delt1_buf_fifo type=fifo impl=SRL 


	// w1blockvec Qa1_buf_fifo[BSIZE];
	hls::stream<w1blockvec> Qa1_buf_fifo("Qa1_buf_fifo");
	#pragma HLS STREAM variable=Qa1_buf_fifo depth=3 //a1depth??????????????????????
	#pragma HLS bind_storage variable=Qa1_buf_fifo type=fifo impl=SRL 


	hls::stream<float> delt2_buf_fifo("delt2_buf_fifo"); //delta2 for wu, produced by obj, parallel access on L3 dimension
	#pragma HLS STREAM variable=delt2_buf_fifo depth=1 //BSIZE??????????????????????
	#pragma HLS bind_storage variable=delt2_buf_fifo type=fifo impl=SRL

	hls::stream<float> loss_pipe("loss_pipe");
	#pragma HLS STREAM variable=loss_pipe depth=4 //L3
	#pragma HLS bind_storage variable=loss_pipe type=fifo impl=SRL

	// float delt1_buf[BSIZE][L2]; 
	// w1blockvec delt1_buf[BSIZE]={0}; //delta1 for wu, produced by sub_backmm2, parallel access on L2 dimension
	float wa1_buf[L1_q/P3][L2/T3][P3][T3]={0};
	float wa2_buf[L2/P4][P4]={0};
	float gr_bias1[L2]={0};
	float gr_bias2=0;

	// #pragma HLS array_partition variable=w2bram_policy type=cyclic  factor=8 

	#pragma HLS array_partition variable=wa1_buf complete  dim=3
	#pragma HLS array_partition variable=wa1_buf complete  dim=4
	#pragma HLS array_partition variable=wa2_buf complete  dim=2

	// must do: copy w2bram Q for bw
	w1blockvec w2bram_Q_copy; //w2 for BW, aggregate dim L2
	#pragma HLS aggregate variable=w2bram_Q_copy
	for (int j = 0; j < L2; j++){
		#pragma HLS PIPELINE
		w2bram_Q_copy.a[j]=w2bram_Q[j];
	}


	// must do: copy Q to SLR2 fot target training (witeout)	// ==============================================DO THIS==============================================
	// Piwsync==1: read pitarg from SLR2 (read. optional:update pitarg parameters)
// hls::stream<ap_axiu<32,0,0,0>> &Qw1_axiout, hls::stream<ap_axiu<32,0,0,0>> &Qw2_axiout, hls::stream<ap_axiu<32,0,0,0>> &Qbias1_axiout, hls::stream<ap_axiu<32,0,0,0>> &Qbias2_axiout,
// hls::stream<ap_axiu<32,0,0,0>> &Pitw1_axiin[L1], hls::stream<ap_axiu<32,0,0,0>> &Pitw2_axiin[L2], hls::stream<ap_axiu<32,0,0,0>> &Pitbias1_axiin, hls::stream<ap_axiu<32,0,0,0>> &Pitbias2_axiin,

	#ifndef __SYNTHESIS__
	printf("\nQ_top: Q init done. Q send axis content:\n");
	#endif

	for (int i=0; i<L1_q;i++){
		for  (int j=0; j<L2;j++){
			#pragma HLS PIPELINE
			ap_axiu<32,0,0,0> v;
			float ww=w1bram_Q[i].a[j];
			v.data=ww; 
			Qw1_axiout.write(v);
			#ifndef __SYNTHESIS__
			printf("%f ", ww);
			#endif

		}
		#ifndef __SYNTHESIS__
		printf("\n");
		#endif
	}
	for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
		ap_axiu<32,0,0,0> v;
		float ww=w2bram_Q[i];
		v.data=ww; 
		Qw2_axiout.write(v);
	}
	for (int i=0; i<L2;i++){
		#pragma HLS PIPELINE
		ap_axiu<32,0,0,0> v;
		float ww=bias1_Q.a[i];
		v.data=ww; 
		Qbias1_axiout.write(v);
	}
	ap_axiu<32,0,0,0> v;
	float ww=bias2_Q;
	v.data=ww; 
	Qbias2_axiout.write(v);


	if (Piwsync==1){
		for (int i=0; i<L1_pi;i++){
			for  (int j=0; j<L2;j++){
				#pragma HLS PIPELINE
				ap_axiu<32,0,0,0> pwtmp=Pitw1_axiin.read();
				w1bram_Pit[i].a[j]=pwtmp.data;
			}
		}
		for (int i=0; i<L2;i++){
			for  (int j=0; j<L3_pi;j++){
				#pragma HLS PIPELINE
				ap_axiu<32,0,0,0> pwtmp=Pitw2_axiin.read();
				w2bram_Pit[i].a[j]=pwtmp.data;
			}
		}
		for (int i=0; i<L2;i++){
			#pragma HLS PIPELINE
			ap_axiu<32,0,0,0> pwtmp=Pitbias1_axiin.read();
			bias1_Pit.a[i]=pwtmp.data;
		}
		for (int i=0; i<L3_pi;i++){
			#pragma HLS PIPELINE
			ap_axiu<32,0,0,0> pwtmp=Pitbias2_axiin.read();
			bias2_Pit.a[i]=pwtmp.data;
		}
	}


	// float Qs_local[BATCHS];
	// float Loss_sqrt_local[BATCHS];
	float Qs_local[128];
	float Loss_sqrt_local[128];
	for(int ind=0; ind<BATCHS; ind++){
		#pragma HLS DATAFLOW
		loadIn(S,  acts, LSApipe, Snt, LSNpipe0, LSNpipe1, Qa0_buf_fifo, ind);
		fw_l1(LSApipe, Qa1_buf_fifo, bias1_Q, w1bram_Q, Ql1_pipe, Ql1actder_fifo,L1_q,L2); //l1 inf for Q
		fw_l1_targ(LSNpipe0, bias1_Pit, w1bram_Pit, Pitl1_pipe, L1_pi,L2);//l1 inf for Pit
		fw_l2_q(Ql1_pipe, bias2_Q,w2bram_Q, Q_pipe,L2);//l2 inf for Q
		fw_l2(Pitl1_pipe, bias2_Pit,w2bram_Pit, Pitl2_pipe,L2,L3_pi);//l2 inf for Pit =================================tanh on Pil2pipe==================

		concatpipe(LSNpipe1,Pitl2_pipe,LSApipe1);
		/*test begin*/
		// for (int i = 0; i < L1_pi; i++){
		// 	LSNpipe1.read();
		// }
		// for (int i = 0; i < L1_q - L1_pi; i++){
		// 	Pitl2_pipe.read();
		// }
		// Qa0_buf_fifo.read();
		// for(int j = 0; j < L2; j++) { //this factor consistent with Qa1_buf_fifo partition
		// 	Ql1actder_fifo.read();
		// }
		// Qa1_buf_fifo.read();
		// Q_pipe.read();
		// for (int i = 0; i < L1_pi+L3_pi; i++){
		// 	LSApipe1.read();
		// }
		/*test*/

		fw_l1_targ(LSApipe1, bias1_Qt, w1bram_Qt, Qtl1_pipe, L1_q,L2);//l1 inf for Qt
		fw_l2_q(Qtl1_pipe, bias2_Qt,w2bram_Qt, Qtl2_pipe,L2);//l2 inf for Qt

		objctv(r, gamma, done, Q_pipe,Qtl2_pipe, loss_pipe, delt2_buf_fifo, ind,pn_out, Qs_local, Loss_sqrt_local);		
		sub_backmm2(loss_pipe, w2bram_Q_copy, Ql1actder_fifo, delt1_buf_fifo, L2);
		wa1(Qa0_buf_fifo, delt1_buf_fifo, wa1_buf, gr_bias1);
		wa2(Qa1_buf_fifo, delt2_buf_fifo, wa2_buf, gr_bias2);

	}

	//================================================================================================

	#ifndef __SYNTHESIS__
	printf("\nQ wa1 content from fpga:\n");
	for(int i = 0; i < L1_q/P3; i++) {
		for(int ii = 0; ii < P3; ii++) {
			for(int j = 0; j < L2/T3; j++) {
				for(int jj = 0; jj < T3; jj++) { 
					printf("%f ",wa1_buf[i][j][ii][jj]);
				}
			}
			printf("\n");
		}
	}
	printf("\nQ wa2 content from fpga:\n");
	for(int i = 0; i < L2/P4; i++) {
			for(int ii = 0; ii < P4; ii++) {
					printf("%f ",wa2_buf[i][ii]);
			}
	}
	printf("\nQ gr_bias1 content from fpga:\n");
	for (int i=0; i<L2;i++){
		printf("%f ",gr_bias1[i]);;
	}
	printf("\nQ gr_bias2 content from fpga:\n");
	printf("%f ",gr_bias2);
	#endif


	float alpha_local=alpha;
	#pragma HLS array_partition variable=w1bram_Q type=cyclic  factor=2
	#pragma HLS array_partition variable=w2bram_Q type=cyclic  factor=8
	// WU: Substract -SGD (Add if SGA) WA from wbrams
	for(int i = 0; i < L1_q/P3; i++) {
		for(int j = 0; j < L2/T3; j++) {
			for(int ii = 0; ii < P3; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w1bram_Q inter false
				for(int jj = 0; jj < T3; jj++) { 
					w1bram_Q[i*P3+ii].a[j*T3+jj] -= alpha_local * wa1_buf[i][j][ii][jj];
					// policyw1_out[i*P3+ii].a[j*T3+jj]=w1bram_policy[i*P3+ii].a[j*T3+jj];
				}
			}
		}
	}

	#ifndef __SYNTHESIS__
	printf("\nw1bram_policy updated.\n");
	#endif

	for(int i = 0; i < L2/P4; i++) {
		// for(int j = 0; j < L3/T4; j++) {
			for(int ii = 0; ii < P4; ii++) {
				#pragma HLS PIPELINE
				#pragma HLS dependence variable=w2bram_Q inter false
				// for(int jj = 0; jj < T4; jj++) { 
					w2bram_Q[i*P4+ii]-= alpha_local * wa2_buf[i][ii];
					// policyw2_out[i*P4+ii].a[j*T4+jj]=w2bram_policy[i*P4+ii].a[j*T4+jj];
				// }
			}
		// }
	}

	#ifndef __SYNTHESIS__
	printf("\nw2bram_policy updated.\n");
	#endif

	for (int i=0; i<L2;i++){
		bias1_Q.a[i] -= alpha_local * gr_bias1[i];
		// bias1_out[i]=bias1.a[i];
	}
	// for (int i=0; i<L3;i++){
		bias2_Q -= alpha_local * gr_bias2;
		// bias2_out[i]=bias2.a[i];
	// }


	#ifndef __SYNTHESIS__
	printf("\nbiases updated.\n");
	#endif

	//sync weights to cpu
	// {
	// #pragma HLS DATAFLOW
	// wb1wb:for(int i = 0; i < L1_q; i++) {
	// 	w1blockvec tmpw1b;
	// 	#pragma HLS PIPELINE
	// 	for(int jj = 0; jj < L2; jj++) { 
	// 		tmpw1b.a[jj]=w1bram_policy[i].a[jj];
	// 	}
	// 	policyw1_out[i]=tmpw1b;
	// }
	// wb2wb:for(int i = 0; i < L2; i++) {
	// 	#pragma HLS PIPELINE
	// 	for(int jj = 0; jj < L3; jj++) {
	// 		policyw2_out[i].a[jj]=w2bram_policy[i].a[jj];
	// 	}

	// }
	// for (int i=0; i<L2;i++){
	// 	#pragma HLS PIPELINE
	// 	bias1_out[i]=bias1.a[i];
	// }
	// for (int i=0; i<L3;i++){
	// 	#pragma HLS PIPELINE
	// 	bias2_out[i]=bias2.a[i];
	// }
	// // }	
	for (int i=0; i<BATCHS;i++){
		#pragma HLS PIPELINE
		Qs[i]=Qs_local[i];
		Loss_sqrt[i]=Loss_sqrt_local[i];
	}
	// // }	
	#ifndef __SYNTHESIS__
	printf("\nAll Transfer Finished.\n");
	#endif

}

}