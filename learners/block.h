
#include "hls_stream.h"
#include "hls_math.h"
#include "ap_fixed.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{
using namespace std;


//model metadata
//batchsize:BSIZE last-layer dimension:LL next-layer dimension:LN
//mmult:A(BSIZE,LL)*B(LL,LN)
const int BATCHS = 16;
const int BSIZE = 1;
const int L1 = 8;
const int L2 = 64;
const int L3 = 4;

typedef ap_fixed<1,1> sglbit;


typedef struct {
	sglbit a[BSIZE];
} bsbit;

//hardware parameters
//PE array dimensions: FW
// #define P 4 //BSiZE
#define P 1 //BSiZE
//#define T 32
#define T 16 //L2

// #define P2 4 //BSiZE
#define P2 1 //BSiZE
#define T2 2 //L3

//PE array dimensions: BW
// #define Pb 4 //BSiZE
#define Pb 1 //BSiZE,=P
#define Tb 16 //L2,=T

//PE array dimensions: WU
// #define P3 2 //L1 partition?
#define P3 2 //L1 partition?
#define T3 8 //L2

// #define P4 8 //L2
#define P4 8 //L2
#define T4 4 //L3


const float w1list[L1][L2]={{0.09915440529584885,-0.8309909701347351,0.13474726676940918,0.5200338363647461,0.48192429542541504,0.035555850714445114,0.051557619124650955,-0.28794270753860474,-0.4198712706565857,0.47878023982048035,-0.01150917075574398,0.20277462899684906,-0.08225031942129135,0.3318197429180145,-0.125244140625,0.27792811393737793,0.3145216703414917,-0.3707144260406494,-0.0437040850520134,0.16427592933177948,-0.395137220621109,0.2238987684249878,0.14374248683452606,0.13287387788295746,-0.11609049886465073,0.19251202046871185,-0.6004387140274048,0.24150680005550385,-0.26290208101272583,-0.040665119886398315,0.003040146781131625,0.4833458662033081,-0.3267051577568054,-0.24715998768806458,0.133119136095047,-0.13170768320560455,0.25504955649375916,0.31076404452323914,-0.03626742586493492,0.10338196158409119,0.20394128561019897,0.3194485604763031,-0.28020772337913513,0.076445572078228,0.04087008163332939,-0.04186327010393143,-0.448120653629303,0.26069989800453186,-0.05521703138947487,0.3642311692237854,0.26647427678108215,-0.3300744593143463,0.535096287727356,0.6776754856109619,0.24327489733695984,0.1575140506029129,-0.04243846982717514,-0.4964044988155365,-0.22224193811416626,-0.3914795517921448,-0.24925018846988678,0.224340558052063,-0.6668941974639893,-0.3923123776912689},
{0.29402482509613037,-0.9124258756637573,0.3868218660354614,0.15521517395973206,0.3627452254295349,0.1945621818304062,-0.05356447398662567,-0.7290803790092468,-0.40737491846084595,-0.06972579658031464,0.5093024373054504,-0.1072436198592186,0.5642536282539368,0.22090131044387817,0.5055398344993591,0.10796497017145157,-0.11692631989717484,0.04013124853372574,0.13325399160385132,0.21380500495433807,-0.1611955314874649,-0.2943866550922394,0.2821629047393799,-0.5286908149719238,-0.07214821875095367,-0.19923631846904755,-0.12820714712142944,0.05931665003299713,-0.13442353904247284,-0.2754851281642914,-0.11323124170303345,-0.3667425513267517,0.03932627663016319,0.3275429606437683,0.47407254576683044,-0.9340253472328186,0.1975119709968567,0.30425944924354553,-0.19112633168697357,-0.3640998601913452,-0.6373295187950134,-0.2897416055202484,-0.6274580955505371,-0.27683424949645996,0.3310238718986511,0.33827605843544006,-0.2314598709344864,0.20969133079051971,0.30090540647506714,0.18265189230442047,0.6124370694160461,0.02167748101055622,-0.06957324594259262,0.4615097939968109,-0.2425176203250885,0.09540935605764389,-0.2902863621711731,-0.3907780349254608,0.3408726751804352,0.09108509123325348,-0.7972047924995422,0.9296848177909851,-0.802964985370636,0.23137196898460388},
{-0.37352311611175537,0.28179872035980225,0.3858104348182678,-0.5339908003807068,0.0900442972779274,0.41252848505973816,0.4958907663822174,0.4973900616168976,0.1033104732632637,-0.2689152956008911,0.014870917424559593,-0.5497537851333618,0.30827853083610535,-0.06827765703201294,0.5602136850357056,-0.3666709363460541,-0.4393084943294525,-0.23005343973636627,-0.3604196608066559,-0.22145211696624756,-0.29811564087867737,-0.42761489748954773,-0.04393099993467331,0.19725771248340607,0.28327298164367676,0.1081341877579689,0.6643447279930115,-0.12907177209854126,-0.039714161306619644,-0.15272729098796844,0.016995809972286224,-0.512271523475647,-0.060260944068431854,0.07478131353855133,0.39070019125938416,0.4271097481250763,-0.3438448905944824,-0.16565744578838348,0.6141490936279297,0.25293394923210144,-0.32208627462387085,-0.20217138528823853,0.31227609515190125,-0.4444103538990021,0.31543442606925964,0.42581382393836975,-0.380502849817276,0.5807292461395264,0.014070669189095497,-0.4794534146785736,-0.2946072816848755,0.40890058875083923,0.36513784527778625,-0.5256085991859436,0.5445166826248169,0.5455231070518494,-0.15579883754253387,0.18869075179100037,0.34308308362960815,0.6005241274833679,-0.17468447983264923,-0.13289466500282288,0.4793927073478699,-0.1427803784608841},
{-0.25456106662750244,0.9773491621017456,0.299529492855072,0.251888245344162,0.07446911931037903,0.19960547983646393,-0.0041734897531569,0.46327346563339233,0.488626629114151,0.16968736052513123,-0.5416762828826904,0.17450395226478577,0.11985796689987183,-0.06186104193329811,0.32334813475608826,0.3375132381916046,0.2176358699798584,-0.22453539073467255,0.7033613920211792,0.3027912378311157,0.3514239192008972,-0.20568811893463135,-0.25031718611717224,-0.4173189699649811,0.07292021065950394,-0.10581216961145401,0.6590656638145447,0.28976449370384216,0.17516271770000458,-0.16585083305835724,0.26592984795570374,-0.10276421159505844,-0.2065814882516861,0.26086196303367615,0.17223313450813293,0.6362319588661194,-0.19731250405311584,0.2625041604042053,0.3410726487636566,-0.18610863387584686,-0.09226897358894348,0.1227460727095604,0.48164644837379456,-0.1789538860321045,-0.41541874408721924,-0.06111585721373558,0.7817159295082092,0.19583731889724731,0.13402779400348663,-0.38287922739982605,-0.8275179862976074,-0.06864684075117111,-0.46500736474990845,-0.3939872980117798,-0.2684434652328491,0.4617224633693695,-0.24161744117736816,0.578322172164917,0.16034111380577087,0.24599230289459229,0.7027877569198608,-0.9096449613571167,0.48795193433761597,0.18304197490215302},
{0.09915440529584885,-0.8309909701347351,0.13474726676940918,0.5200338363647461,0.48192429542541504,0.035555850714445114,0.051557619124650955,-0.28794270753860474,-0.4198712706565857,0.47878023982048035,-0.01150917075574398,0.20277462899684906,-0.08225031942129135,0.3318197429180145,-0.125244140625,0.27792811393737793,0.3145216703414917,-0.3707144260406494,-0.0437040850520134,0.16427592933177948,-0.395137220621109,0.2238987684249878,0.14374248683452606,0.13287387788295746,-0.11609049886465073,0.19251202046871185,-0.6004387140274048,0.24150680005550385,-0.26290208101272583,-0.040665119886398315,0.003040146781131625,0.4833458662033081,-0.3267051577568054,-0.24715998768806458,0.133119136095047,-0.13170768320560455,0.25504955649375916,0.31076404452323914,-0.03626742586493492,0.10338196158409119,0.20394128561019897,0.3194485604763031,-0.28020772337913513,0.076445572078228,0.04087008163332939,-0.04186327010393143,-0.448120653629303,0.26069989800453186,-0.05521703138947487,0.3642311692237854,0.26647427678108215,-0.3300744593143463,0.535096287727356,0.6776754856109619,0.24327489733695984,0.1575140506029129,-0.04243846982717514,-0.4964044988155365,-0.22224193811416626,-0.3914795517921448,-0.24925018846988678,0.224340558052063,-0.6668941974639893,-0.3923123776912689},
{0.29402482509613037,-0.9124258756637573,0.3868218660354614,0.15521517395973206,0.3627452254295349,0.1945621818304062,-0.05356447398662567,-0.7290803790092468,-0.40737491846084595,-0.06972579658031464,0.5093024373054504,-0.1072436198592186,0.5642536282539368,0.22090131044387817,0.5055398344993591,0.10796497017145157,-0.11692631989717484,0.04013124853372574,0.13325399160385132,0.21380500495433807,-0.1611955314874649,-0.2943866550922394,0.2821629047393799,-0.5286908149719238,-0.07214821875095367,-0.19923631846904755,-0.12820714712142944,0.05931665003299713,-0.13442353904247284,-0.2754851281642914,-0.11323124170303345,-0.3667425513267517,0.03932627663016319,0.3275429606437683,0.47407254576683044,-0.9340253472328186,0.1975119709968567,0.30425944924354553,-0.19112633168697357,-0.3640998601913452,-0.6373295187950134,-0.2897416055202484,-0.6274580955505371,-0.27683424949645996,0.3310238718986511,0.33827605843544006,-0.2314598709344864,0.20969133079051971,0.30090540647506714,0.18265189230442047,0.6124370694160461,0.02167748101055622,-0.06957324594259262,0.4615097939968109,-0.2425176203250885,0.09540935605764389,-0.2902863621711731,-0.3907780349254608,0.3408726751804352,0.09108509123325348,-0.7972047924995422,0.9296848177909851,-0.802964985370636,0.23137196898460388},
{-0.37352311611175537,0.28179872035980225,0.3858104348182678,-0.5339908003807068,0.0900442972779274,0.41252848505973816,0.4958907663822174,0.4973900616168976,0.1033104732632637,-0.2689152956008911,0.014870917424559593,-0.5497537851333618,0.30827853083610535,-0.06827765703201294,0.5602136850357056,-0.3666709363460541,-0.4393084943294525,-0.23005343973636627,-0.3604196608066559,-0.22145211696624756,-0.29811564087867737,-0.42761489748954773,-0.04393099993467331,0.19725771248340607,0.28327298164367676,0.1081341877579689,0.6643447279930115,-0.12907177209854126,-0.039714161306619644,-0.15272729098796844,0.016995809972286224,-0.512271523475647,-0.060260944068431854,0.07478131353855133,0.39070019125938416,0.4271097481250763,-0.3438448905944824,-0.16565744578838348,0.6141490936279297,0.25293394923210144,-0.32208627462387085,-0.20217138528823853,0.31227609515190125,-0.4444103538990021,0.31543442606925964,0.42581382393836975,-0.380502849817276,0.5807292461395264,0.014070669189095497,-0.4794534146785736,-0.2946072816848755,0.40890058875083923,0.36513784527778625,-0.5256085991859436,0.5445166826248169,0.5455231070518494,-0.15579883754253387,0.18869075179100037,0.34308308362960815,0.6005241274833679,-0.17468447983264923,-0.13289466500282288,0.4793927073478699,-0.1427803784608841},
{-0.25456106662750244,0.9773491621017456,0.299529492855072,0.251888245344162,0.07446911931037903,0.19960547983646393,-0.0041734897531569,0.46327346563339233,0.488626629114151,0.16968736052513123,-0.5416762828826904,0.17450395226478577,0.11985796689987183,-0.06186104193329811,0.32334813475608826,0.3375132381916046,0.2176358699798584,-0.22453539073467255,0.7033613920211792,0.3027912378311157,0.3514239192008972,-0.20568811893463135,-0.25031718611717224,-0.4173189699649811,0.07292021065950394,-0.10581216961145401,0.6590656638145447,0.28976449370384216,0.17516271770000458,-0.16585083305835724,0.26592984795570374,-0.10276421159505844,-0.2065814882516861,0.26086196303367615,0.17223313450813293,0.6362319588661194,-0.19731250405311584,0.2625041604042053,0.3410726487636566,-0.18610863387584686,-0.09226897358894348,0.1227460727095604,0.48164644837379456,-0.1789538860321045,-0.41541874408721924,-0.06111585721373558,0.7817159295082092,0.19583731889724731,0.13402779400348663,-0.38287922739982605,-0.8275179862976074,-0.06864684075117111,-0.46500736474990845,-0.3939872980117798,-0.2684434652328491,0.4617224633693695,-0.24161744117736816,0.578322172164917,0.16034111380577087,0.24599230289459229,0.7027877569198608,-0.9096449613571167,0.48795193433761597,0.18304197490215302}};

const float w2list_or[L2][2]={{-0.4689616858959198,-0.4767073094844818},
{-0.37833964824676514,0.022839855402708054},
{0.6040470600128174,0.6315500736236572},
{-0.41051146388053894,-0.6404814124107361},
{0.6181333661079407,0.5410789251327515},
{0.39266183972358704,0.5167668461799622},
{0.428280770778656,0.6119735240936279},
{-0.10109279304742813,0.021177515387535095},
{0.6874445676803589,0.4756735861301422},
{0.6230390071868896,0.3775746822357178},
{0.1173565611243248,-0.11833798140287399},
{-0.5806930661201477,-0.630471408367157},
{0.5844029784202576,0.5395036339759827},
{0.2512100636959076,0.6177409887313843},
{0.49379339814186096,0.5373432636260986},
{-0.6176011562347412,-0.5000641942024231},
{-0.3841506540775299,-0.5539496541023254},
{0.45566579699516296,0.48201173543930054},
{-0.5290756225585938,-0.44343817234039307},
{-0.49623972177505493,-0.6636725068092346},
{0.20165085792541504,0.1148453801870346},
{-0.49979132413864136,-0.6255320906639099},
{-0.5090369582176208,-0.530059278011322},
{-0.4088982939720154,-0.5858308672904968},
{0.35558342933654785,0.5102614164352417},
{-0.46581774950027466,-0.5211271047592163},
{0.8065334558486938,0.46706101298332214},
{-0.24635440111160278,0.1712368130683899},
{0.5917474031448364,0.6263952255249023},
{0.40904301404953003,0.5923664569854736},
{-0.4242604076862335,-0.5671769380569458},
{-0.5024348497390747,-0.5641963481903076},
{0.42168480157852173,0.6133358478546143},
{-0.2648867666721344,-0.529242217540741},
{0.33836278319358826,0.5613265037536621},
{-0.4739691913127899,-0.023466341197490692},
{-0.584425151348114,-0.5758948922157288},
{0.4822394847869873,0.6085575222969055},
{0.3310735523700714,0.28702402114868164},
{-0.5098034739494324,-0.5074024200439453},
{-0.15460051596164703,0.031576965004205704},
{-0.36798739433288574,-0.6111109852790833},
{-0.2626047432422638,0.11659052968025208},
{-0.4526749849319458,-0.5236098766326904},
{0.525458037853241,0.46553167700767517},
{0.4760657548904419,0.6353740096092224},
{-0.4929998219013214,-0.30910512804985046},
{0.5883882641792297,0.5674290657043457},
{0.6220492720603943,0.6470704674720764},
{0.11523482203483582,-0.029681595042347908},
{0.3072708249092102,0.06090688332915306},
{0.49542486667633057,0.6723721027374268},
{0.016627812758088112,-0.07643172144889832},
{0.05909078195691109,-0.20803500711917877},
{0.4083706736564636,0.612116277217865},
{0.4871865510940552,0.5440035462379456},
{-0.569324254989624,-0.6282317042350769},
{0.5354588627815247,0.18616966903209686},
{0.322813481092453,0.4987541437149048},
{0.6255694031715393,0.5202617645263672},
{-0.4636342227458954,-0.2169896364212036},
{0.3186837434768677,-0.03358057513833046},
{-0.4077446460723877,-0.18266665935516357},
{-0.4635222256183624,-0.6030444502830505}};


typedef struct {
	float a[BSIZE];
} blockvec;

typedef struct {
	int a[BSIZE];
} actvec;

typedef struct {
	float a[L2];
} w1blockvec;
typedef struct {
	float a[L3];
} w3blockvec;
//typedef struct {
//	float a[L4];
//} w3blockvec;
//typedef struct {
//	float out[BLOCK_SIZE][BLOCK_SIZE];
//} blockmat;

void loadIn(blockvec In[],  hls::stream<blockvec> &Inrows,const int LL,int ind);
//void loadW(w1blockvec W[], hls::stream<blockvec> &Wcols, int LL);
// void fw_l1(hls::stream<blockvec> &Inrows, float z1_buf[BSIZE/P][L2/T][P][T], float bias[], w1blockvec Wcols[], hls::stream<blockvec> &Crows, const int LL,const int LN);
void fw_l1(hls::stream<blockvec> &Inrows, float z1_buf[BSIZE/P][L2/T][P][T], float bias[], w1blockvec Wcols[], hls::stream<blockvec> &Crows, bsbit actder[L2],const int LL,const int LN);
void fw_l2(hls::stream<blockvec> &Inrows, float z2_buf[BSIZE/P2][L3/T2][P2][T2], float bias[],w3blockvec Wcols[], hls::stream<blockvec> &Crows,const int LL,const int LN);
void objctv(blockvec r, actvec action, hls::stream<blockvec> &Qrows,hls::stream<blockvec> &Qtrows,blockvec outs[],float delt2_buf[BSIZE][L3]);
// void sub_backmm2(hls::stream<blockvec> &Inrows, 
// 	w3blockvec Wcols0, w3blockvec Wcols1, w3blockvec Wcols2, w3blockvec Wcols3,
// 	w3blockvec Wcols4,w3blockvec Wcols5,w3blockvec Wcols6,w3blockvec Wcols7, hls::stream<blockvec> &Crows, 
// 	float z2_buf[BSIZE/Pb][L3/Tb][Pb][Tb], const int LL,const int LN,int ind);
// void activation(hls::stream<blockvec> &Inrows, hls::stream<blockvec> &Outrows,const int L);
// vvoid sub_backmm2(blockvec Inrows[], w3blockvec Wcols[], float z1_buf[BSIZE/P][L2/T][P][T],float delt1_buf[BSIZE/P][L2/T][P][T], const int LL,const int LN);
void sub_backmm2(blockvec Inrows[], w3blockvec Wcols[], bsbit actder[L2],float delt1_buf[BSIZE/P][L2/T][P][T], const int LL,const int LN);
void actderiv(hls::stream<blockvec> &Inrows, hls::stream<blockvec> &Outrows,const int L);
void storeDDR(blockvec C[],  hls::stream<blockvec> &Crows,  const int LN);
void fw_bw(blockvec *A,w1blockvec w1bram[],w3blockvec w2bram[],float bias1[],float bias2[],float a0_buf[L1][BSIZE],float a1_buf[L2][BSIZE],float delt2_buf[BSIZE][L3],float delt1_buf[BSIZE/P][64/T][P][T]);
void learners_top(blockvec *S);
}
