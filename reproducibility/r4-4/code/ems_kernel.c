/* Deterministic hourly EMS kernel; all powers are fleet totals, dt=1h.
   Legacy diagnostic and state-limited executable policies share economics. */
#include <math.h>
int simulate(const double *x,const double *pv,const double *wt,const double *load,
 const double *w,const double *p,int mode,double *o,double *tr){
 int n[6];for(int j=0;j<6;j++)n[j]=x[j]>0?(int)ceil(x[j]):0;
 double ecmax=n[2]*p[0],fcmax=n[3]*p[1],hc=n[4]*p[2],eb=n[5]*p[3];
 double hmin=hc*p[4],hmax=hc*p[5],emin=eb*p[6],emax=eb*p[7];
 double etaec=p[8],etafc=p[9],etac=p[10],etad=p[11],hhv=p[12];
 double throughput=0,fcfade=0,ecfade=0,varom=0;
 for(int d=0;d<5;d++){
 double h=.5*(hmin+hmax),e=.5*(emin+emax),h0=h,e0=e,prevfc=0,prevec=0;
 for(int t=0;t<24;t++){
 int k=d*24+t;double net=n[0]*pv[k]+n[1]*wt[k]-load[k],ec=0,fc=0,ch=0,dis=0;
 if(net>0){ec=fmin(net,ecmax);
 if(mode)ec=fmin(ec,fmax(0,(hmax-h)*hhv/etaec));
 ch=fmin(net-ec,n[5]*p[13]);if(mode)ch=fmin(ch,fmax(0,(emax-e)/etac));
 }else {double dem=-net;fc=fmin(dem,fcmax);
 if(mode)fc=fmin(fc,fmax(0,(h-hmin)*hhv*etafc));
 dis=fmin(dem-fc,n[5]*p[14]);if(mode)dis=fmin(dis,fmax(0,(e-emin)*etad));
 if(dem-fc-dis>.01){
 int energy_limited=mode && ((fc+1e-9<fmin(dem,fcmax)) || (dis+1e-9<fmin(dem-fc,n[5]*p[14])));
 o[0]=energy_limited?3:1;o[1]=dem-fc-dis;o[2]=energy_limited?2:0;o[3]=d;o[4]=t;o[5]=1e6+1000*o[1];return (int)o[0];}}
 double hp=h,ep=e;h+=(ec*etaec-fc/etafc)/hhv;e+=ch*etac-dis/etad;
 if(!mode){if(h>hmax || h<hmin){o[0]=h>hmax?2:3;o[1]=h>hmax?h-hmax:hmin-h;o[2]=0;o[3]=d;o[4]=t;o[5]=1e6+500*fabs(o[1]);return (int)o[0];}
 if(eb>0 && (e/eb>p[7] || e/eb<p[6])){o[0]=e/eb>p[7]?2:3;o[1]=e/eb>p[7]?e/eb-p[7]:p[6]-e/eb;o[2]=1;o[3]=d;o[4]=t;o[5]=1e6+500*fabs(o[1]);return (int)o[0];}}
 throughput+=(ch+dis)*w[d];varom+=(ec*p[15]+fc*p[16]+(ch+dis)*p[17])*w[d];
 fcfade+=(p[18]*fabs(fc-prevfc)+((fc>0&&fc<.2*fcmax)?p[19]:0)+((fc>.8*fcmax)?p[20]:0)+((fc>0&&prevfc==0)?p[21]:0))*w[d];
 ecfade+=((ec>0?p[22]:0)+(ec>0&&prevec==0?p[23]:0))*w[d];prevfc=fc;prevec=ec;
 if(tr){double *r=tr+k*12;r[0]=net;r[1]=ec;r[2]=fc;r[3]=ch;r[4]=dis;r[5]=h;r[6]=eb>0?e/eb:.5;r[7]=fmax(0,net-ec-ch);r[8]=ec*(1-etaec)+fc*(1/etafc-1)+ch*(1-etac)+dis*(1/etad-1);r[9]=(h-hp)*hhv;r[10]=e-ep;r[11]=0;}
 }
 if(eb>0 && e/eb+1e-12<.5*(p[6]+p[7])-1e-6){o[0]=3;o[1]=(e0-e)/eb;o[2]=1;o[3]=d;o[4]=24;o[5]=1e6+1e4*o[1];return 3;}
 if(h+1e-12<h0-1e-9){o[0]=3;o[1]=h0-h;o[2]=0;o[3]=d;o[4]=24;o[5]=1e6+1e4*o[1];return 3;}
 }
 o[0]=0;o[6]=throughput;o[7]=fcfade;o[8]=ecfade;o[9]=varom;return 0;
}
