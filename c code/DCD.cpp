#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>

#define MAXITR 500
#define inf HUGE_VAL
#define EPS 0.05

enum ModelType{L1LOSS=1,L2LOSS};
using namespace std;

typedef long int li;
typedef vector< vector<double> > vvd;
typedef vector< vector<li> > vvl;
typedef vector<double> vd;
typedef vector<li> vl;

class Data{
public:
    vvd x;
    vvl ind;
    vector<int> y;
    li m,n,x0,elements;

    Data(){
        x0=1;
    }

    void readinput(string file){
        ifstream input(file.c_str());
        string line,token,a; 
        stringstream iss,ss;
        string::size_type sz;
        vd tx;
        vl ti;
        m=0;
        n=0;
        elements=0;
        while ( getline(input, line) )
        {   
        	li elem=0;
            li count=0;
            iss << line;
            int ty;
            iss>>ty;
            m++;
            y.push_back(ty);
            while(iss>>token){
                ss<<token;
                getline(ss, a, ':');
                count=stol(a,&sz);
                ti.push_back(count-1);
                getline(ss, a, ':');
                tx.push_back(stof(a,&sz));
                elem++;
                ss.clear();
            }
            iss.clear();
            x.push_back(tx);
            ind.push_back(ti);
            ti.clear();
            tx.clear();
            if (count > n)
                n=count;
            if (elem > elements)
                elements=elem;
        }
        cout<<"reading file complete\n\n";
    }

    void printData(){
        cout<<m<<" "<<n<<endl;
        for(li i=0;i<m;i++){
            cout<<y[i]<<" ";
            for(li j=0;j<ind[i].size();j++){
                cout<<ind[i][j]+1<<":"<<x[i][j]<<" ";
            }
            cout<<endl;
        }
    }
};

class TrainingModel{
public:
    double lambda,c1,c2,w0;
    vd w;
    int nsv;
    ModelType type;
    vd Beta,Alpha,Qii;

    vl initialzation(Data tr,double *alpha,double* beta,double* qii,double &d){
    	lambda=23.5;
        c1=1.0;
        c2=1.0;
    	if(type==L2LOSS)
    		d=0;
    	else
    		d=0.5/c1;
    		d=0.5/c1;
        
        w=vd(tr.n,0);
        w0=0;
        vl seq;
        li m=tr.m;
        li n;
        // qii is sum of sqare of each data point(sum of sqaure of each feature)
        // qii=sum(x(i,:).^2,2);
        // w=(y.*(beta-alpha))'*x;
        for (li i=0;i<m;i++){
            double Q=1.0;
            beta[i]=0;
            alpha[i]=1.0/m;
            n=tr.ind[i].size();
            for(li j=0;j<n;j++){
            	double x=tr.x[i][j];
            	li p=tr.ind[i][j];
                w[p]+=(tr.y[i]*(beta[i]-alpha[i])*x);
                Q+=(x*x);
            }
            w0+=(tr.y[i]*(beta[i]-alpha[i]));
            qii[i]=(Q+d);
            seq.push_back(i);

        }
        //printModel(tr.m,tr.n);
        // int u;
        // cin>>u;
        
        cout<<"initialzation done\n";
        return seq;
    }

    void train(Data tr){
    	double alpha[tr.m],beta[tr.m],qii[tr.m];
    	bool converge=false;
    	double d;
    	vl seq=initialzation(tr,alpha,beta,qii,d);
    	clock_t begin, end;
		double time_spent;

		begin = clock();
		
        li changedvariable=0;
        double dela,delb,Ga,Gb;
        double alphaold,betaold;
        int itr=0;

        while(!converge){
            changedvariable=0;
            random_shuffle(seq.begin(), seq.end());
            for(li k=0;k<tr.m;k++){
            	li i=seq[k];
                double Gb=w0;
                dela=0,delb=0;
                li n=tr.ind[i].size();
                double xi[n];
				copy(tr.x[i].begin(), tr.x[i].end(), xi);
                for(li j=0;j<n;j++)
                    Gb+=(xi[j]*w[tr.ind[i][j]]);
                Gb=tr.y[i]*Gb-1+beta[i]*d;
                double Ga=lambda-Gb-1+(beta[i]+alpha[i])*d;
                //cout<<i+1 <<" Ga ="<<Ga<<" Gb="<<Gb<<" qii="<<qii[i]<<endl;
                bool fa=true,fb=true;
                // updating alpha
                if(fabs(Ga)<1e-4)
                    fa=false;
                else if(alpha[i]<=1e-5 && Ga >= 0)
                    fa=false;
                else if(alpha[i]>=c2-1e-5 && Ga <= 0)
                    fa=false;
                else
                    fa=true;
                if(fa){
                    alphaold=alpha[i];
                    alpha[i]=min(max((alpha[i]-Ga/qii[i]),0.0),c2);
                    //cout<<alpha[i]<<endl;
                    dela=alpha[i]-alphaold;
                    if(fabs(dela)>=1e-4)
                        changedvariable++;
                }

                // updating beta
                if(fabs(Gb)<1e-4)
                    fb=false;
                else if(beta[i]<=1e-5 && Gb >= 0)
                    fb=false;
                else if(beta[i]>=c1-1e-5 && Gb <= 0)
                    fb=false;
                else
                    fb=true;
                if(fb){
                    betaold=beta[i];
                    beta[i]=min(max((beta[i]-Gb/qii[i]),0.0),c1);
                   	//cout<<beta[i]<<endl;
                    delb=beta[i]-betaold;
                    if(fabs(delb)>=1e-4)
                        changedvariable++;
                }

                //updating w
                if(fa||fb){
                    for(li j=0;j<tr.ind[i].size();j++)
                        w[tr.ind[i][j]]+=(tr.y[i]*(delb-dela)*xi[j]);
                    w0+=(delb-dela)*tr.y[i];
           //          for(li i=0;i<tr.n;i++){
			        //     cout<<w[i]<<" ";
			        // }
			        // cout<<w0<<endl;
                }

            }
            itr++;
            if(itr==MAXITR || changedvariable==0){
                cout<<"itr="<<itr<<"\t"<<"changedvariable="<<changedvariable<<"\n\n\n";
                converge=true;
            }
        }
        nsv=0;
        double sumalpha=0;
        for(li i=0;i<tr.m;i++){
            sumalpha+=alpha[i];
            if((beta[i]-alpha[i])!=0)
                nsv++;
        }

        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        
        cout<<"sum of alphas="<<sumalpha<<endl<<"nummber of support vector="<<nsv<<endl; 
        cout<<"lambda="<<lambda<<endl; 
    }

    void swap(vector<li> &v,li i,li j){
    	li t=v[i];
    	v[i]=v[j];
    	v[j]=t;
    }

    void trainWithShrinking(Data tr){
    	li m=tr.m;
    	li active=m;
 		li i,n,w_size=tr.n;

    	int itr=0;

    	double *alpha = new double[m];
    	double *beta = new double[m];
    	double *qii = new double[m];

    	bool converge=false;
    	clock_t begin, end;
		double time_spent;

		double maxgaold=inf,maxgbold=inf,mingaold=-inf,mingbold=-inf;
 		double maxga,maxgb,minga,mingb;
 		double pga,pgb;

    	double d;
    	vl seq=initialzation(tr,alpha,beta,qii,d);
    
    	begin = clock();
		//double inf = std::numeric_limits<double>::infinity();
 		
        double dela,delb,Ga,Gb;
        double betaold,alphaold;
        
        while(itr < MAXITR){

 			maxga=-inf,maxgb=-inf,minga=inf,mingb=inf;
 			// random_shuffle(seq.begin(), seq.end());

 			for(int k=0;k<active;k++){
 				i=seq[k];
 				int yi=tr.y[i];

 				n=tr.ind[i].size();

                double xi[n];
				copy(tr.x[i].begin(), tr.x[i].end(), xi);
                li index[n];
				copy(tr.ind[i].begin(), tr.ind[i].end(), index);
				

				Gb=w0;
 				for(li j=0;j<n;j++)
                    Gb += (xi[j]*w[index[j]]);
                Gb = yi*Gb - 1 + d*beta[i];

                Ga = lambda - Gb - 1 + d*(beta[i]+alpha[i]);
                
                // cout<<Ga<<" "<<Gb<<endl;
                pga=0,pgb=0;
                delb=0,dela=0;
                
                if(alpha[i]==0){
                	
                    if(Ga > maxgaold){
                    	active--;
                    	swap(seq,k,active);
                    	// cout<<i<<" removed from alpha\n";
                    	k--;
                    	continue;
                    }
					else if(Ga < 0)
                    	pga=Ga;
                } 
                else if(alpha[i]==c2){
                	
                	if(Ga < mingaold){
                		active--;
                    	swap(seq,k,active);
                    	// cout<<i<<" removed from alpha\n";
                    	k--;
                    	continue;
                	}
                	else if(Ga > 0)
                		pga=Ga;
                	
                }  
                else
                    pga = Ga;

                if(beta[i]==0){
                	
                	if(Gb > maxgbold){
                		active--;
                    	swap(seq,k,active);
                    	// cout<<i<<" removed from beta\n";
                    	k--;
                    	continue;
                	}
                	else if(Gb < 0)
                		pgb=Gb;
                } 
                else if(beta[i]==c1){
                	
                    if(Gb < mingbold){
                    	active--;
                    	swap(seq,k,active);
                    	// cout<<i<<" removed from beta\n";
                    	k--;
                    	continue;
                    }
                    else if(Gb > 0)
                    	pgb=Gb;
                } 
                else
                    pgb=Gb;

                if(fabs(pga) > 1e-10){
                    alphaold = alpha[i];
                    alpha[i] = min(max((alpha[i] - Ga/qii[i]),0.0),c2);
                    dela = yi*(alpha[i]-alphaold);
                }

                if(fabs(pgb) > 1e-10){
                    betaold = beta[i];
                    beta[i] = min(max((beta[i]-Gb/qii[i]),0.0),c1);
                    delb = yi*(beta[i]-betaold);
                }

                if(fabs(pga) > 1e-10 || fabs(pgb) > 1e-10){

                    for(li j=0;j<n;j++){
                        w[index[j]]+=((delb-dela)*xi[j]);
                        // cout<<w[index[j]]<<" ";
                    }
                    w0+=(delb-dela);
                }
                
                maxga=max(pga,maxga);
                maxgb=max(pgb,maxgb);

                minga=min(pga,minga);
                mingb=min(pgb,mingb);

            }

            itr++;
            if(itr % 10 == 0){
            	cout<<".";
            // 	cout<<endl;
            // for(int j=0;j<15;j++){
            //     // Gb=tr.y[j]*gbeta1[j] - 1 + gbeta2[j];
            //     // Ga=lambda - Gb - 1 + d*(beta[j]+alpha[j]);
            //     cout<<alpha[j]<<" "<<beta[j]<<endl;
            // }
            // int tyui;
            // cin>>tyui;
            }
            fflush(stdout);
            // cout<<",";
            // cout<<endl;
            // cout<<itr<<endl;
            // cout<<EPS;
 			if((maxga-minga <= EPS && maxgb-mingb <= EPS)){
 				if(active==m){
 					cout<<active<<" "<<itr<<endl;
 					break;
 				}
 				else{
 					cout<<"*";
 					maxgaold=inf,maxgbold=inf;
 					mingaold=-inf,mingbold=-inf;
 					active=m;
 					continue;
 				}

 			}

 			if(maxga<=0)
 				maxgaold=inf;
 			else
 				maxgaold=maxga;

 			if(maxgb<=0)
 				maxgbold=inf;
 			else
 				maxgbold=maxgb;

 			if(minga>=0)
 				mingaold=-inf;
 			else
 				mingaold=minga;

 			if(mingb>=0)
 				mingbold=-inf;
 			else
 				mingbold=mingb;
 			
 		}
 		if(itr>=MAXITR)
 			cout<<"max iteration reached\n";
		end = clock();
		nsv=0;
		double sumalpha=0;
		ofstream ofs("dcd.out");
        for(li i=0;i<m;i++){
        	sumalpha+=alpha[i];
        	if(beta[i]-alpha[i]!=0){
        		nsv++;
        		ofs<<beta[i]<<" "<<alpha[i]<<endl;
        	}
        }
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
		cout<<nsv<<endl<<endl;
        cout<<"sum of alphas="<<sumalpha<<endl;  
        delete [] qii;
		delete [] alpha;
		delete [] beta;
        seq.clear();
    }

    double kernel(vd xi,double* xj,vl indi, long* indj,li ni,li nj){
    	double kij=1.0;
    	li i=0,j=0;
    	while(i!=ni&&j!=nj){
    		if(indi[i]==indj[j]){
    			kij+=xi[i]*xj[j];
    			i++;
    			j++;
    		}
    		else if(indi[i]<indj[j])
    			i++;
    		else
    			j++;
    	}
    	return kij;
    }

    double kernel(double* xi,double* xj,long* indi, long* indj,li ni,li nj){
    	double kij=1.0;
    	li i=0,j=0;
    	while(i!=ni&&j!=nj){
    		if(indi[i]==indj[j]){
    			kij+=xi[i]*xj[j];
    			i++;
    			j++;
    		}
    		else if(indi[i]<indj[j])
    			i++;
    		else
    			j++;
    	}
    	return kij;
    }

    double kernel(vd xi,vd xj,vl indi,vl indj,li ni,li nj){
    	double kij=1.0;
    	li i=0,j=0;
    	while(i!=ni&&j!=nj){
    		if(indi[i]==indj[j]){
    			kij+=xi[i]*xj[j];
    			i++;
    			j++;
    		}
    		else if(indi[i]<indj[j])
    			i++;
    		else
    			j++;
    	}
    	return kij;
    }

    void trainWithShrinkingKernel(Data tr){
		li m=tr.m,n;
    	double *alpha = new double[m];
    	double *beta = new double[m];
    	double *qii = new double[m];

    	lambda=1.5f;
        c1=1.0;
        c2=1.0;
        double d;
    	if(type==L2LOSS)
    		d=0;
    	else
    		d=0.5/c1;
        
        vl seq;
        
        double *gbeta1=new double[m];
        double *gbeta2=new double[m];

        double dela,delb,Ga,Gb;
        double betaold,alphaold;
        int itr=0;
        li active=m,i;

        double maxgaold=inf,maxgbold=inf,mingaold=-inf,mingbold=-inf;
        double pga,pgb;
        double maxga,maxgb,minga,mingb;

        clock_t begin, end;
		double time_spent;

		begin = clock();

        for (li i=0;i<m;i++){
            double Q=0.0;
            beta[i]=0.0;
            alpha[i]=1.0/m;
            gbeta1[i]=0;
            gbeta2[i]=0;
            n=tr.ind[i].size();
            double ai=alpha[0];
            double xi[n];
			copy(tr.x[i].begin(), tr.x[i].end(), xi);
            li index[n];
			copy(tr.ind[i].begin(), tr.ind[i].end(), index);
			int yi=tr.y[i];
			Q=kernel(xi,xi,index,index,n,n);
            for(li j=0;j<m;j++)
            	gbeta1[i] += (kernel(tr.x[j],xi,tr.ind[j],index,tr.ind[j].size(),n)*tr.y[j]);
            gbeta1[i] = gbeta1[i]*-ai;
            qii[i]=(Q+d);
            seq.push_back(i);
        }

        cout<<"initialzation done\n";

        while(itr < MAXITR){
 			
 			maxga=-inf,maxgb=-inf,minga=inf,mingb=inf;
 			// random_shuffle(seq.begin(), seq.end());

 			for(int k=0;k<active;k++){
 				i=seq[k];
 				
 				double alphai=alpha[i],betai=beta[i];

 				// if(fabs(betai-alphai)<1e-7){
 				// 	seq.erase(seq.begin()+k);
     //            	cout<<i<<"beta - alpha differance too low\n";
     //            	active--;
     //            	continue;
 				// }


 				n=tr.ind[i].size();

                double xi[n];
				copy(tr.x[i].begin(), tr.x[i].end(), xi);
                li index[n];
				copy(tr.ind[i].begin(), tr.ind[i].end(), index);
				int yi=tr.y[i];
				
			    Gb = yi*gbeta1[i] - 1 + gbeta2[i];

                Ga = lambda - Gb - 1 + d*(betai+alphai);
                
                // cout<<Ga<<" "<<Gb;

                pga=0,pgb=0;
                delb=0,dela=0;
                betaold=betai,alphaold=alphai;

                if(alphai==0){
                	
                    if(Ga > maxgaold){
                    	active--;
                    	swap(seq,active,k);
                    	cout<<i<<" removed from alpha\n";
                    	k--;
                    	continue;
                    }
                    else if(Ga < 0)
                    	pga=Ga;
                } 
                else if(alphai==c2){
                	
                	if(Ga < mingaold){
                		active--;
                    	swap(seq,active,k);
                    	cout<<i<<" removed from alpha\n";
                    	k--;
                    	continue;
                	}
                	else if(Ga > 0)
                		pga=Ga;
                }  
                else
                    pga=Ga;

                if(betai==0){
                	
                	if(Gb > maxgbold){
                		active--;
                    	swap(seq,active,k);
                    	cout<<i<<" removed from beta\n";
                    	k--;
                    	continue;;
                	}
                	else if(Gb < 0)
                		pgb=Gb;
                } 
                else if(betai==c1){
                	
                    if(Gb < mingbold){
                    	active--;
                    	swap(seq,active,k);
                    	cout<<i<<" removed from beta\n";
                    	k--;
                    	continue;
                    }
                    else if(Gb > 0)
                    	pgb=Gb;
                } 
                else
                    pgb=Gb;

                if(fabs(pga) > 1e-10){
                	// cout<<alpha[i]-Ga/qii[i]<<endl;
                	// cout<<" A-"<<alpha[i]<<" ";
                	alphaold=alphai;
                    alpha[i]=min(max((alphai-Ga/qii[i]),0.0),c2);
                    dela=alpha[i]-alphaold;
                    // cout<<"a-"<<alpha[i]<<" ";
                }

                if(fabs(pgb) > 1e-10){
                	// cout<<beta[i]-Gb/qii[i]<<endl;
                	betaold=betai;
                    beta[i]=min(max((betai-Gb/qii[i]),0.0),c1);
                    delb=beta[i]-betaold;
                    // cout<<"b-"<<beta[i]<<endl;
                }

                if(fabs(pga) > 1e-10 || fabs(pgb) > 1e-10){
                	// cout<<beta[i]-Gb/qii[i]<<endl;
                	for(li j=0;j<active;j++){
                		gbeta1[j]+= (kernel(tr.x[j],xi,tr.ind[j],index,tr.ind[j].size(),n)*(delb-dela)*tr.y[j]);
                		gbeta2[j] = d*beta[j];
                	}
                }

				// cout<<endl;
                maxga=max(pga,maxga);
                maxgb=max(pgb,maxgb);

                minga=min(pga,minga);
                mingb=min(pgb,mingb);

            }

            itr++;
            if(itr % 10 == 0)
            	cout<<".";
            fflush(stdout);
            // cout<<",";
            // cout<<endl;
            cout<<itr<<endl;
            // cout<<EPS;
 			if((maxga-minga <= EPS && maxgb-mingb <= EPS)){
 				if(active==m){
 					cout<<active<<" "<<itr<<endl;
 					break;
 				}
 				else{
 					cout<<"*";
 					maxgaold=inf,maxgbold=inf;
 					mingaold=-inf,mingbold=-inf;
 					active=m;
 					continue;
 				}

 			}
 			if(maxga<=0)
 				maxgaold=inf;
 			else
 				maxgaold=maxga;

 			if(maxgb<=0)
 				maxgbold=inf;
 			else
 				maxgbold=maxgb;

 			if(minga>=0)
 				mingaold=-inf;
 			else
 				mingaold=minga;

 			if(mingb>=0)
 				mingbold=-inf;
 			else
 				mingbold=mingb;
 			
 		}
 		cout<<"max iteration reached\n";
		end = clock();
		nsv=0;
		double sumalpha=0;
        for(li i=0;i<m;i++){
        	sumalpha+=alpha[i];
        	if(beta[i]-alpha[i]!=0)
        		nsv++;
        }
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
		
        cout<<"sum of alphas="<<sumalpha<<endl;
        Beta.assign(beta,beta+m);
        Alpha.assign(alpha,alpha+m); 
        Qii.assign(qii,qii+m);
        delete [] qii;
		delete [] alpha;
		delete [] beta;
		delete[] gbeta1;
		delete[] gbeta2;
    }

    void predictionWithKernel(Data tt,Data tr){
    	
    	li correct=0;

    	for(li i=0;i<tt.m;i++){
    		double d=0;
    		for(li j=0;j<tr.m;j++){
    			d+=(kernel(tt.x[i],tr.x[j],tt.ind[i],tr.ind[j],tt.ind[i].size(),tr.ind[j].size())*tr.y[j]*(Beta[j]-Alpha[j]));
    		}
    		if((d>=0&&tt.y[i]==1)||(d<0&&tt.y[i]==-1))
    			correct++;
    		cout<<correct<<endl;
    	}
    	
    	double accuracy=correct*100.0/tt.m;

    	cout<<"accuracy="<<accuracy<<"("<<correct<<"/"<<tt.m<<")"<<endl;


    	/*
margin=1/norm(w1)
%figure
%end*/
    }

    void printModel(li m,li n){
    	cout<<"nummber of support vector : "<<nsv<<endl;
        // cout<<"alpha\tbeta\tqii\n";
        // for(li i=0;i<m;i++){
        //     cout<<Alpha[i]<<"\t"<<Beta[i]<<"\t"<<Qii[i]<<endl;
        // }
       
        cout<<"w : ";
        double margin=0;
        for(li i=0;i<n;i++){
            cout<<w[i]<<" ";
            margin+=w[i]*w[i];
        }
        //cout<<w0<<endl;
        margin=1/sqrt(margin);
        cout<<"margin = "<<margin<<endl;
    }

    void prediction(Data tt){
    	long n1 = w.size();

    	long n=min(n1,tt.n);
    	cout<<n1<<" "<<tt.n<<endl;
    	//vector<int> pred;
    	li correct=0;

    	for(li i=0;i<tt.m;i++){
    		double d=0;
    		li maxelem=tt.x[i].size();
    		for(li j=0;j<min(n,maxelem);j++){
    			d+=tt.x[i][j]*w[tt.ind[i][j]];
    		}
    		// cout<<i<<" "<<d;
    		// fflush(stdout);
    		if(n1==maxelem){
    			d+=w0;
    		}
    		else if(n1<maxelem){
    			//if the ind[n1] exist and is n1 then it is non-zero so mutiply x[ind[n1]] with w0
    			if(tt.ind[i][n1]+1==n1)
    				d+=w0*tt.x[i][n1];
    		}
    		else{
    			//n1>nt
    			d+=w[maxelem];
    		}
    		// cout<<" again "<<d<<endl;
    		// fflush(stdout);
    		//if pred is 1 and the yt =1 then it is correct;
    		if((d>=0&&tt.y[i]==1)||(d<0&&tt.y[i]==-1))
    			correct++;
            // else
            //     cout<<i<<" "<<((d>=0)?1:-1)<<" "<<tt.y[i]<<endl;
    	}
    	
    	double accuracy=correct*100.0/tt.m;

    	cout<<"accuracy="<<accuracy<<"("<<correct<<"/"<<tt.m<<")"<<endl;


    	/*
margin=1/norm(w1)
%figure
%end*/
    }
};



int main(){
    Data *tr=new Data();
    Data *tt=new Data();
    // string trainfile="../../Data_ML/news20b_sparse_1.train";
    // string testfile="../../Data_ML/news20b_sparse_1.test";
    
   //  string trainfile="sparsedata1.train";//"kddb_unnorm_svm_1.train"
   // string testfile="sparsedata1.test";//"kddb_unnorm_svm_1.test";
    // string trainfile="../../Data_ML/kddb_unnorm_svm_1.train";
   	// string testfile="../../Data_ML/kddb_unnorm_svm_1.test";
    string trainfile="../../Data_ML/real-sim_sparse_1.train";
    string testfile="../../Data_ML/real-sim_sparse_1.test";
    

	tr->readinput(trainfile);
    //tr->printData();
    // TrainingModel *model=new TrainingModel(*tr);
    int option=1;
    cout<<"choose option\n";
    cout<<" 1 -- L2-regularized L1-loss MCM classification\n";
	cout<<" 2 -- L2-regularized L2-loss MCM classification\n";
	// cin>>option;

    TrainingModel *model=new TrainingModel();
    model->type=option;
    // model->trainWithShrinkingKernel(*tr);
    
    tt->readinput(testfile);
    // model->predictionWithKernel(*tt,*tr);
    // model->train(*tr);
    model->trainWithShrinking(*tr);
    // model->printModel(tr->m,tr->n);
    // // model->printModel(tr->m,tr->n);
    model->prediction(*tt);
    // //cout<<tr->m<<" "<<tr->n<<endl<<tt->m<<" "<<tt->n<<endl;
    //tt->printData();
	//cout<<"reading complete\n";
    delete tr;
    delete tt;
    delete model;
	return 0;
}