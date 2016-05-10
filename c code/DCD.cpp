#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>

#define MAXITR 400
#define EPS 1e-4

using namespace std;

typedef long int li;
typedef vector< vector<float> > vvf;
typedef vector< vector<li> > vvl;
typedef vector<float> vf;
typedef vector<li> vl;

class Data{
public:
    vvf x;
    vvl ind;
    vector<int> y;
    li m,n,x0;

    Data(){
        x0=1;
    }

    void readinput(string file){
        ifstream input(file.c_str());
        string line,token,a; 
        stringstream iss,ss;
        string::size_type sz;
        vf tx;
        vl ti;
        m=0;
        n=0;
        while ( getline(input, line) )
        {   
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
                ss.clear();
            }
            iss.clear();
            x.push_back(tx);
            ind.push_back(ti);
            ti.clear();
            tx.clear();
            if (count > n)
                n=count;
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
    bool converge;
    float lambda,c1,c2,w0;
    vf alpha,alphaold,beta,betaold,w,Ga,Gb,qii;
    int nsv;

    vl initialzation(Data tr){
        alpha=vf(tr.m,1.0/tr.m);
        alphaold=alpha;
        beta=vf(tr.m,0);
        betaold=beta;
        w=vf(tr.n,0);
        w0=0;
        converge=false;
        vl seq;
        // qii is sum of sqare of each data point(sum of sqaure of each feature)
        // qii=sum(x(i,:).^2,2);
        // w=(y.*(beta-alpha))'*x;
        for (li i=0;i<tr.m;i++){
            float Q=1.0f;
            for(li j=0;j<tr.ind[i].size();j++){
                w[tr.ind[i][j]]+=(tr.y[i]*(beta[i]-alpha[i])*tr.x[i][j]);
                Q+=(tr.x[i][j]*tr.x[i][j]);
            }
            w0+=(tr.y[i]*(beta[i]-alpha[i]));
            qii.push_back(Q);
            seq.push_back(i);

        }
        //printModel(tr.m,tr.n);
        // int u;
        // cin>>u;
        lambda=1.0f;
        c1=1.0f;
        c2=1.0f;
        cout<<"initialzation done\n";
        return seq;
    }

    void train(Data tr){
    	vl seq=initialzation(tr);
    	clock_t begin, end;
		double time_spent;

		begin = clock();
        li changedvariable=0;
        float dela,delb;
        int itr=0;
        while(!converge){
            changedvariable=0;
            random_shuffle(seq.begin(), seq.end());
            for(li k=0;k<tr.m;k++){
            	li i=seq[k];
                float Gb=w0;
                dela=0,delb=0;
                for(li j=0;j<tr.ind[i].size();j++)
                    Gb+=(tr.x[i][j]*w[tr.ind[i][j]]);
                Gb=tr.y[i]*Gb-1;
                float Ga=lambda-Gb-1;
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
                    alphaold[i]=alpha[i];
                    alpha[i]=min(max((alpha[i]-Ga/qii[i]),0.0f),c2);
                    //cout<<alpha[i]<<endl;
                    dela=alpha[i]-alphaold[i];
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
                    betaold[i]=beta[i];
                    beta[i]=min(max((beta[i]-Gb/qii[i]),0.0f),c1);
                   	//cout<<beta[i]<<endl;
                    delb=beta[i]-betaold[i];
                    if(fabs(delb)>=1e-4)
                        changedvariable++;
                }

                //updating w
                if(fa||fb){
                    for(li j=0;j<tr.ind[i].size();j++)
                        w[tr.ind[i][j]]+=(tr.y[i]*(delb-dela)*tr.x[i][j]);
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
        for(li i=0;i<tr.m;i++){
        	if(beta[i]-alpha[i]!=0)
        		nsv++;
        }
        end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
    }

    void trainWithShrinking(Data tr){
    	vl seq=initialzation(tr);
    	clock_t begin, end;
		double time_spent;

		begin = clock();
		float inf = std::numeric_limits<float>::infinity();
 		float maxga1=inf,maxgb1=inf,minga1=-inf,mingb1=-inf;
 		vl A=seq;
 		vl B=seq;
 		li activea=tr.m,activeb=tr.m;
 		li changedvariable=0;
        float dela,delb;
        int itr=0;
        while(!converge){
            changedvariable=0;
 			float maxga=inf,maxgb=inf,minga=-inf,mingb=-inf;
 			for(int k=0;k<max(activeb,activea);k++){
 				li i;
 				if(A[k]<B[k])
 					i=A[k];
 				else
 					i=B[k];

 				float Gb=w0;
 				for(li j=0;j<tr.ind[i].size();j++)
                    Gb+=(tr.x[i][j]*w[tr.ind[i][j]]);
                Gb=tr.y[i]*Gb-1;
                float Ga=lambda-Gb-1;
                float pga=0,pgb=0;
                // updating alpha
                if(fabs(Ga)>1e-4)
                    pga=Ga;
                else if(alpha[i]<=1e-5){
                	if(Ga < 0)
                    	pga=Ga;
                    if(Ga > maxga1){
                    	A.erase(A.begin()+k);
                    	activea--;
                    }

                } 
                else if(alpha[i]>=c2-1e-5){
                	if(Ga > 0)
                		pga=Ga;
                	if(Ga < minga1){
                		A.erase(A.begin()+k);
                		activea--;
                	}
                	
                }  
                else
                    pga=Ga;
                if(pga!=0){
                    alphaold[i]=alpha[i];
                    alpha[i]=min(max((alpha[i]-Ga/qii[i]),0.0f),c2);
                    //cout<<alpha[i]<<endl;
                    dela=alpha[i]-alphaold[i];
                    if(fabs(dela)>=1e-4)
                        changedvariable++;
                }

                // updating beta
                if(fabs(Gb)>1e-4)
                    pgb=Gb;
                else if(beta[i]<=1e-5){
                	if(Gb < 0)
                		pgb=Gb;
                	if(Gb > maxgb1){
                		B.erase(B.begin()+k);
                		activeb--;
                	}
                } 
                else if(beta[i]>=c1-1e-5){
                	if(Gb > 0)
                    	pgb=Gb;
                    if(Gb < mingb1){
                    	B.erase(B.begin()+k);
                		activeb--;
                    }
                } 
                else
                    pgb=Gb;
                if(pgb!=0){
                    betaold[i]=beta[i];
                    beta[i]=min(max((beta[i]-Gb/qii[i]),0.0f),c1);
                   	//cout<<beta[i]<<endl;
                    delb=beta[i]-betaold[i];
                    if(fabs(delb)>=1e-4)
                        changedvariable++;
                }

                //updating w
                if(pga!=0||pgb!=0){
                    for(li j=0;j<tr.ind[i].size();j++)
                        w[tr.ind[i][j]]+=(tr.y[i]*(delb-dela)*tr.x[i][j]);
                    w0+=(delb-dela)*tr.y[i];
           //          for(li i=0;i<tr.n;i++){
			        //     cout<<w[i]<<" ";
			        // }
			        // cout<<w0<<endl;
                }
                maxga=max(pga,maxga);
                maxgb=max(pgb,maxgb);
                minga=min(pga,minga);
                mingb=min(pgb,mingb);

            }
            itr++;
            cout<<itr;
 			if(maxga-minga < EPS && maxgb-mingb < EPS || itr==MAXITR){
 				if(activea==tr.m&&activeb==tr.m||itr==MAXITR)
 					converge=true;
 				else{
 					A=seq;
 					B=seq;
 					maxga=inf,maxgb=inf,minga=-inf,mingb=-inf;
 					activeb=activea=tr.m;
 				}

 			}
 			if(maxga1<=0)
 				maxga1=inf;
 			else
 				maxga1=maxga;
 			if(maxgb1<=0)
 				maxgb1=inf;
 			else
 				maxgb1=maxgb;
 			if(minga1>=0)
 				minga1=-inf;
 			else
 				minga1=minga;
 			if(mingb>=0)
 				mingb1=-inf;
 			else
 				mingb1=mingb;
 			
 		}
		end = clock();
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
    }

    void printModel(li m,li n){
    	cout<<"nummber of support vector : "<<nsv<<endl;
        //cout<<"alpha\tbeta\tqii\n";
        float sumalpha=0;
        for (li i = 0; i < m; i++){
            //cout<<alpha[i]<<"\t"<<beta[i]<<"\t"<<qii[i]<<endl;
            sumalpha+=alpha[i];
        }
        cout<<"sum of alphas="<<sumalpha<<"\nw : ";
        float margin=0;
        for(li i=0;i<n;i++){
            //cout<<w[i]<<" ";
            margin+=w[i]*w[i];
        }
        //cout<<w0<<endl;
        margin=1/sqrt(margin);
        cout<<"margin = "<<margin<<endl;
    }

    void prediction(Data tt){
    	long n1 = w.size();

    	long n=min(n1,tt.n);

    	//vector<int> pred;
    	li correct=0;

    	for(li i=0;i<tt.m;i++){
    		float d=0;
    		for(li j=0;j<n;j++){
    			d+=tt.x[i][j]*w[tt.ind[i][j]];
    		}
    		if(n1==tt.n){
    			d+=w0;
    		}
    		else if(n1<tt.n){
    			//if the ind[n1] exist and is n1 then it is non-zero so mutiply x[ind[n1]] with w0
    			if(tt.ind[i][n1]+1==n1)
    				d+=w0*tt.x[i][n1];
    		}
    		else{
    			//n1>nt
    			d+=w[tt.n];
    		}

    		//if pred is 1 and the yt =1 then it is correct;
    		if((d>=0&&tt.y[i]==1)||(d<0&&tt.y[i]==-1))
    			correct++;
    	}
    	
    	float accuracy=correct*100.0/tt.m;

    	cout<<"accuracy="<<accuracy<<endl;


    	/*
margin=1/norm(w1)
%figure
%end*/
    }
};



int main(){
    Data *tr=new Data();
    Data *tt=new Data();
    //string trainfile="../../Data_ML/mnist38_norm_svm_full_1.train";//
    string trainfile="sparsedata1.train";//"kddb_unnorm_svm_1.train";
   //string testfile="../../Data_ML/mnist38_norm_svm_full_1.test";//
   string testfile="sparsedata1.test";//"kddb_unnorm_svm_1.test";
	tr->readinput(trainfile);
    //tr->printData();
    // TrainingModel *model=new TrainingModel(*tr);
    TrainingModel *model=new TrainingModel();
    model->train(*tr);
    model->printModel(tr->m,tr->n);
    tt->readinput(testfile);
    model->prediction(*tt);
    //cout<<tr->m<<" "<<tr->n<<endl<<tt->m<<" "<<tt->n<<endl;
    //tt->printData();
	//cout<<"reading complete\n";
    delete tr;
    delete tt;
    delete model;
	return 0;
}