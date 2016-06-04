#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cstring>

#define MAXITR 400
#define inf HUGE_VAL
#define EPS 0.05
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

enum ModelType{L1LOSS=1,L2LOSS};
using namespace std;

typedef long int li;
typedef vector< vector<float> > vvd;
typedef vector< vector<li> > vvl;
typedef vector<float> vd;
typedef vector<li> vl;

class Data{
public:
    float** x;
    long** ind;
    li* i_space;
    float* x_space;
    int* y;
    li m,n,x0,elements;

    Data(){
        x0=1;
    }

    void readinput(string file){
        int max_index, inst_max_index, i;
        size_t elements, j;
        ifstream fp;
        fp.open(file.c_str());
        char *endptr;
        char *idx, *val, *label;

        if(fp == NULL){
            cout<<"file not found\n";
            exit(1);
        }

        m = 0;
        elements = 0;
        string line,token,a;
        string::size_type sz;
        bool isSparse=true;
        int count=0,count1=0;
        while(getline(fp,line))
        {
            char *p = strtok(line.c_str()," \t"); // label
            if(m == 0 && line.find(":")==-1){
                //cout<<"isSparse becoming false"<<line.find(":");
                isSparse=false;
            }
            // features
            count=0;
            while(1)
            {
                p = strtok(NULL," \t");
                if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                    break;
                if(!isSparse){
                    if(m==0)
                        count1++;
                    else{
                        count++;
                        if(count1 < count){
                            cout<<"unknown format\n";
                            exit(0);
                        }
                    }
                
                }
                
                elements++;
            }
            elements++; // for bias term
            m++;
            
        }
        fp.close();
        fp.open(file.c_str());
        y = Malloc(int,m);
        x = Malloc(float *,m);
        ind = Malloc(li *,m);
        x_space = Malloc(float,elements+m);
        i_space = Malloc(li,elements+m);
        cout<<"space allocated\n"<<elements+m<<endl;
        max_index = 0;
        j=0;
        stringstream iss,ss;
        if(isSparse){
            for(i=0;i<m;i++){
       //         cout<<i<<" "<<j<<endl;
                inst_max_index = 0; // strtol gives 0 if wrong format
                getline(fp,line);
                x[i] = &x_space[j];
                ind[i] = &i_space[j];
                
                /*while(1)
                {
                    idx = strtok(NULL,":");
                    val = strtok(NULL," \t");

                    if(val == NULL)
                        break;

                    errno = 0;
                    i_space[j] = (int) strtol(idx,&endptr,10);
                    inst_max_index = i_space[j]>inst_max_index?i_space[j]:inst_max_index;

                    x_space[j] = strtod(val,&endptr);

                    ++j;
                }*/
                iss << line;
                int ty;
                iss>>ty;
                y[i]= ty>=1?1:-1;
                // cout<<y[i];
                while(iss>>token){
                    // cout<<token<<"-";
                    fflush(stdout);
                    ss<<token;
                    getline(ss, a, ':');
                    // cout<<a<<":";
                    i_space[j]=stol(a,&sz)-1;
                    getline(ss, a, ':');
                    // cout<<a<<" ";
                    x_space[j]= stod(a,&sz);
                    inst_max_index = i_space[j]+1>inst_max_index?i_space[j]+1:inst_max_index;
                    ss.clear();
                    j++;
                }
                // cout<<" ]"<<endl;
                iss.clear();

                if(inst_max_index > max_index)
                    max_index = inst_max_index;

                i_space[j++]= -1;
                
            }
                n=max_index;
        }
        else{
            cout<<"complete data\n";
            for(i=0;i<m;i++){
   //         cout<<i<<" "<<j<<endl;
                inst_max_index = 0; // strtol gives 0 if wrong format
                getline(fp,line);
                x[i] = &x_space[j];
                ind[i] = &i_space[j];
                
                /*while(1)
                {
                    idx = strtok(NULL,":");
                    val = strtok(NULL," \t");

                    if(val == NULL)
                        break;

                    errno = 0;
                    i_space[j] = (int) strtol(idx,&endptr,10);
                    inst_max_index = i_space[j]>inst_max_index?i_space[j]:inst_max_index;

                    x_space[j] = strtod(val,&endptr);

                    ++j;
                }*/
                iss << line;
                int ty;
                iss>>ty;
                if(i==0)
                    y[i]=1;
                else
                    y[i]= ty==y[0]?1:-1;
                // cout<<y[i];
                count=0;
                while(iss>>token){
                    // cout<<token<<"-";
                    i_space[j]=count;
                    // cout<<token<<" ";
                    x_space[j]= stod(token,&sz);
                    count++;
                    ss.clear();
                    j++;
                }
                // cout<<" ]"<<endl;
                iss.clear();
                i_space[j++]= -1;
                
            }
                n=count1;
        }

        fp.close();
        cout<<"reading complete\n";
    }
   

    void printData(){
        cout<<m<<" "<<n<<endl;
        for(li i=0;i<m;i++){
            cout<<i+1<<": "<<y[i]<<" ";
            for(li j=0;ind[i][j]!=-1;j++){
                cout<<ind[i][j]+1<<":"<<x[i][j]<<" ";
            }
            cout<<endl;
        }
    }

    ~Data(){
        // for(int i = 0; i < m; i++){
        //     delete [] x[i];
        //     delete [] ind[i];
        // }
        // delete [] x;
        // delete [] ind;
        // delete [] y;
        cout<<"destructor called\n"<<m<<" "<<n<<endl;
        fflush(stdout);
        free(y);
        // cout<<"y\n";
        // fflush(stdout);
        free(i_space);
        // cout<<"i_space\n";
        // fflush(stdout);
        free(x_space);
        // cout<<"x_space\n";
        // fflush(stdout);
        free(ind);
        // cout<<"ind\n";
        // fflush(stdout);
        free(x);
        cout<<"data memory freed\n";
    }
};

class TrainingModel{
public:
    float lambda,c1,c2,w0;
    float* w;
    int nsv;
    li w_size,m;
    ModelType type;
    float* beta,*alpha;

    TrainingModel(li m,li n){
        this->w_size = n;
        this->w = new float[w_size];
        this->w0 = 0;
        this->alpha = new float[m];
        this->beta = new float[m];
        this->m=m;
        this->c1=1.0f;
        this->c2=1.0f;

    }

    ~TrainingModel(){
        cout<<"model destructor called\n";
        delete [] w;
        delete [] alpha;
        delete [] beta;
        cout<<"model memory freed\n";
    }


    li* initialzation(Data& tr,float* qii,float &d){
        // lambda=1.5f;
        if(type==L1LOSS)
            d=0;
        else
            d=0.5/c1;

        for(li i=0;i<w_size;i++)
            w[i]=0;

        li* seq= new li[m];  
        li n;
        // qii is sum of sqare of each data point(sum of sqaure of each feature)
        // qii=sum(x(i,:).^2,2);
        // w=(y.*(beta-alpha))'*x;
        for (li i=0;i<m;i++){
            float Q=1.0f;
            beta[i]=0;
            alpha[i]=1.0f/m;
            
            for(li j=0;tr.ind[i][j]!=-1;j++){
                float x=tr.x[i][j];
                li p=tr.ind[i][j];
                w[p]+=(tr.y[i]*(beta[i]-alpha[i])*x);
                
                Q+=(x*x);
                // cout<<x<<":"<<Q<<" ";
            }
            // cout<<endl;
            w0+=(tr.y[i]*(beta[i]-alpha[i]));
            qii[i]=(Q+d);
            // cout<<"d="<<d<<" qii[i]="<<qii[i]<<endl;
            seq[i]=i;

        }
        //printModel(tr.m,tr.n);
        // int u;
        // cin>>u;
        
        cout<<"initialzation done\n";
        return seq;
    }

    void train(Data &tr){
        float *qii = new float[m];
        bool converge=false;
        float d;
        
        clock_t begin, end;
        float time_spent;

        begin = clock();

        li* seq=initialzation(tr,qii,d);
        cout<<"alpha\tbeta\tqii\n";
            for(li i=0;i<m;i++){
                cout<<alpha[i]<<"\t"<<beta[i]<<"\t"<<qii[i]<<endl;
            }
           
            cout<<"w : ";
            float margin=0;
            for(li i=0;i<w_size;i++){
                cout<<w[i]<<" ";
                margin+=w[i]*w[i];
            }
            cout<<w0<<endl;
        int huio;
        cin>>huio;
        li changedvariable=0;
        float dela,delb,Ga,Gb;
        float alphaold,betaold;
        int itr=0;

        while(!converge){
            changedvariable=0;
            // random_shuffle(&seq[0], &seq[m]);
            for(li k=0;k<m;k++){
                li i=seq[k];
                int yi=tr.y[i];
                float Gb=w0;
                dela=0,delb=0;
                float *xi=tr.x[i];
                li *index=tr.ind[i];

                for(li j=0;index[j]!=-1;j++)
                    Gb+=(xi[j]*w[index[j]]);
                Gb=yi*Gb-1+beta[i]*d;
                float Ga=lambda-Gb-1+(beta[i]+alpha[i])*d;
                cout<<i+1 <<" Ga ="<<Ga<<" Gb="<<Gb<<" qii="<<qii[i]<<endl;
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
                    alpha[i]=min(max((alpha[i]-Ga/qii[i]),0.0f),c2);
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
                    beta[i]=min(max((beta[i]-Gb/qii[i]),0.0f),c1);
                    //cout<<beta[i]<<endl;
                    delb=beta[i]-betaold;
                    if(fabs(delb)>=1e-4)
                        changedvariable++;
                }

                //updating w
                if(fa||fb){
                    for(li j=0;index[j]!=-1;j++)
                        w[index[j]]+=(yi*(delb-dela)*xi[j]);
                    w0+=(delb-dela)*yi;
           //          for(li i=0;i<tr.n;i++){
                    //     cout<<w[i]<<" ";
                    // }
                    // cout<<w0<<endl;
                }

            }
            cout<<"alpha\tbeta\tqii\n";
            for(li i=0;i<m;i++){
                cout<<alpha[i]<<"\t"<<beta[i]<<"\t"<<qii[i]<<endl;
            }
           
            cout<<"w : ";
            float margin=0;
            for(li i=0;i<w_size;i++){
                cout<<w[i]<<" ";
                margin+=w[i]*w[i];
            }
            cout<<w0<<endl;
            // int hui;
            // cin>>hui;
            itr++;
            if(itr==MAXITR || changedvariable==0){
                cout<<"itr="<<itr<<"\t"<<"changedvariable="<<changedvariable<<"\n\n\n";
                converge=true;
            }
        }
        nsv=0;
        float sumalpha=0;
        for(li i=0;i<m;i++){
            sumalpha+=alpha[i];
            cout<<beta[i]-alpha[i]<<" ";
            if((beta[i]-alpha[i])!=0){
                nsv++;
            }
        }

        time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        
        cout<<"sum of alphas="<<sumalpha<<endl<<"nummber of support vector="<<nsv<<endl; 
        cout<<"lambda="<<lambda<<endl; 
    }

    void swap(li *v,li i,li j){
        li t=v[i];
        v[i]=v[j];
        v[j]=t;
    }

    void trainWithShrinking(Data& tr){
        
        li active=m;
        li i,n;

        int itr=0;

        float *qii = new float[m];

        bool converge=false;
        clock_t begin, end;
        float time_spent;

        float maxgaold=inf,maxgbold=inf,mingaold=-inf,mingbold=-inf;
        float maxga,maxgb,minga,mingb;
        float pga,pgb;

        float d;
        begin = clock();
        li* seq=initialzation(tr,qii,d);
        
        //float inf = std::numeric_limits<float>::infinity();
        
        float dela,delb,Ga,Gb;
        float betaold,alphaold;
        
        while(itr < MAXITR){

            maxga=-inf,maxgb=-inf,minga=inf,mingb=inf;
            // random_shuffle(&seq[0], &seq[m]);

            for(li k=0;k<active;k++){
                i=seq[k];
                int yi=tr.y[i];

                float *xi=tr.x[i];
                li *index=tr.ind[i];
                
                Gb=w0;
                for(li j=0;index[j]!=-1;j++)
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
                else if(alpha[i] == c2){
                    
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

                if(beta[i]== 0){
                    
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
                else if(beta[i] == c1){
                    
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
                    alpha[i] = min(max((alpha[i] - Ga/qii[i]),0.0f),c2);
                    dela = yi*(alpha[i]-alphaold);
                }

                if(fabs(pgb) > 1e-10){
                    betaold = beta[i];
                    beta[i] = min(max((beta[i]-Gb/qii[i]),0.0f),c1);
                    delb = yi*(beta[i]-betaold);
                }

                if(fabs(pga) > 1e-10 || fabs(pgb) > 1e-10){

                    for(li j=0;index[j]!=-1;j++){
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
            //      cout<<endl;
            // for(int j=0;j<15;j++){
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
            cout<<"\nmax iteration reached\n";
        end = clock();
        nsv=0;
        // ofstream ofs("dcd1.out");
        float sumalpha=0;
        for(li i=0;i<m;i++){
            sumalpha+=alpha[i];
            if(fabs(beta[i]-alpha[i])>= 1e-5){
                nsv++;
                // ofs<<beta[i]<<" "<<alpha[i]<<endl;
            }
        }

        time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        
        cout<<"sum of alphas="<<sumalpha<<endl<<"nummber of support vector="<<nsv<<endl; 
        cout<<"lambda="<<lambda<<endl; 
 
       delete [] qii;
       delete [] seq;
    }

    float kernel(vd xi,float* xj,vl indi, long* indj,li ni,li nj){
        float kij=1.0f;
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

    float kernel(float* xi,float* xj,long* indi, long* indj){
        float kij=1.0f;
        li i=0,j=0;
        while(indi[i]!=-1&&indj[j]!=-1){
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

    float kernel(vd xi,vd xj,vl indi,vl indj,li ni,li nj){
        float kij=1.0;
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

    void trainWithShrinkingKernel(Data &tr,float lambda){
        float *qii = new float[m];

        c1=1.0f;
        c2=1.0f;
        float d;
        if(type==L2LOSS)
            d=0;
        else
            d=0.5/c1;
        
        li* seq= new li[m];
        
        float *gbeta1=new float[m];
        float *gbeta2=new float[m];

        float dela,delb,Ga,Gb;
        float betaold,alphaold;

        int itr=0;
        li active=m,i;

        float maxgaold=inf,maxgbold=inf,mingaold=-inf,mingbold=-inf;
        float pga,pgb;
        float maxga,maxgb,minga,mingb;

        clock_t begin, end;
        float time_spent;

        begin = clock();

        for (li i=0;i<m;i++){
            float Q=0.0f;
            beta[i]=0.0f;
            alpha[i]=1.0f/m;
            gbeta1[i]=0;
            gbeta2[i]=0;

            float ai=alpha[0];
            float* xi=tr.x[i];
            li *index=tr.ind[i];
            int yi=tr.y[i];
            Q=kernel(xi,xi,index,index);
            for(li j=0;j<m;j++)
                gbeta1[i] += (kernel(tr.x[j],xi,tr.ind[j],index)*tr.y[j]);
            gbeta1[i] = gbeta1[i]*-ai;
            qii[i]=(Q+d);
            seq[i]=i;
        }

        cout<<"initialzation done\n";

        while(itr < MAXITR){
            
            maxga=-inf,maxgb=-inf,minga=inf,mingb=inf;
            // random_shuffle(seq.begin(), seq.end());

            for(li k=0;k<active;k++){
                i=seq[k];
                
                float alphai=alpha[i],betai=beta[i];

                // if(fabs(betai-alphai)<1e-7){
                //  seq.erase(seq.begin()+k);
     //             cout<<i<<"beta - alpha differance too low\n";
     //             active--;
     //             continue;
                // }

                float* xi=tr.x[i];
                li* index=tr.ind[i];
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
                    alpha[i]=min(max((alphai-Ga/qii[i]),0.0f),c2);
                    dela=alpha[i]-alphaold;
                    // cout<<"a-"<<alpha[i]<<" ";
                }

                if(fabs(pgb) > 1e-10){
                    // cout<<beta[i]-Gb/qii[i]<<endl;
                    betaold=betai;
                    beta[i]=min(max((betai-Gb/qii[i]),0.0f),c1);
                    delb=beta[i]-betaold;
                    // cout<<"b-"<<beta[i]<<endl;
                }

                if(fabs(pga) > 1e-10 || fabs(pgb) > 1e-10){
                    // cout<<beta[i]-Gb/qii[i]<<endl;
                    for(li j=0;j<active;j++){
                        gbeta1[j]+= (kernel(tr.x[j],xi,tr.ind[j],index)*(delb-dela)*tr.y[j]);
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
        float sumalpha=0;
        for(li i=0;i<m;i++){
            sumalpha+=alpha[i];
            if(beta[i]-alpha[i]!=0)
                nsv++;
        }
        time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        
        cout<<"sum of alphas="<<sumalpha<<" nsv="<<nsv<<endl;

        delete [] qii;
        delete [] gbeta1;
        delete [] gbeta2;
        delete [] seq;
    }

    void predictionWithKernel(Data &tt,Data &tr){
        
        li correct=0;
        for(li i=0;i<tt.m;i++){
            float d=0;
            for(li j=0;j<m;j++){
                d+=(kernel(tt.x[i],tr.x[j],tt.ind[i],tr.ind[j])*tr.y[j]*(beta[j]-alpha[j]));
            }
            if((d>=0&&tt.y[i]==1)||(d<0&&tt.y[i]==-1))
                correct++;
            // cout<<correct<<endl;
        }
        
        float accuracy=correct*100.0/tt.m;

        cout<<"accuracy="<<accuracy<<"("<<correct<<"/"<<tt.m<<")"<<endl;
    }

    void printModel(li m,li n){
        cout<<"nummber of support vector : "<<nsv<<endl;
        // cout<<"alpha\tbeta\tqii\n";
        // for(li i=0;i<m;i++){
        //     cout<<Alpha[i]<<"\t"<<Beta[i]<<"\t"<<Qii[i]<<endl;
        // }
       
        cout<<"w : ";
        float margin=0;
        for(li i=0;i<w_size;i++){
            cout<<w[i]<<" ";
            margin+=w[i]*w[i];
        }
        //cout<<w0<<endl;
        margin=1/sqrt(margin);
        cout<<"margin = "<<margin<<endl;
    }

    void prediction(Data &tt){
        long n1 = w_size;
        li pos=0,neg=0,fp=0,fn=0;
        long n=min(n1,tt.n);
        cout<<n1<<" "<<tt.n<<endl;
        //vector<int> pred;
        li correct=0;

        for(li i=0;i<tt.m;i++){
            float d=0;
            li maxelem=0;
            li j;
            for(j=0;tt.ind[i][j]<w_size&&tt.ind[i][j]!=-1;j++){
                d+=tt.x[i][j]*w[tt.ind[i][j]];
            }
            // cout<<i<<" "<<d;
            // fflush(stdout);
            if(tt.ind[i][j]==w_size){
                d+=w0*tt.x[i][j];
            }
            else if(tt.ind[i][j]==-1){
                if(w_size==j){
                    d+=w0;
                }
                else
                    d+=w[j];
            }
            // else if(n1<maxelem){
            //     //if the ind[n1] exist and is n1 then it is non-zero so mutiply x[ind[n1]] with w0
            //     if(tt.ind[i][n1]+1==n1)
            //         d+=w0*tt.x[i][n1];
            // }
            // else{
            //     //n1>nt
            //     d+=w[maxelem];
            // }
            // cout<<" again "<<d<<endl;
            // fflush(stdout);
            //if pred is 1 and the yt =1 then it is correct;
            if((d>=0&&tt.y[i]==1)){
                pos++;
                correct++;
            }
            else if(d<0&&tt.y[i]==-1){
                neg++;
                correct++;
            }
            else if(d<0&&tt.y[i]==1)
                fn++;
            else
                fp++;
        }
            // else
            //     cout<<i<<" "<<((d>=0)?1:-1)<<" "<<tt.y[i]<<endl;
        
        float accuracy=correct*100.0f/tt.m;

        cout<<"accuracy="<<accuracy<<"("<<correct<<"/"<<tt.m<<")"<<endl;

        // actual > pred v
        cout<<"confusion matrix\n";
        cout<<"\tpos\tneg\n";
        cout<<"pos\t"<<pos<<"\t"<<fp<<endl;
        cout<<"neg\t"<<fn<<"\t"<<neg<<endl;
    }
};


int main(int agrc,char* argv[]){
    Data tr = Data();
    Data tt = Data();
    string file="iono";
    float lambda=1;
    cout<<"enter file name : ";
    // cin>>file;
    
    string trainfile="../split/"+file+".train";
    string testfile="../split/"+file+".test";
    // string trainfile="../../Data_ML/"+file+".train";
    // string testfile="../../Data_ML/"+file+".test";
    // string trainfile="../../Data_ML/mnist38_norm_svm_full_1.train";
    // string testfile="../../Data_ML/mnist38_norm_svm_full_1.test";
    // string trainfile="sparsedata1.train";//"kddb_unnorm_svm_1.train"
    // string testfile="sparsedata1.test";//"kddb_unnorm_svm_1.test";
    // string trainfile="../../Data_ML/kddb_unnorm_svm_1.train";
    // string testfile="../../Data_ML/kddb_unnorm_svm_1.test";

    tr.readinput(trainfile);
    cout<<tr.m<<" "<<tr.n<<endl;
    cout<<"out\n";
    fflush(stdout);
    // tr.printData();
    // TrainingModel *model=new TrainingModel(*tr);
    int option=1;
    // cout<<"choose option\n";
    // cout<<" 1 -- L2-regularized L1-loss MCM classification\n";
    // cout<<" 2 -- L2-regularized L2-loss MCM classification\n";
    // cin>>option;
    tt.readinput(testfile);
    // tt.printData();
    cout<<tt.m<<" "<<tt.n<<endl;
    cout<<"out2\n";
    TrainingModel model=TrainingModel(tr.m,tr.n);
    model.type=option;
    while(1){
        cout<<"enter lambda : ";
    cin>>lambda;
    if(lambda==0)
        break;
    model.lambda=lambda;
     model.trainWithShrinking(tr);
    // model.train(tr);
        // model.printModel(tr.m,tr.n);
    //     // // // model->printModel(tr->m,tr->n);
    //     cout<<"test ";
        model.prediction(tt);
    }
    
    // model.trainWithShrinkingKernel(tr,lambda);
    
    // model.train(tr,lambda);
    
    // model.predictionWithKernel(tt,tr);
    // while(1){
        // cin>>lambda;
        // if(lambda < 0)
        //     break;
       
    //     cout<<"train ";
    //     model.prediction(tr);
    //     cout<<"returning\n";
    //     fflush(stdout);
    // // }
    
    // //cout<<tr->m<<" "<<tr->n<<endl<<tt->m<<" "<<tt->n<<endl;
    //tt->printData();
    //cout<<"reading complete\n";
    // delete tr;

    // delete tt;
    // delete model;
    return 0;
}