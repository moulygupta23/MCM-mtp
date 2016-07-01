#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cstring>

#define MAXITR 1000
#define inf HUGE_VAL
#define EPS 0.1
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

enum KernelType{LINEAR,GAUSSIAN};
enum ModelType{L1LOSS=1,L2LOSS};
using namespace std;
double thresh=0;
typedef int li;
typedef vector< vector<double> > vvd;
typedef vector< vector<li> > vvl;
typedef vector<double> vd;
typedef vector<li> vl;

string firstlabel;
class Data{
public:
    double** x;
    int** ind;
    li* i_space;
    double* x_space;
    int* y;
    li* perm;
    li start,end;
    li m,n,x0,elements;

    Data(){
        x0=1;
        start=0;
    }

    void readinput(string file,bool istrain){
        int max_index, inst_max_index, i;
        size_t j;
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
        x = Malloc(double *,m);
        ind = Malloc(li *,m);
        x_space = Malloc(double,elements+m);
        i_space = Malloc(li,elements+m);
        cout<<"space allocated\n"<<elements+m<<endl;
        max_index = 0;
        j=0;
        stringstream iss,ss;
        if(isSparse){
            for(i=0;i<m;i++){
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
                string ty;
                iss>>ty;
                // cout<<ty<<"  =  ";
                if(i==0 && istrain)
                    firstlabel=ty;
                y[i] = (strcmp(ty.c_str(),firstlabel.c_str())==0)?1:-1;
                // cout<<y[i]<<"\n\n";
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
                string ty;
                iss>>ty;
                if(i==0 && istrain)
                    firstlabel=ty;
                y[i] = (strcmp(ty.c_str(),firstlabel.c_str())==0)?1:-1;
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
            cout<<i+1<<": "<<y[perm[i]]<<" ";
            int k=0;
            for(li j=0;ind[perm[i]][j]!=-1&&k < 4;j++){
                cout<<ind[perm[i]][j]+1<<":"<<x[perm[i]][j]<<" ";
                k++;
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

    void copy(Data &data,bool in,vector<int> seq){
        if(!in)
            m = data.m + data.start - data.end;
        else
            m = data.end- data.start;
        // cout<<"m = "<<m<<endl;
        n=data.n;
        y = Malloc(int,m);
        x = Malloc(double *,m);
        ind = Malloc(li *,m);
        x_space = Malloc(double,data.elements+data.m);
        i_space = Malloc(li,data.elements+data.m);
        li j=0,k=0;
        if(in==true){
            for(li i=data.start;i<data.end;i++,k++){
                y[k]=data.y[seq[i]];
                x[k]=&x_space[j];
                ind[k]=&i_space[j];
                for(li p=0;data.ind[seq[i]][p]!=-1;p++){
                    x_space[j]=data.x[seq[i]][p];
                    i_space[j]=data.ind[seq[i]][p];
                    j++;
                }
                i_space[j]=-1;
                j++;
            }
        }
        else{
            for(li i=0;i<data.start;i++,k++){
                y[k]=data.y[seq[i]];
                x[k]=&x_space[j];
                ind[k]=&i_space[j];
                for(li p=0;data.ind[seq[i]][p]!=-1;p++){
                    x_space[j]=data.x[seq[i]][p];
                    i_space[j]=data.ind[seq[i]][p];
                    j++;
                }
                i_space[j]=-1;
                j++;
            }
            for(li i=data.end;i<data.m;i++,k++){
                y[k]=data.y[seq[i]];
                x[k]=&x_space[j];
                ind[k]=&i_space[j];
                for(li p=0;data.ind[seq[i]][p]!=-1;p++){
                    x_space[j]=data.x[seq[i]][p];
                    i_space[j]=data.ind[seq[i]][p];
                    j++;
                }
                i_space[j]=-1;
                j++;
            }
        }
        // for(li i=data.start;i<data.m;i++,k++){
        //     if(data.start < data.end && i>data.end){
        //         break;
        //     }
        //     y[k]=data.y[i];
        //     x[k]=&x_space[j];
        //     ind[k]=&i_space[j];
        //     for(li p=0;data.ind[i][p]!=-1;p++){
        //         x_space[j]=data.x[i][p];
        //         i_space[j]=data.ind[i][p];
        //         j++;
        //     }
        //     i_space[j]=-1;
        //     j++;
        // }
        // if(data.end < data.start){
        //     for(li i=0;i<=data.end;i++,k++){
        //         x[k]=&x_space[j];
        //         ind[k]=&i_space[j];
        //         for(li p=0;data.ind[i][p]!=-1;p++){
        //             x_space[j]=data.x[i][p];
        //             i_space[j]=data.ind[i][p];
        //             j++;
        //         }
        //         i_space[j]=-1;
        //         j++;
        //     }
        // }
        cout<<"data copied\n";
    }

    void sort(){
        vector<li> pos,neg;
        perm = new li[m];
        li j=0;
        for(li i=0;i<m;i++){
            if(y[i] == 1 )
                perm[j++]=i;
            else
                neg.push_back(i);
        }
        for(int i=0;i<neg.size();i++)
            perm[j++]=neg[i];
        neg.clear();
    }
};

class TrainingModel{
public:
    double lambda,c1,c2,w0;
    double* w;
    int nsv;
    li w_size,m;
    ModelType type;
    double* beta,*alpha;

    TrainingModel(li m,li n){
        this->m=m;
        this->w_size = n;
        this->w = new double[w_size];
        this->alpha = new double[m];
        this->beta = new double[m];
        this->c1=1.0;
        this->c2=1.0;
        this->w0=0;
    }

    ~TrainingModel(){
        cout<<"model destructor called\n";
        delete [] w;
        delete [] alpha;
        delete [] beta;
        cout<<"model memory freed\n";
    }


    li* initialzation(Data& tr,double* qii,double &d){
        // lambda=1.5f;
        li* seq= new li[m];
        if(type==L1LOSS)
            d=0;
        else
            d=0.5/c1;
        for (li i=0;i<m;i++){
            beta[i]=0;
            alpha[i]=1.0/m;
        }

        for(li i=0;i<w_size;i++)
            w[i] = 0;
        // w0=0;

        // qii is sum of sqare of each data point(sum of sqaure of each feature)
        // qii=sum(x(i,:).^2,2);
        // w=(y.*(beta-alpha))'*x;
        for (li i=0;i<m;i++){
            qii[i]=d;
            
            for(li j=0;tr.ind[i][j]!=-1;j++){
                double x=tr.x[i][j];
                li p = tr.ind[i][j];
                w[p] += (tr.y[i]*(beta[i]-alpha[i])*x);
                // cout<<tr.y[i]*(beta[i]-alpha[i])<<" "<<w[p]<<" "<<p<<"\n";
                
                qii[i]+=(x*x);
                // cout<<x<<":"<<Q<<" ";
            }
            // cout<<endl;
            // w0+=(tr.y[k]*(beta[i]-alpha[i]));
            // qii[i]=Q;
            // cout<<qii[i]<<" ";
            // cout<<"d="<<d<<" qii[i]="<<qii[i]<<endl;
            seq[i]=tr.perm[i];
        }
        cout<<"initialzation done\n";
        return seq;
    }

    double train(Data &tr){
        double *qii = new double[m];
        bool converge=false;
        double d;
        
        clock_t begin, end;
        double time_spent;

        begin = clock();

        li* seq=initialzation(tr,qii,d);
        // cout<<"alpha\tbeta\tqii\n";
        //     for(li i=0;i<m;i++){
        //         cout<<alpha[i]<<"\t"<<beta[i]<<"\t"<<qii[i]<<endl;
        //     }
           
        //     cout<<"w : ";
        //     double margin=0;
        //     for(li i=0;i<w_size;i++){
        //         cout<<w[i]<<" ";
        //         margin+=w[i]*w[i];
        //     }
        //     cout<<w0<<endl;
        // int huio;
        // cin>>huio;
        li changedvariable=0;
        double dela,delb,Ga,Gb;
        double alphaold,betaold;
        int itr=0;
        while(!converge){
            changedvariable=0;
            random_shuffle(&seq[0], &seq[m]);
            for(li k=0;k<m;k++){
                li i=seq[k];
                int yi=tr.y[i];
                double Gb=0;
                dela=0,delb=0;
                double *xi=tr.x[i];
                li *index=tr.ind[i];

                for(li j=0;index[j]!=-1;j++)
                    Gb+=(xi[j]*w[index[j]]);
                Gb=yi*Gb-1+beta[i]*d;
                double Ga=lambda-Gb-1+(beta[i]+alpha[i])*d;
                // cout<<i+1 <<" Ga ="<<Ga<<" Gb="<<Gb<<" qii="<<qii[i]<<endl;
                bool fa=true,fb=true;
                // updating alpha
                if(fabs(Ga)<1e-8)
                    fa=false;
                else if(alpha[i]<=1e-8 && Ga >= 0)
                    fa=false;
                else if(alpha[i]>=c2-1e-8 && Ga <= 0)
                    fa=false;
                else
                    fa=true;
                if(fa){
                    alphaold=alpha[i];
                    alpha[i]=min(max((alpha[i]-Ga/qii[i]),0.0),c2);
                    //cout<<alpha[i]<<endl;
                    dela=alpha[i]-alphaold;
                    if(fabs(dela)>=1e-8)
                        changedvariable++;
                }

                // updating beta
                if(fabs(Gb)<1e-4)
                    fb=false;
                else if(beta[i]<=1e-8 && Gb >= 0)
                    fb=false;
                else if(beta[i]>=c1-1e-8 && Gb <= 0)
                    fb=false;
                else
                    fb=true;
                if(fb){
                    betaold=beta[i];
                    beta[i]=min(max((beta[i]-Gb/qii[i]),0.0),c1);
                    //cout<<beta[i]<<endl;
                    delb=beta[i]-betaold;
                    if(fabs(delb)>=1e-8)
                        changedvariable++;
                }

                //updating w
                if(fa||fb){
                    for(li j=0;index[j]!=-1;j++)
                        w[index[j]]+=(yi*(delb-dela)*xi[j]);
                    // w0+=(delb-dela)*yi;
           //          for(li i=0;i<tr.n;i++){
                    //     cout<<w[i]<<" ";
                    // }
                    // cout<<w0<<endl;
                }

            }
            // cout<<"alpha\tbeta\tqii\n";
            // for(li i=0;i<m;i++){
            //     cout<<alpha[i]<<"\t"<<beta[i]<<"\t"<<qii[i]<<endl;
            // }
           
            // cout<<"w : ";
            // double margin=0;
            // for(li i=0;i<w_size;i++){
            //     cout<<w[i]<<" ";
            //     margin+=w[i]*w[i];
            // }
            // cout<<w0<<endl;
            // int hui;
            // cin>>hui;
            itr++;
            if(itr==MAXITR || changedvariable==0){
                cout<<"itr="<<itr<<"\t"<<"changedvariable="<<changedvariable<<"\n\n\n";
                converge=true;
            }
        }
        nsv=0;
        double sumalpha=0;
        for(li i=0;i<m;i++){
            sumalpha+=alpha[i];
            // cout<<beta[i]-alpha[i]<<" ";
            if((beta[i]-alpha[i])!=0){
                nsv++;
            }
        }
        end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        
        cout<<"sum of alphas="<<sumalpha<<endl<<"nummber of support vector="<<nsv<<endl; 
        cout<<"lambda="<<lambda<<endl; 
        return time_spent;
    }

    void swap(li *v,li i,li j){
        li t=v[i];
        v[i]=v[j];
        v[j]=t;
    }
    
    double trainWithShrinking(Data& tr){
        
        li i,n;
        int itr=0;
        double dela,delb,Ga,Gb;
        double *qii = new double[m];
        li *seq;
        li active=m;
        
        clock_t begin, end;
        double time_spent;

        double pga;
        double maxgaold = inf;
        double mingaold =-inf;
        double maxga,minga;

        double pgb;
        double maxgbold=inf;
        double mingbold=-inf;
        double maxgb,mingb;

        double d;
        begin = clock();
        seq=initialzation(tr,qii,d);
        
        int removedalpha=0,removedbeta=0;
        while(itr < MAXITR){
        	removedalpha=0,removedbeta=0;
            maxga=-inf;
            minga=inf;

            maxgb=-inf;
            mingb=inf;
            random_shuffle(&seq[0], &seq[active]);

            for(li k=0;k<active;k++){
                i=seq[k];
                int yi=tr.y[i];

                double *const xi = tr.x[i];
                li *const index = tr.ind[i];
                
                Gb = 0;
                for(li j=0;index[j]!=-1;j++)
                    Gb += (xi[j]*w[index[j]]);
                Gb = yi*Gb - 1 + d*beta[i];

                Ga = lambda - Gb - 1 + d*(beta[i]+alpha[i]);
                
                pga=0;
                pgb=0;
                
                if(alpha[i] == 0){
                    
                    if(Ga > maxgaold){
                        active--;
                        swap(seq,k,active);
                        removedalpha++;
                        k--;
                        continue;
                    }
                    else if(Ga < 0)
                        pga = Ga;
                } 
                else if(alpha[i] == c2){
                    
                    if(Ga < mingaold){
                        active--;
                        swap(seq,k,active);
                        removedalpha++;
                        k--;
                        continue;
                    }
                    else if(Ga > 0)
                        pga = Ga;
                    
                }  
                else
                    pga = Ga;

                maxga = max(pga,maxga);
                minga = min(pga,minga);

                if(beta[i] == 0){
                    
                    if(Gb > maxgbold){
                        active--;
                        swap(seq,k,active);
                        removedbeta++;
                        // cout<<i<<" removed from beta\n";
                        k--;
                        continue;
                    }
                    else if(Gb < 0)
                        pgb = Gb;
                } 
                else if(beta[i] == c1){
                    
                    if(Gb < mingbold){
                        active--;
                        swap(seq,k,active);
                        removedbeta++;
                        // cout<<i<<" removed from beta\n";
                        k--;
                        continue;
                    }
                    else if(Gb > 0)
                        pgb = Gb;
                } 
                else
                    pgb = Gb;

                maxgb = max(pgb,maxgb);
                mingb = min(pgb,mingb);

                dela=0,delb=0;
                if(fabs(pga) > 1.0e-10){
                    double alphaold = alpha[i];
                    alpha[i] = min(max((alpha[i] - Ga/qii[i]),0.0),c2);
                    dela = yi*(alpha[i] - alphaold);
                    // cout<<" a= "<<alpha[i];
                }

                if(fabs(pgb) > 1.0e-10){
                    double betaold = beta[i];
                    beta[i] = min(max((beta[i] - Gb/qii[i]),0.0),c1);
                    delb = yi*(beta[i] - betaold);
                    // cout<<" b= "<<beta[i];
                }

                if(fabs(pga) > 1.0e-10 || fabs(pgb) > 1.0e-10){

                    for(li j=0;index[j]!=-1;j++){
                        w[index[j]]+=((delb-dela)*xi[j]);
                        // cout<<w[index[j]]<<" ";
                    }
                    // w0+=(delb-dela);
                }
                
            }
            
            itr++;
            if(itr % 10 == 0)
                cout<<".";
                 // cout<<endl;
            fflush(stdout);
            if((maxga - minga <= EPS && maxgb - mingb <= EPS)){
                if(active==m){
                    cout<<"\ngetting out "<<active<<" "<<itr<<endl;
                    break;
                }
                else{
                    active=m;
                    cout<<"*";
                    maxgaold=inf;
                    mingaold=-inf;
                    maxgbold=inf;
                    mingbold=-inf;
                    continue;
                }

            }

            maxgaold = maxga;
            mingaold = minga;
            maxgbold = maxgb;
            mingbold = mingb;

            if(maxgaold <= 0)
                maxgaold = inf;
            if(mingaold >= 0)
                mingaold = -inf;

            if(maxgbold <= 0)
                maxgbold = inf;
            if(mingbold >= 0)
                mingbold = -inf;     
        }

        cout<<"\noptimization complete, itr = "<<itr<<endl;
        if(itr >= MAXITR)
            cout<<"\nmax iteration reached\n";
        end = clock();

        nsv=0;
        // ofstream ofs("dcd.out");
        double sumalpha=0,sumbeta=0,v=0;
        for(i=0; i<w_size; i++)
            v += w[i]*w[i];
        for(li i=0;i<m;i++){
            sumalpha+=alpha[i];
            sumbeta+=beta[i];
            v += (-beta[i]*2 + 2*lambda*alpha[i] + d*(beta[i]*beta[i]+alpha[i]*alpha[i]));
            if(beta[i]-alpha[i]!=0){
                nsv++;
            }
            // cout<<alpha[i]<<" "<<beta[i]<<endl;
        }
        v-= 2*lambda;
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        cout<<"objective value = "<<v/2<<endl;
        cout<<"sum of alphas="<<sumalpha<<endl<<"nummber of support vector="<<nsv<<endl; 
        cout<<"sum of beta = "<<sumbeta<<endl;
        cout<<"lambda="<<lambda<<endl; 
 
       delete [] qii;
       delete [] seq;
       return time_spent;
    }


    double kernel(vd xi,double* xj,vl indi, int* indj,li ni,li nj){
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

    double kernel(double* xi,double* xj,int* indi, int* indj,KernelType type){
        double kij=0;
        if(type==LINEAR){
            kij=0.0;
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
        }
        else{
            kij=0.0;
            li i=0,j=0;
            while(indi[i]!=-1&&indj[j]!=-1){
                if(indi[i]==indj[j]){
                    kij+=(xi[i]-xj[j])*(xi[i]-xj[j]);
                    i++;
                    j++;
                }
                else if(indi[i]<indj[j])
                    i++;
                else
                    j++;
            }
            double gamma=1.0/w_size;
            kij=exp(-kij*gamma);

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

    double trainWithShrinkingKernel(Data &tr,KernelType type){
        double *qii = new double[m];

        c1=1.0;
        c2=1.0;
        double d;
        if(type==L1LOSS)
            d=0;
        else
            d=0.5/c1;
        
        li* seq= new li[m];
        
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

            double ai=alpha[0];
            double* xi=tr.x[i];
            li *index=tr.ind[i];
            int yi=tr.y[i];
            Q=kernel(xi,xi,index,index,type);
            for(li j=0;j<m;j++)
                gbeta1[i] += (kernel(tr.x[j],xi,tr.ind[j],index,type)*tr.y[j]);
            gbeta1[i] = gbeta1[i]*-ai;
            qii[i]=(Q+d);
            seq[i]=i;
            // cout<<qii[i]<<" "<<beta[i]<<" "<<alpha[i]<<endl;
        }

        cout<<"initialzation done\n";
        // int uo;
        // cin>>uo;
        int maxitr = 500;
        while(itr < maxitr){
            
            maxga=-inf,maxgb=-inf,minga=inf,mingb=inf;
            random_shuffle(&seq[0], &seq[active]);

            for(li k=0;k<active;k++){
                i=seq[k];
                
                double alphai=alpha[i],betai=beta[i];

                // if(fabs(betai-alphai)<1e-7){
                //  seq.erase(seq.begin()+k);
     //             cout<<i<<"beta - alpha differance too low\n";
     //             active--;
     //             continue;
                // }

                double* xi=tr.x[i];
                li* index=tr.ind[i];
                int yi=tr.y[i];
                
                Gb = yi*gbeta1[i] - 1 + gbeta2[i];

                Ga = lambda - Gb - 1 + d*(betai+alphai);
                
                // cout<<Ga<<" "<<Gb<<" ";

                pga=0,pgb=0;
                delb=0,dela=0;
                // betaold=betai,alphaold=alphai;

                if(alphai==0){
                    
                    if(Ga > maxgaold){
                        active--;
                        swap(seq,active,k);
                        // cout<<i<<" removed from alpha\n";
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
                        // cout<<i<<" removed from alpha\n";
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
                        // cout<<i<<" removed from beta\n";
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
                    // cout<<alpha[i]-Ga/qii[i]<<endl;
                    // cout<<" A-"<<alpha[i]<<" ";
                    alphaold=alphai;
                    alpha[i]=min(max((alphai-Ga/qii[i]),0.0),c2);
                    dela=alpha[i]-alphaold;
                    // cout<<" a-"<<alpha[i]<<" ";
                }

                if(fabs(pgb) > 1e-10){
                    // cout<<beta[i]-Gb/qii[i]<<endl;
                    betaold=betai;
                    beta[i]=min(max((betai-Gb/qii[i]),0.0),c1);
                    delb=beta[i]-betaold;
                    // cout<<" b-"<<beta[i]<<" ";
                }

                if(fabs(pga) > 1e-10 || fabs(pgb) > 1e-10){
                    // cout<<beta[i]-Gb/qii[i]<<endl;
                    gbeta2[i] = d*beta[i];
                    for(li j=0;j<m;j++){
                        gbeta1[j]+= (kernel(tr.x[j],xi,tr.ind[j],index,type)*(delb-dela)*yi); 
                    }
                }

                // cout<<alpha[i]<<" "<<beta[i]<<endl;
                // cout<<endl;
                maxga=max(pga,maxga);
                maxgb=max(pgb,maxgb);

                minga=min(pga,minga);
                mingb=min(pgb,mingb);

            }
            // cin>>uo;
            end = clock();
            time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            if(time_spent>=10800){
                cout<<"time out!!\n";
                break;
            }
            itr++;
            if(itr % 10 == 0)
                cout<<".";
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
        if(itr>=maxitr)
            cout<<"max iteration reached\n";
        end = clock();
        nsv=0;
        double sumalpha=0;
        for(li i=0;i<m;i++){
            sumalpha+=alpha[i];
            if(fabs(beta[i]-alpha[i])>=1e-5)
                nsv++;
        }
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        cout<<"training complete\n\ntime for training : "<<time_spent<<"sec"<<endl;
        cout<<"lambda="<<lambda<<endl;
        cout<<"sum of alphas="<<sumalpha<<" nsv="<<nsv<<endl;

        delete [] qii;
        delete [] gbeta1;
        delete [] gbeta2;
        delete [] seq;
        return time_spent;
    }

    double predictionWithKernel(Data &tt,Data &tr,KernelType type){
        
        li correct=0;
        for(li i=0;i<tt.m;i++){
            double d=0;
            for(li j=0;j<m;j++){
                d+=(kernel(tt.x[i],tr.x[j],tt.ind[i],tr.ind[j],type)*tr.y[j]*(beta[j]-alpha[j]));
            }
            if((d>=0&&tt.y[i]==1)||(d<0&&tt.y[i]==-1))
                correct++;
            // cout<<correct<<endl;
        }
        
        double accuracy=correct*100.0/tt.m;

        cout<<"accuracy="<<accuracy<<"("<<correct<<"/"<<tt.m<<")"<<endl;
        return accuracy;
    }

    void printModel(li m,li n){
        cout<<"nummber of support vector : "<<nsv<<endl;
        // cout<<"alpha\tbeta\tqii\n";
        // for(li i=0;i<m;i++){
        //     cout<<Alpha[i]<<"\t"<<Beta[i]<<"\t"<<Qii[i]<<endl;
        // }
       
        cout<<"w : ";
        double margin=0;
        for(li i=0;i<w_size;i++){
            cout<<w[i]<<" ";
            margin+=w[i]*w[i];
        }
        //cout<<w0<<endl;
        margin=1/sqrt(margin);
        cout<<"margin = "<<margin<<endl;
    }

    double prediction(Data &tt){
        std::fstream f;
       string::size_type sz;
       string str="out"+to_string(thresh);
        f.open (str, std::fstream::in | std::fstream::out | std::fstream::app);
        li pos=0,neg=0,fp=0,fn=0;
        //vector<int> pred;
        li correct=0;
        for(li i=0;i<tt.m;i++){
            double d=0;
            li j;
            for(j=0;tt.ind[i][j]<w_size && tt.ind[i][j]!=-1;j++){
                d += tt.x[i][j]*w[tt.ind[i][j]];
            }
            // cout<<d<<" "<<tt.y[i]<<endl;
            if((d >= thresh && tt.y[i] == 1)){
                pos++;
                correct++;
                f << "1\n";
            }
            else if(d < thresh && tt.y[i] == -1){
                neg++;
                correct++;
                f << "-1\n";
            }
            else if(d < thresh && tt.y[i] == 1){
                fn++;
                f << "-1\n";
            }
            else{
                f << "1\n";
                fp++;
            }
        }
        double accuracy=correct*100.0/tt.m;

        cout<<"accuracy="<<accuracy<<"("<<correct<<"/"<<tt.m<<")"<<endl;
        // f<<"------------------\n";
        // actual > pred v
        cout<<"confusion matrix\n";
        cout<<"\tpos\tneg\n";
        cout<<"pos\t"<<pos<<"\t"<<fp<<endl;
        cout<<"neg\t"<<fn<<"\t"<<neg<<endl;
        f.close();
        return accuracy;
    }
};

void crossValidate(string file,int fold, Data &tr){
    // cout<<file<<endl;
    if(fold == 0)
        fold= tr.m;   
    vector<int> trainsize(fold+1,0);
    for(int i=0;i<=fold;i++){
        trainsize[i]=i*tr.m/fold;
        // cout<<trainsize[i]<<" ";
    }
    std::vector<int> seq;
    for(int i=0;i<tr.m;i++)
        seq.push_back(i);
    random_shuffle(seq.begin(),seq.end());
    // tr.printData();
    double lambda;
    double avgacc=0;
    int option=1;
    vector< pair<double,double> > acc;
    vector< pair<double,pair<double,pair<double,li> > > >history;
    int nsv=inf;
     for(lambda=0.1;lambda<0.6;){
        // cout<<"enter lambda or enter 0 to stop the execution: : ";
        avgacc=0;
        for(int i=0;i<fold;i++){
            tr.start=trainsize[i];
            tr.end=trainsize[i+1];
            // cout<<tr.start<<" "<<tr.end<<" ";
            Data tr1=Data();
            tr1.copy(tr,false,seq);
            tr1.sort();
            // tr1.printData();
            // cout<<"--------------------------\n";
            TrainingModel model=TrainingModel(tr1.m,tr1.n);
            model.type=option;
            model.lambda=lambda;
            model.trainWithShrinking(tr1);
            Data tt=Data();
            tt.copy(tr,true,seq);
            avgacc+=model.prediction(tt);
            cout<<"........................................\n\n";
            nsv=min(model.nsv,nsv);
            // tr1.printData();
        }
        cout<<"cross validation accuracy = "<<avgacc/fold<<endl;
        // acc.push_back(make_pair(lambda,avgacc/fold));
        history.push_back(make_pair(lambda,make_pair(avgacc/fold,make_pair(0,nsv))));
                // sort(history.begin(), history.end() ,compare);
        cout<<"..............................................................\nlambda\taccuracy\ttraining time\tnsv\n";
        for(int i=0;i<history.size();i++)
            cout<<history[i].first<<"\t"<<history[i].second.first<<"\t\t"<<history[i].second.second.first<<"\t\t"<<history[i].second.second.second<<endl;
        if(lambda<=0.9)
            lambda+=0.1;
        else
            lambda+=0.5;
    }
    while(1){
        cout<<"enter lambda or enter 0 to stop the execution: : ";
        cin>>lambda;
        if(lambda==0)
            break;
        avgacc=0;
        for(int i=0;i<fold;i++){
            tr.start=trainsize[i];
            tr.end=trainsize[i+1];
            // cout<<tr.start<<" "<<tr.end<<" ";
            Data tr1=Data();
            tr1.copy(tr,false,seq);
            tr1.sort();
            // tr1.printData();
            // cout<<"--------------------------\n";
            Data tt=Data();
            tt.copy(tr,true,seq);
            // tt.sort();
            // tt.printData();
            TrainingModel model=TrainingModel(tr1.m,tr1.n);
            model.type=option;
            model.lambda=lambda;
            model.trainWithShrinking(tr1);
            avgacc+=model.prediction(tt);
            nsv=min(model.nsv,nsv);
            cout<<"........................................\n\n";

            // tr1.printData();
        }
        cout<<"cross validation accuracy = "<<avgacc/fold<<endl;
        acc.push_back(make_pair(lambda,avgacc/fold));
        cout<<"\n\nlambda\t average accuracy\n";
        for(int i=0;i<acc.size();i++)
            cout<<acc[i].first<<"\t\t"<<acc[i].second<<endl;
        cout<<endl<<endl;
    }
    
}
void crossValidate(string file,int fold){
    // cout<<file<<endl;
    Data tr = Data();
    tr.readinput(file,true);
    if(fold == 0)
        fold= tr.m;   
    vector<int> trainsize(fold+1,0);
    for(int i=0;i<=fold;i++){
        trainsize[i]=i*tr.m/fold;
        // cout<<trainsize[i]<<" ";
    }
    std::vector<int> seq;
    for(int i=0;i<tr.m;i++)
        seq.push_back(i);
    random_shuffle(seq.begin(),seq.end());
    // tr.printData();
    double lambda;
    // double avgacc=0;
    int option=1;
    cout<<"enter lambda : ";
    cin>>lambda;
    vector< pair<double,pair<double,pair<double,li> > > >history;
    // for(lambda=0.1;lambda<25;){
        double avgacc=0;
        int nsv=inf;
        for(int i=0;i<fold;i++){
            tr.start=trainsize[i];
            tr.end=trainsize[i+1];
            // cout<<tr.start<<" "<<tr.end<<" ";
            Data tr1=Data();
            tr1.copy(tr,false,seq);
            tr1.sort();
            // tr1.printData();
            // cout<<"--------------------------\n";
            Data tt=Data();
            tt.copy(tr,true,seq);
            // tt.printData();
            TrainingModel model=TrainingModel(tr1.m,tr1.n);
            model.type=option;
            model.lambda=lambda;
            model.trainWithShrinking(tr1);
            avgacc+=model.prediction(tt);
            nsv=min(model.nsv,nsv);
            cout<<"........................................\n\n";

            // tr1.printData();
        }
        history.push_back(make_pair(lambda,make_pair(avgacc/fold,make_pair(0,nsv))));
                // sort(history.begin(), history.end() ,compare);
        cout<<"..............................................................\nlambda\taccuracy\ttraining time\tnsv\n";
        for(int i=0;i<history.size();i++)
            cout<<history[i].first<<"\t"<<history[i].second.first<<"\t\t"<<history[i].second.second.first<<"\t\t"<<history[i].second.second.second<<endl;
        if(lambda <= 0.9)
            lambda+=0.1;
        else
            lambda+=0.5;
    // cout<<"cross validation accuracy = "<<avgacc/fold<<endl;

}


// bool compare(pair<double,pair<double,li> > x,pair<double,pair<double,li> >y) {
//     if(x.second.first > y.second.first)
//         return true;
//     else if(x.second.first == y.second.first){
//         if(x.second.second <= x.second.second)
//             return true;
//     }
//     return false;
// }

int main(int argc,char* argv[]){
    if(argc==1){
        cout<<"./dcd [train-file] [option]\n -test [filename]\n";
        cout<<" -v 0 for leave one out validation\n -v n for n fold validation\n";
        cout<<" -vt n [testfile] for setting lambda while doing n-fold validation and report test accuracy for dataset testfile\n";
        cout<<" -k 2 [testfile] for gaussian kernel\n";
        exit(0);
    }
    else{
        // cout<<strcmp(argv[2],"-v")<<" "<<strcmp(argv[3],"0")<<endl;
        if(strcmp(argv[2],"-test")==0){
            Data tr = Data();
            Data tt = Data();
            double lambda=1;
            tr.readinput(argv[1],true);
            cout<<tr.m<<" "<<tr.n<<endl;
            // tr.printData();
            tr.sort();
            // tr.printData();
            cout<<"out\n";
            fflush(stdout);
            int option=1;
            tt.readinput(argv[3],false);
            cout<<tt.m<<" "<<tt.n<<endl;
            cout<<"out2\n";
            TrainingModel model=TrainingModel(tr.m,tr.n);
            model.type=option;
            vector< pair<double,pair<double,pair<double,li> > > >history;
            model.lambda=0.3;
            for(lambda=0.1;lambda<=30.0;){
            // for(double i = 0;i<1;i+=0.1){
                // tr.m=i;
                // model.m=i;
                // thresh=i;
                model.lambda = lambda;
                double timespent = model.trainWithShrinking(tr);
                double acc = model.prediction(tt);
                // double timespent = model.trainWithShrinking(tr);
                // double acc=model.prediction(tt);

                history.push_back(make_pair(lambda,make_pair(acc,make_pair(timespent,model.nsv))));
                // sort(history.begin(), history.end() ,compare);
                cout<<"..............................................................\nsamples\taccuracy\ttraining time\tnsv\n";
                for(int i=0;i<history.size();i++)
                    cout<<history[i].first<<"\t"<<history[i].second.first<<"\t\t"<<history[i].second.second.first<<"\t\t"<<history[i].second.second.second<<endl;
                if(lambda <= 0.9)
                    lambda+=0.1;
                else
                    lambda+=0.5;
            }
            while(1){
                cout<<"enter lambda or enter 0 to stop the execution: ";
                cin>>lambda;
                if(lambda==0)
                    break;
                model.lambda=lambda;
                double timespent = model.trainWithShrinking(tr);
                double acc=model.prediction(tt);
                history.push_back(make_pair(lambda,make_pair(acc,make_pair(timespent,model.nsv))));
                // sort(history.begin(), history.end() ,compare);
                cout<<"..............................................................\nlambda\taccuracy\ttraining time\tnsv\n";
                for(int i=0;i<history.size();i++)
                    cout<<history[i].first<<"\t"<<history[i].second.first<<"\t\t"<<history[i].second.second.first<<"\t\t"<<history[i].second.second.second<<endl;
            }
            history.clear();
        }
        else if(strcmp(argv[2],"-v")==0)
            crossValidate(string(argv[1]),stoi(argv[3]));
        else if(strcmp(argv[2],"-vt")==0){
            Data tr = Data();
            tr.readinput(argv[1],true);
            cout<<"train file read\n";
            crossValidate(string(argv[1]),stoi(argv[3]),tr);
            
            Data tt = Data();
            double lambda=1;
            
            fflush(stdout);
            int option=1;
            tt.readinput(argv[4],false);
            cout<<"test file read\n";
            TrainingModel model=TrainingModel(tr.m,tr.n);
            model.type=option;
            cout<<"enter lambda : ";
            cin>>lambda;
            model.lambda=lambda;
            tr.sort();
            model.trainWithShrinking(tr);
            model.prediction(tt);
        }
        else if(strcmp(argv[2],"-k")==0 && strcmp(argv[3],"2")==0){
            Data tr = Data();
            tr.readinput(argv[1],true);
            cout<<"train file read\n";
            Data tt = Data();
            double lambda=1;
            int option=1;
            tt.readinput(argv[4],false);
            cout<<"test file read\n";
            TrainingModel model=TrainingModel(tr.m,tr.n);
            model.type=option;
            cout<<"enter lambda : ";
            cin>>lambda;
            model.lambda=lambda;
            model.trainWithShrinkingKernel(tr,GAUSSIAN);
            model.predictionWithKernel(tt,tr,GAUSSIAN);
        }
        else{
            cout<<"./dcd [train-file] [option]\n -test [filename]\n";
            cout<<" -v 0 for leave one out validation\n -v n for n fold validation\n";
            cout<<" -vt n [testfile] for setting lambda while doing n-fold validation and report test accuracy for dataset testfile\n";
            cout<<" -k 2 [testfile] for gaussian kernel\n";
        }
            exit(0); 

    }
    Data tr = Data();
    Data tt = Data();
    string file="real-sim_sparse_1";
    double lambda=1;
    cout<<"enter file name : ";
    // cin>>file;
    
    // string trainfile="../../split/"+file+".train";
    // string testfile="../../split/"+file+".test";
    string trainfile="../../Data_ML/"+file+".train";
    string testfile="../../Data_ML/"+file+".test";
    // string trainfile="../../Data_ML/mnist38_norm_svm_full_1.train";
    // string testfile="../../Data_ML/mnist38_norm_svm_full_1.test";
    // string trainfile="sparsedata1.train";//"kddb_unnorm_svm_1.train"
    // string testfile="sparsedata1.test";//"kddb_unnorm_svm_1.test";
    // string trainfile="../../Data_ML/kddb_unnorm_svm_1.train";
    // string testfile="../../Data_ML/kddb_unnorm_svm_1.test";

    tr.readinput(trainfile,true);
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
    tt.readinput(testfile,false);
    // tt.printData();
    cout<<tt.m<<" "<<tt.n<<endl;
    cout<<"out2\n";
    TrainingModel model=TrainingModel(tr.m,tr.n);
    // for(int i=0;i<tr.m;i++){
    //     for(int j=0;j<tr.m;j++){
    //         cout<<model.kernel(tr.x[i],tr.x[j],tr.ind[i],tr.ind[j],GAUSSIAN)<<" ";
    //     }
    //     cout<<endl;
    // }
    
    model.type=option;
    // for(double i=1;i<30.0f&&lambda!=0;i+=0.5){
    //     // cout<<"enter lambda : ";
    //     // cin>>lambda;
    //     model.lambda=i;
    //     model.trainWithShrinking(tr);
    //     model.prediction(tt);
    // }
    // while(1){

        cout<<"enter lambda : ";
    cin>>lambda;
    // if(lambda==0)
    //     break;
    model.lambda=lambda;
    cout<<"with or without kernel? ";
    int op=2;
    // cin>>op;
    if(op==1){
        model.trainWithShrinking(tr);
        model.prediction(tt);
    }
    else{
        model.trainWithShrinkingKernel(tr,GAUSSIAN);
        model.predictionWithKernel(tt,tr,GAUSSIAN);

    }
    // model.train(tr);
    //     model.printModel(tr.m,tr.n);
    //     // // // model->printModel(tr->m,tr->n);
    //     cout<<"test ";
        
    
    // model.train(tr,lambda);
    
    // model.predictionWithKernel(tt,tr);
 
    return 0;
}