#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>
#define MAXITR 50;

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
    vf alpha,alphaold,beta,betaold,w,Ga,Gb,qii;

    TrainingModel(){}

    TrainingModel(Data tr){
        alpha=vf(tr.m,1.0/tr.m);
        alphaold=alpha;
        beta=vf(tr.m,0);
        betaold=beta;
        w=vf(tr.n,0);

        for (li i=0;i<tr.m;i++){
            for(li j=0;j<tr.ind[i].size();j++){
                w[tr.ind[i][j]]+=(tr.y[i]*(beta[i]-alpha[i])*tr.x[i][(tr.ind[i][j])]);
            }
        }

//         w=(y.*(b1-a1))'*x;
// j=0;
// Gb=y.*(x*w')-1;
// Ga=lambda-Gb-1;
// Q=zeros(1,m);
// %qii=<x(i,:),x(i,:)>
// for i=1:m
//     Q(i)=sum(x(i,:).^2);
// end
// disp('initialization complete');
    }


    void initialization(){

    }
    void printModel(li m,li n){
        cout<<"alpha\tbeta\n";
        for (li i = 0; i < m; i++){
            cout<<alpha[i]<<"\t"<<beta[i]<<endl;
        }
        cout<<"w : ";
        for(li i=0;i<n;i++){
            cout<<w[i]<<" ";
        }
        cout<<endl;
    }
};



int main(){
    Data *tr=new Data();
    Data *tt=new Data();
    string trainfile="sparsedata1.train";//"kddb_unnorm_svm_1.train";
    string testfile="sparsedata1.test";//"kddb_unnorm_svm_1.test";
	tr->readinput(trainfile);
    tr->printData();
    TrainingModel *model=new TrainingModel(*tr);
    model->printModel(tr->m,tr->n);
    tt->readinput(testfile);
    cout<<tr->m<<" "<<tr->n<<endl<<tt->m<<" "<<tt->n<<endl;
    //tt->printData();
	cout<<"reading complete\n";
    delete tr;
    delete tt;
    delete model;
	return 0;
}