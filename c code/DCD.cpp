#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

void readinput(){
	ifstream input("webspam_tr80K" );
	string line,token,a; 
	stringstream iss,ss;

    while ( getline(input, line) )
    {
        iss << line;
        int y;
        iss>>y;
        while(iss>>token){
        	ss<<token;
        	while ( getline(ss, a, ':') )
	        {
	            //cout << a << endl;
	        }
	        ss.clear();
        }
        iss.clear();
    }
}

int main(){
	readinput();
	cout<<"reading complete\n";
	return 0;
}