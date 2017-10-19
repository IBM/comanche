#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

int main(int argc, char * argv[])
{
	if(argc!=3) {
		printf("removeheader <infile> <headerfile> \n");
		return -1;
	}

	ifstream iFile;
  try {
		iFile.open(argv[1]);
  }
  catch(...) {
		cerr << "Error: unable to open source file [" << argv[1] << "]\n";
		return -1;
  }

	/* read input file */
	std::string iBuf;
  {
		std::string line;
		while(std::getline(iFile,line)) {
			iBuf += line;
			iBuf += "\n";
		}
  }
	/* close in file */
	iFile.close();

	/* open code header */
	ifstream iHeader;
	try { iHeader.open(argv[2]); }
	catch(...) { cerr << "Error: unable to open header file [" << argv[2] << "]\n"; return -1; }
	std::string iHeaderTxt;
  {
		std::string line;
		while(std::getline(iHeader,line)) {
			iHeaderTxt += line;
			iHeaderTxt += "\n";
		}
  }

	/* open out file */
	ofstream oFile;
  try {
		oFile.open(argv[1]);
  }
  catch(...) {
		cerr << "Error: unable to open target file [" << argv[1] << "]\n";
		return -1;
  }


	/* check a header already exists */
	{
		string::size_type pos = 0;
		while(iBuf[pos] == ' ' || iBuf[pos] == '\t' || iBuf[pos] == '\n') pos++;
		if(iBuf[pos] == '/' && iBuf[pos+1] == '*') {			
			/* modify string */
			string::size_type pos = 0;
			pos = iBuf.find("*/");
			assert(pos != string::npos);
			pos+=2;
			if(iBuf[pos]==' ') {
				while(iBuf[pos]!=' ') pos++;
			}

			iBuf.replace(0,pos,iHeaderTxt);

		}
		else{
			//			printf("[%c%c]\n",iBuf[pos],iBuf[pos+1]);
			iBuf.insert(0,iHeaderTxt);
		}
		
	}


	/* write out result */
	oFile.write(iBuf.c_str(),iBuf.length());

	// clean up
	oFile.close();

}

	
