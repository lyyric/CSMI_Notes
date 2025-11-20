/*
 * file:                  main.cpp
 * description:           Interactive program for isosurface extraction.
 * author:                Julien Tierny <tierny@telecom-paristech.fr>.
 * date:                  January 2013.
 */

#include  <iostream>

#include  "Editor.h"
#include  "UserInterface.h"



Editor editor_;
UserInterface userInterface_;

int printUsage(const string binPath){
  
  cout << "Usage:" << endl;
  cout << "  " << binPath << " -t <input tet mesh>" << endl;
  
  return 0;
}

int main(int argc, char **argv) {

  string inputFileName;
  
  for(int i = 0; i < argc; i++){
    
    if((string(argv[i]) == "-t")&&(i + 1 < argc)){
      inputFileName = string(argv[i + 1]);
    }
  }

  if(!inputFileName.length()){
    printUsage(argv[0]);
    return -1;
  }

  editor_.loadInputMesh(inputFileName);

  userInterface_.setEditor(&editor_);
  userInterface_.run();

  return 0;
}
