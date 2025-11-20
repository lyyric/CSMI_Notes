/*
  Editor.cpp: Data-structures and processing. 
  Copyright (C) 2013  Julien Tierny <tierny@telecom-paristech.fr>

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/


#include  "Editor.h"

Editor::Editor(){

  inputMesh_ = NULL;
  meshReader_ = NULL;
  type_ = SIMPLE;
}

Editor::Editor(const Editor& other){

}

Editor::~Editor(){
 
#ifndef _WIN32
  if(meshReader_) meshReader_->Delete();

  for(unsigned int i = 0; i < vtkIsoSurfacers_.size(); i++){
    if(vtkIsoSurfacers_[i])
      vtkIsoSurfacers_[i]->Delete();
  }
  
  for(unsigned int i = 0; i < isoSurfacers_.size(); i++){
    if(isoSurfacers_[i])
      isoSurfacers_[i]->Delete();
  }
#endif
}

Editor& Editor::operator=(const Editor& other){
  return *this;
}

bool Editor::operator==(const Editor& other) const{

  ///TODO: return ...;
  return false;
}

int Editor::addIsoSurface(const double &isoValue, const bool &useVtk){
 
  if(useVtk){
    vtkContourGrid *isoSurfacer = extractIsoSurfaceVTK(isoValue);
    if(isoSurfacer){
      isoSurfaceList_.push_back(isoSurfacer->GetOutput());
      vtkIsoSurfacers_.push_back(isoSurfacer);
      isoSurfacers_.push_back(NULL);
    }
  }
  else{
    IsoSurfacer *isoSurfacer = extractIsoSurface(isoValue);
    
    if(isoSurfacer){
      isoSurfaceList_.push_back(isoSurfacer->GetOutput());
      isoSurfacers_.push_back(isoSurfacer);
      vtkIsoSurfacers_.push_back(NULL);
    }
  }
 
  isoValueList_.push_back(isoValue);
 
  return isoSurfaceList_.size() - 1;
}

int Editor::deleteIsoSurface(const int &isoSurfaceId){
  
  if((isoSurfaceId < 0)||(isoSurfaceId >= (int) isoSurfaceList_.size()))
    return -1;
  
  isoValueList_.erase(isoValueList_.begin() + isoSurfaceId);
  isoSurfaceList_.erase(isoSurfaceList_.begin() + isoSurfaceId);
  
  if(vtkIsoSurfacers_[isoSurfaceId]){
    // vtk was used
    vtkIsoSurfacers_[isoSurfaceId]->Delete();
    vtkIsoSurfacers_.erase(vtkIsoSurfacers_.begin() + isoSurfaceId);
    isoSurfacers_.erase(isoSurfacers_.begin() + isoSurfaceId);
  }
  else{
    isoSurfacers_[isoSurfaceId]->Delete();
    isoSurfacers_.erase(isoSurfacers_.begin() + isoSurfaceId);
    vtkIsoSurfacers_.erase(vtkIsoSurfacers_.begin() + isoSurfaceId);
  }
  
  cout << "[Editor] Isosurface #" << isoSurfaceId << " deleted." << endl;
  
  if(!isoSurfaceList_.size())
    return -1;
  
  if(isoSurfaceId >= (int) isoSurfaceList_.size())
    return isoSurfaceList_.size() - 1;

  return -1;
}

IsoSurfacer* Editor::extractIsoSurface(const double &isoValue){

  return NULL;
}

vtkContourGrid* Editor::extractIsoSurfaceVTK(const double &isoValue){

  DebugTimer t;
  
  cout << "[Editor] Isosurface extracted in " << t.getElapsedTime()
    << " s." << endl;

  return NULL;
}

vtkPolyData* Editor::getIsoSurface(const int &isoSurfaceId) const{
  
  if((isoSurfaceId < 0)||(isoSurfaceId >= (int) isoSurfaceList_.size()))
    return NULL;
  
  return isoSurfaceList_[isoSurfaceId];
}

int Editor::loadInputMesh (const string &fileName){

  // TODO

  cout << "[Editor] done! (read " << inputMesh_->GetNumberOfCells() << " cells)" << endl;
  
  return 0;
}

int Editor::moveIsoSurface(const int& isoSurfaceId, const double &shift,
  const bool &useVtk){

  if((isoSurfaceId < 0)||(isoSurfaceId >= (int) isoSurfaceList_.size()))
    return -1;
  
  double newIsoValue = isoValueList_[isoSurfaceId] + shift;
 
  double range[2];
  getScalarRange(range[0], range[1]);
  cout << "[Editor] Moving isosurface #" << isoSurfaceId
    << " to " << newIsoValue << " from range [" << range[0] << ";" << range[1] << "] ..." << endl; 
 
  // delete the previous computation
  if(isoSurfacers_[isoSurfaceId]){
    isoSurfacers_[isoSurfaceId]->Delete();
    isoSurfacers_[isoSurfaceId] = NULL;
  }
  if(vtkIsoSurfacers_[isoSurfaceId]){
    vtkIsoSurfacers_[isoSurfaceId]->Delete();
    vtkIsoSurfacers_[isoSurfaceId] = NULL;
  }
   
  // now update
  if(useVtk){
    vtkContourGrid *isoSurfacer = extractIsoSurfaceVTK(newIsoValue);
    
    isoSurfaceList_[isoSurfaceId] = isoSurfacer->GetOutput();
    vtkIsoSurfacers_[isoSurfaceId] = isoSurfacer;
  }
  else{
    IsoSurfacer *isoSurfacer = extractIsoSurface(newIsoValue);
    
    isoSurfaceList_[isoSurfaceId] = isoSurfacer->GetOutput();
    isoSurfacers_[isoSurfaceId] = isoSurfacer;
  }
  
  isoValueList_[isoSurfaceId] = newIsoValue;
  
  return 0;
}

