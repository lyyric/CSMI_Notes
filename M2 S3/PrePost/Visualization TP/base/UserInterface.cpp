/*
  UserInterface.cpp: Interactions and rendering
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


#include "UserInterface.h"

vtkStandardNewMacro(CustomInteractor);

void CustomInteractor::OnKeyPress() {
  
  vtkRenderWindowInteractor *interactor = this->Interactor;
  string key = interactor->GetKeySym();
 
  if((key == "a")||(key == "A")){
    // add an isosurface in the middle of the range
    userInterface_->addIsoSurface();
  }
  else if((key == "d")||(key == "D")){
    userInterface_->deleteCurrentIsoSurface();
  }
  else if(key == "s"){
    userInterface_->setCustomImplementation(SIMPLE);
  }
  else if(key == "S"){
    userInterface_->setCustomImplementation(STANDARD);
  }
  else if(key == "t"){
    userInterface_->switchCurrentIsoSurfaceTransparency();
  }
  else if(key == "x"){
    userInterface_->setCustomImplementation(FAST);
  }
  else if(key == "Up"){
    userInterface_->moveCurrentIsoSurface(true);
  }
  else if(key == "Down"){
    userInterface_->moveCurrentIsoSurface(false);
  }
  else if(key == "Prior"){
    userInterface_->moveCurrentIsoSurface(true, true);
  }
  else if(key == "Next"){
    userInterface_->moveCurrentIsoSurface(false, true);
  }
  else if(key == "Right"){
    userInterface_->changeIsoSurface(1);
  }
  else if(key == "Left"){
    userInterface_->changeIsoSurface(-1);
  }
  else if(key == "Home"){
    userInterface_->changeIsoSurface(-INT_MAX);
  }
  else if(key == "End"){
    userInterface_->changeIsoSurface(INT_MAX);
  }
  else if((key == "v")||(key == "V")){
    userInterface_->setVtk(!userInterface_->getVtk());
  }
  else if(key == "Escape"){
    exit(0);
  }
  
  // forward events to the rest of the VTK pipeline
  vtkInteractorStyleTrackballCamera::OnKeyPress();
}

UserInterface::UserInterface() {
  
  editor_ = NULL;
  
  renderer_ = NULL;
  renderWindow_ = NULL;
  interactor_ = NULL;
  geometryFilter_ = NULL;
  boundaryMapper_ = NULL;
  mainActor_ = NULL;
  customInteractor_ = NULL;
  currentIsoSurfaceId_ = -1;
  
  useVtk_ = true;
}

UserInterface::UserInterface ( const UserInterface& other ) {

}

UserInterface::~UserInterface() {

#ifndef _WIN32
  if(mainActor_) mainActor_->Delete();
  if(geometryFilter_) geometryFilter_->Delete();
  if(boundaryMapper_) boundaryMapper_->Delete();
  if(renderer_) renderer_->Delete();
  if(renderWindow_) renderWindow_->Delete();
  if(customInteractor_) customInteractor_->Delete();
  if(interactor_) interactor_->Delete();

  for(unsigned int i = 0; i < isoSurfaceMappers_.size(); i++)
    isoSurfaceMappers_[i]->Delete();
#endif

}

UserInterface& UserInterface::operator= ( const UserInterface& other ) {
  return *this;
}

bool UserInterface::operator== ( const UserInterface& other ) const {
  
  ///TODO: return ...;
  return false;
}

int UserInterface::addIsoSurface(){
 
  cout << "[UserInterface] Adding new isosurface..." << endl;
  
  double range[2];
  
  editor_->getScalarRange(range[0], range[1]);
 
  cout << "[UserInterface] Adding an isosurface at isovalue "
    << (range[0] + range[1])/2 << " from range [" << range[0] << ";" << range[1] << "] ..." << endl;

  currentIsoSurfaceId_ = 
    editor_->addIsoSurface((range[0] + range[1])/2, useVtk_);
 
  // feed that to the renderer
  vtkPolyDataMapper *surfaceMapper = vtkPolyDataMapper::New();
  vtkActor *surfaceActor = vtkActor::New();
  
  surfaceMapper->SetInputData(editor_->getIsoSurface(currentIsoSurfaceId_));
  surfaceMapper->SetScalarRange(range);
  
  isoSurfaceMappers_.push_back(surfaceMapper);
  surfaceActor->SetMapper(surfaceMapper);
  isoSurfaceActors_.push_back(surfaceActor);
  
  isTransparent_.push_back(false);
  
  renderer_->AddActor(surfaceActor);
  renderWindow_->Render();
  
  return 0;
}

int UserInterface::changeIsoSurface (const int &shift){

  if(currentIsoSurfaceId_ + shift < 0)
    currentIsoSurfaceId_ = 0;
  else if(currentIsoSurfaceId_ + shift >= (int) isoSurfaceActors_.size())
    currentIsoSurfaceId_ = isoSurfaceActors_.size() - 1;
  else 
    currentIsoSurfaceId_ += shift;
  
  cout << "[UserInterface] Current isosurface is #" << currentIsoSurfaceId_
    << endl;
  
  return 0;
}


int UserInterface::deleteCurrentIsoSurface() {

  if(!isoSurfaceActors_.size()) 
    return -1;
  
  int newIsoSurfaceId = editor_->deleteIsoSurface(currentIsoSurfaceId_);
  
  // remove the actor from the renderer
  renderer_->RemoveActor(isoSurfaceActors_[currentIsoSurfaceId_]);
  renderWindow_->Render();
  
  // now delete the associated data
  isoSurfaceActors_[currentIsoSurfaceId_]->Delete();
  isoSurfaceActors_.erase(isoSurfaceActors_.begin() + currentIsoSurfaceId_);
  isoSurfaceMappers_[currentIsoSurfaceId_]->Delete();
  isoSurfaceMappers_.erase(isoSurfaceMappers_.begin() + currentIsoSurfaceId_);
  isTransparent_.erase(isTransparent_.begin() + currentIsoSurfaceId_);
  
  currentIsoSurfaceId_ = newIsoSurfaceId;
  
  return 0;
}

int UserInterface::moveCurrentIsoSurface(const bool &upOnOff, 
  const bool &fastOnOff) {
 
  if(currentIsoSurfaceId_ == -1)
    return 0; 
  
  double range[2];
  editor_->getScalarRange(range[0], range[1]);
  
  double shift = ISOSURFACE_SHIFT*(range[1] - range[0]);
  
  if(!upOnOff)
    shift = -shift;
  
  if(fastOnOff)
    shift *= 10;
  
  editor_->moveIsoSurface(currentIsoSurfaceId_, shift, useVtk_);
  
  // update the interface
    
  // update the isosurface render helpers
  isoSurfaceMappers_[currentIsoSurfaceId_]->SetInputData(
    editor_->getIsoSurface(currentIsoSurfaceId_));
  isoSurfaceActors_[currentIsoSurfaceId_]->SetMapper(
    isoSurfaceMappers_[currentIsoSurfaceId_]);
  
  renderWindow_->Render();
    
  return 0;
}

int UserInterface::run(){

  cout << "[UserInterace] Initializing user interface..." << endl;

  // TODO
  
  cout << "[UserInterface] Running user interface!" << endl;
  
  return 0;
}

int UserInterface::switchCurrentIsoSurfaceTransparency(){

  if((currentIsoSurfaceId_ < 0)
    ||(currentIsoSurfaceId_ >= (int) isTransparent_.size()))
    return -1;
  
  if(!isTransparent_[currentIsoSurfaceId_]){ 
    isoSurfaceActors_[
      currentIsoSurfaceId_]->GetProperty()->SetOpacity(ISOSURFACE_ALPHA);
    isTransparent_[currentIsoSurfaceId_] = true;
  }
  else{
    isoSurfaceActors_[currentIsoSurfaceId_]->GetProperty()->SetOpacity(1);
    isTransparent_[currentIsoSurfaceId_] = false;
  }
  
  renderWindow_->Render();
  
  return 0;
}
