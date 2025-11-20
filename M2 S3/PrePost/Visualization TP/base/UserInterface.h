/*
  UserInterface.h: Interactions and rendering.
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


#ifndef USERINTERFACE_H
#define USERINTERFACE_H

#include  <vtkGeometryFilter.h>
#include  <vtkInteractorStyleSwitch.h>
#include  <vtkInteractorStyleTrackballCamera.h>
#include  <vtkObjectFactory.h>
#include  <vtkPointData.h>
#include  <vtkPolyDataMapper.h>
#include  <vtkProperty.h>
#include  <vtkRenderer.h>
#include  <vtkRenderWindow.h>
#include  <vtkRenderWindowInteractor.h>

#include  "Editor.h"

#define   ISOSURFACE_SHIFT  0.005
#define   ISOSURFACE_ALPHA  0.6

class UserInterface;

class CustomInteractor : public vtkInteractorStyleTrackballCamera{

  public:
    
    static CustomInteractor* New();
    vtkTypeMacro(CustomInteractor, vtkInteractorStyleTrackballCamera);
    
    virtual void OnKeyPress();
    
    inline int setUserInterface(UserInterface* userInterface){
      
      userInterface_ = userInterface;
      return 0;
    };
  
  protected:
    
    UserInterface  *userInterface_;
};

class UserInterface {

  public:
    
    UserInterface();
    
    UserInterface ( const UserInterface& other );
    
    virtual ~UserInterface();
    
    virtual UserInterface& operator= ( const UserInterface& other );
    
    virtual bool operator== ( const UserInterface& other ) const;
    
    int addIsoSurface();
  
    int changeIsoSurface(const int &shift);
    
    int deleteCurrentIsoSurface();
    
    const bool getVtk() const {
      return useVtk_;
    }
    
    int moveCurrentIsoSurface(const bool &upOnOff, 
      const bool &fastOnOff = false);
    
    int run();
    
    inline int setEditor(Editor *editor){
      editor_ = editor;
      return 0;
    };
    
    inline int setVtk(const bool &onOff){
      cout << "[UserInterface] Using "
        << (onOff ? "VTK": "custom") << " implementation..." << endl;
      useVtk_ = onOff;
      return 0;
    }
    
    inline int setCustomImplementation(const extractionType &type){
      return editor_->setCustomImplementation(type);
    };
    
    int switchCurrentIsoSurfaceTransparency();
    
  protected:
    
    bool                          useVtk_;
    int                           currentIsoSurfaceId_;
    Editor                        *editor_;
    vtkActor                      *mainActor_;
    vtkGeometryFilter             *geometryFilter_;
    vtkPolyDataMapper             *boundaryMapper_;
    vtkRenderer                   *renderer_;
    vtkRenderWindow               *renderWindow_;
    CustomInteractor              *customInteractor_;
    vtkRenderWindowInteractor     *interactor_;
    vector<vtkActor *>            isoSurfaceActors_;
    vector<vtkPolyDataMapper *>   isoSurfaceMappers_;
    vector<bool>                  isTransparent_;
};

#endif // USERINTERFACE_H
