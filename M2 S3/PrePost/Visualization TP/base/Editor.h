/*
  Editor.H: Data-structures and processing.
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


#ifndef EDITOR_H
#define EDITOR_H

#include  <Debug.h>
#include  <IsoSurfacer.h>

#include  <vtkContourGrid.h>
#include  <vtkPointData.h>
#include  <vtkPolyData.h>
#include  <vtkUnstructuredGrid.h>
#include  <vtkXMLUnstructuredGridReader.h>

class Editor{

  
  public:
      
    Editor();
      
    Editor(const Editor& other);
    
    virtual ~Editor();
    
    virtual Editor& operator=(const Editor& other);
    
    virtual bool operator==(const Editor& other) const;
 
    int addIsoSurface(const double &isoValue, const bool &useVtk = true);
   
    int deleteIsoSurface(const int &isoSurfaceId);
    
    vtkDataSet* getData() const { return inputMesh_;};

    vtkPolyData* getIsoSurface(const int &isoSurfaceId) const;
    
    inline const double getIsoValue(const int &isoSurfaceId) const{
      if((isoSurfaceId < 0)||(isoSurfaceId >= (int) isoValueList_.size()))
        return -INT_MAX;
      
      return isoValueList_[isoSurfaceId];
    };
    
    inline int getScalarRange(double &min, double &max) const{
      
      double vtkRet[2];
      
      inputMesh_->GetScalarRange(vtkRet);
      min = vtkRet[0];
      max = vtkRet[1];
      
      return 0;
    };
    
    int loadInputMesh(const string &fileName);
    
    int moveIsoSurface(const int &isoSurfaceId, const double &shift, 
      const bool &useVtk = true);
    
    inline int setCustomImplementation(const extractionType &type){
      cout << "[Editor] ";
      switch(type){
        case SIMPLE:
          cout << "Simple ";
          break;
        case STANDARD:
          cout << "Standard ";
          break;
        case FAST:
          cout << "Fast ";
          break;
      };
      cout << "custom implementation selected." << endl;
      type_ = type;
      return 0;
    };
    
  protected:
  
    IsoSurfacer* extractIsoSurface(const double &isoValue);
    
    vtkContourGrid* extractIsoSurfaceVTK(const double &isoValue); 
    
    extractionType                type_;
    vtkUnstructuredGrid           *inputMesh_;
    vtkXMLUnstructuredGridReader  *meshReader_;
    vector<vtkContourGrid *>      vtkIsoSurfacers_;
    vector<IsoSurfacer *>         isoSurfacers_;
    vector<double>                isoValueList_;
    vector<vtkPolyData *>         isoSurfaceList_;
    
    // traversal speed-up
    vector<vector<vtkIdType> >    tetNeighbors_;
    
    // seeds speed up
    TetIndex                      tetIndex_;
};

#endif // EDITOR_H
