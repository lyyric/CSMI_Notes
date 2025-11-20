/*
  IsoSurfacer.h: Isosurface computation class.
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


#ifndef ISOSURFACER_H
#define ISOSURFACER_H

#include  <map>
#include  <queue>

#include  <Debug.h>

#include  <vtkAlgorithm.h>
#include  <vtkCellArray.h>
#include  <vtkDoubleArray.h>
#include  <vtkObjectFactory.h>
#include  <vtkPointData.h>
#include  <vtkPolyData.h>
#include  <vtkUnstructuredGrid.h>

enum extractionType {SIMPLE, STANDARD, FAST};

class TetIndex{
 
  public:
    
    TetIndex(){
    };
    
  protected:
    
};

class EdgeIntersection {
  
  public:
    
};

class VTK_EXPORT IsoSurfacer : public vtkAlgorithm {
  
  public:
   
    static IsoSurfacer* New();
    // vtkTypeRevisionMacro(IsoSurfacer, vtkObject);
    
    vtkSetMacro(Value, double);
    vtkSetMacro(Input, vtkUnstructuredGrid *);
    vtkSetMacro(Type, extractionType);
    
    vtkGetMacro(Output, vtkPolyData *);
   
    void Update();
  
  protected:
    
    IsoSurfacer();
    ~IsoSurfacer();
    
    inline vector<double> ComputeEdgeIntersection(
      pair<vtkIdType, vtkIdType> &edge) const{
    
      vector<double> p(3);
      
      return p;
    };
    
    int ComputePartialIntersection(const int &tetId);
      
    int ComputeSimpleIntersection(vtkCell *tet);
   
    int FastExtraction(); 
    
    inline bool IsCellOnLevelSet(vtkCell *cell) const{
      
      std::cout << "Number of points in the cell = " << cell->GetNumberOfPoints() << std::endl;

      // vtkIdList *ids = cell->GetPointIds();
      // vtkDoubleArray *scalarsCell;
      // scalarField_->GetTuples(ids, scalarsCell);
      // std::cout << "Scalar 1 = " << scalarsCell[0] << std::endl;
      
      // std::cout << "Value = " << Value << std::endl;

      double scalar1 = scalarField_->GetTuple(cell->GetPointId(0))[0];
      double scalar2 = scalarField_->GetTuple(cell->GetPointId(1))[0];
      double scalar3 = scalarField_->GetTuple(cell->GetPointId(2))[0];
      double scalar4 = scalarField_->GetTuple(cell->GetPointId(3))[0];

      // std::cout << "Scalar 1 = " << scalar1 << std::endl;
      // std::cout << "Scalar 2 = " << scalar2 << std::endl;
      // std::cout << "Scalar 3 = " << scalar3 << std::endl;
      // std::cout << "Scalar 4 = " << scalar4 << std::endl;

      double min = scalar1;
      if(scalar2<min) min = scalar2;
      if(scalar3<min) min = scalar3;
      if(scalar4<min) min = scalar4;
      double max = scalar1;
      if(scalar2>max) max = scalar2;
      if(scalar3>max) max = scalar3;
      if(scalar4>max) max = scalar4;

      // std::cout << "Min = " << min << std::endl;
      // std::cout << "Max = " << max << std::endl;

      return (Value >= min) && (Value<=max);
    };

    int ReOrderTetEdges(vector<pair<vtkIdType, vtkIdType> > &edgeList) const;
    
    int SimpleExtraction();
    
    int StandardExtraction();
    
    // variables for get/set macros
    vtkUnstructuredGrid               *Input;
    vtkPolyData                       *Output;
    extractionType                    Type;
    double                            Value;
    
    // internal variables
    vtkPoints                         *pointSet_;
    vtkCellArray                      *cellArray_;
    vtkDoubleArray                    *fakeScalars_;
    
    // speed up
    vtkDataArray                      *scalarField_;
    
  private:
    // not implemented
    
    IsoSurfacer(const IsoSurfacer&);
    
    void operator=(const IsoSurfacer &);
};

#endif // ISOSURFACER_H
