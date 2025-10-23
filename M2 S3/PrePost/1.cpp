#include <set>
#include <gmsh.h>
int main(int argc, char **argv)
{
  gmsh::initialize();

  gmsh::model::add("square");
  gmsh::model::geo::addPoint(.1, 0, 0, lc, 2);
  
  roundedRadius = 0;
  tag = -1;
  gmsh::model::geo::addRectangle(-0.5, -0.5, -0.5, 1, 1, tag, roundedRadius);

  gmsh::model::geo::synchronize();
  gmsh::model::addPhysicalGroup(2, {1}, -1, "My surface");

  std::set<std::string> args(argv, argv + argc);
  if(!args.count("-nopopup")) gmsh::fltk::run();
  
  gmsh::finalize();
  return 0;
}