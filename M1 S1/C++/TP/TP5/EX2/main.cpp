#include <iostream>

template <typename T, size_t Dimension>
class Point{
private:
    std::array<T,Dimension> coor;

public:
    Point() {
        coor = 
    }

    void setcoor(size_t index, T vaule) {
        corr[index] = vaule;
    }

    T getcoor(size_t index) {
        return coor[index];
    }

    void printcoor() {
        std::cout << "(" ;
        for(int i = 0; i < Dimension; ++i) {
            std::cout << corr[i];
            if(i < Dimension - 1)
                std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
}

