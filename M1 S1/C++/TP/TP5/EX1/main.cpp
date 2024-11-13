#include <iostream>

template <typename T>
T sumArray(T* array, int size) {
    T sum = T();
    for(int i = 0; i < size; ++i){
        sum += array[i];
    }
    return sum;
}

template <>
std::string sumArray(std::string* array, int size){
    std::string sum;
    for(int i = 0; i < size; ++i){
        sum += array[i];
        sum += " ";
    }
    return sum;
}

int main() {
    int intArray[] = {1, 2, 3, 4, 5};
    int intSize = sizeof(intArray) / sizeof(intArray[0]);
    std::cout << "intArray: " << sumArray(intArray, intSize) << std::endl;

    double doubleArray[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    int doubleSize = sizeof(doubleArray) / sizeof(doubleArray[0]);
    std::cout << "doubleArray: " << sumArray(doubleArray, doubleSize) << std::endl;

    std::string strArray[] = {"hello", "bonjour", "Hallo"};
    int strSize = sizeof(strArray) / sizeof(strArray[0]);
    std::cout << "string: " << sumArray(strArray, strSize) << std::endl;
    return 0;
}