#include <iostream>

class Vecteur {
private:
    double* data;
    int size;
    int capacity;

    // Helper function to resize the array when needed
    void resize(int new_capacity) {
        if (new_capacity <= capacity)
            return;

        double* new_data = new double[new_capacity];
        for (int i = 0; i < size; ++i)
            new_data[i] = data[i];

        // Initialize new entries to zero
        for (int i = size; i < new_capacity; ++i)
            new_data[i] = 0.0;

        delete[] data;
        data = new_data;
        capacity = new_capacity;
    }

public:
    // Constructor with size, initializes array to zero
    Vecteur(int size_) : size(size_), capacity(size_) {
        data = new double[capacity];
        for (int i = 0; i < size; ++i)
            data[i] = 0.0;
    }

    // Destructor
    ~Vecteur() {
        delete[] data;
    }

    // Copy constructor
    Vecteur(const Vecteur& other) : size(other.size), capacity(other.capacity) {
        data = new double[capacity];
        for (int i = 0; i < size; ++i)
            data[i] = other.data[i];
    }

    // Assignment operator (only if sizes are identical)
    Vecteur& operator=(const Vecteur& other) {
        if (this == &other)
            return *this;

        if (size != other.size) {
            std::cerr << "Error: Sizes are not identical for assignment." << std::endl;
            return *this;
        }

        for (int i = 0; i < size; ++i)
            data[i] = other.data[i];

        return *this;
    }

    // Returns the size of the vector
    int getSize() const {
        return size;
    }

    // Sets all values of the vector to 'd'
    void setConstant(double d) {
        for (int i = 0; i < size; ++i)
            data[i] = d;
    }

    // Sets all values of the vector to zero
    void zero() {
        setConstant(0.0);
    }

    // Computes the sum of all elements in the vector
    double sum() const {
        double total = 0.0;
        for (int i = 0; i < size; ++i)
            total += data[i];
        return total;
    }

    // Access and modify entries using operator[]
    double& operator[](int index) {
        if (index < 0 || index >= size) {
            std::cerr << "Error: Index out of bounds in operator[]" << std::endl;
            exit(EXIT_FAILURE);
        }
        return data[index];
    }

    // Access entries using operator[] (const version)
    const double& operator[](int index) const {
        if (index < 0 || index >= size) {
            std::cerr << "Error: Index out of bounds in operator[]" << std::endl;
            exit(EXIT_FAILURE);
        }
        return data[index];
    }

    // Access and modify entries using operator()
    double& operator()(int index) {
        return (*this)[index];
    }

    // Access entries using operator() (const version)
    const double& operator()(int index) const {
        return (*this)[index];
    }

    // Operator +=
    Vecteur& operator+=(const Vecteur& other) {
        if (size != other.size) {
            std::cerr << "Error: Sizes are not identical for operator+=" << std::endl;
            return *this;
        }

        for (int i = 0; i < size; ++i)
            data[i] += other.data[i];

        return *this;
    }

    // Operator -=
    Vecteur& operator-=(const Vecteur& other) {
        if (size != other.size) {
            std::cerr << "Error: Sizes are not identical for operator-=" << std::endl;
            return *this;
        }

        for (int i = 0; i < size; ++i)
            data[i] -= other.data[i];

        return *this;
    }

    // Operator +
    Vecteur operator+(const Vecteur& other) const {
        if (size != other.size) {
            std::cerr << "Error: Sizes are not identical for operator+" << std::endl;
            return Vecteur(0);
        }

        Vecteur result(size);
        for (int i = 0; i < size; ++i)
            result.data[i] = data[i] + other.data[i];

        return result;
    }

    // Operator -
    Vecteur operator-(const Vecteur& other) const {
        if (size != other.size) {
            std::cerr << "Error: Sizes are not identical for operator-" << std::endl;
            return Vecteur(0);
        }

        Vecteur result(size);
        for (int i = 0; i < size; ++i)
            result.data[i] = data[i] - other.data[i];

        return result;
    }

    // Adds an element at the end of the vector
    void push_back(double d) {
        if (size >= capacity) {
            int new_capacity = (capacity == 0) ? 1 : capacity * 2;
            resize(new_capacity);
        }
        data[size++] = d;
    }

    // Friend function for inner product
    friend double inner_product(const Vecteur& v1, const Vecteur& v2);
};

// Computes the inner product of two vectors
double inner_product(const Vecteur& v1, const Vecteur& v2) {
    if (v1.size != v2.size) {
        std::cerr << "Error: Sizes are not identical for inner_product" << std::endl;
        return 0.0;
    }

    double result = 0.0;
    for (int i = 0; i < v1.size; ++i)
        result += v1.data[i] * v2.data[i];

    return result;
}

// Test the Vecteur class
int main() {
    // Create a vector of size 5
    Vecteur vec1(5);
    vec1.setConstant(2.0);

    // Display the sum of vec1
    std::cout << "Sum of vec1: " << vec1.sum() << std::endl;

    // Access and modify entries using operator[]
    vec1[2] = 10.0;
    std::cout << "vec1[2] after modification: " << vec1[2] << std::endl;

    // Create another vector and test operator+
    Vecteur vec2(5);
    vec2.setConstant(3.0);
    Vecteur vec3 = vec1 + vec2;

    // Display the sum of vec3
    std::cout << "Sum of vec3 (vec1 + vec2): " << vec3.sum() << std::endl;

    // Compute the inner product
    double ip = inner_product(vec1, vec2);
    std::cout << "Inner product of vec1 and vec2: " << ip << std::endl;

    // Test the push_back functionality
    Vecteur vec4(0);
    vec4.push_back(1.0);
    vec4.push_back(2.0);
    vec4.push_back(3.0);
    vec4.push_back(4.0);

    // Display the elements of vec4
    std::cout << "Elements of vec4 after push_back operations: ";
    for (int i = 0; i < vec4.getSize(); ++i)
        std::cout << vec4[i] << " ";
    std::cout << std::endl;

    return 0;
}
