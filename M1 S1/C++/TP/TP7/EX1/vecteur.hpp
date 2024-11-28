#ifndef VECTEUR_HPP
#define VECTEUR_HPP

#include <cstddef>
#include <stdexcept>

template <typename T>
class Vecteur {
public:
    Vecteur();
    Vecteur(std::size_t size);
    ~Vecteur();

    T& operator[](std::size_t index);
    const T& operator[](std::size_t index) const;

    std::size_t size() const;
    void push_back(T const& v);
    void resize(std::size_t newSize);
    T& front();
    T& back();

private:
    T* data_;
    std::size_t size_;
    std::size_t capacity_;

    void reallocate(std::size_t newCapacity);
};

#endif // VECTEUR_HPP
