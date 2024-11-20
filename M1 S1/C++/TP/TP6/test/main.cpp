#include <iostream>
#include <vector>


int main() {
    std::vector <int> v = {32 ,71 ,12 ,45 ,26 ,80 ,53 ,33};
    std::vector <int >:: const_iterator it = v.begin ();
    std::vector <int >:: const_iterator en = v.end ();
    for ( ; it!=en; ++it )
        std :: cout << *it << "\n";

    std :: vector <int >:: iterator it2 = v.begin ();
    std :: vector <int >:: iterator en2 = v.end ();
    for ( ; it2 != en2; ++ it2 )
        *it2 = 23;
}