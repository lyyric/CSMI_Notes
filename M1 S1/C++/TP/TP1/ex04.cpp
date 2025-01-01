#include<iostream>

int main()
{
    std::cout << "loop for :\n";
    for(int i=8;i<=23;i++)
        std::cout<<i<<std::endl;

    std::cout << "loop for with even number :\n";
    for(int i=8;i<=23;i++)
        if ((i%2)==0)
            std::cout<<i<<std::endl;

    std::cout << "loop while :\n";
    int k=8;
    while( k <= 23 )
    std::cout << k++ << std::endl;
    std::cout << "loop while with even number :\n";
    k=8;
    while( k <= 23 )
    {
    if ((k%2)==0)
        std::cout<<k<<std::endl;
    ++k;
    }
    return 0;
}
