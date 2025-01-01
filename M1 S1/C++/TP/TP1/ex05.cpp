#include <iostream>

int main()
{
  int nFibonacciNumber;
  std::cout << "Tapez la longueur de la suite : "; std::cin >> nFibonacciNumber;
  if ( nFibonacciNumber < 2 )
    {
      std::cout <<"Doit etre supèrieur à 1! Exit!";
      return EXIT_SUCCESS;
    }
      
  int nm2=0,nm1=1;
  std::cout<< "Suite de Fibonacci : " << nm2 << " " << nm1;
  for (int k=2;k<=nFibonacciNumber;++k)
    {
      int n = nm2+nm1;
        std::cout << " " << n;
        nm2 = nm1;
        nm1 = n;
    }
  std::cout << "\n";
  std::cout << "Appuyez sur une touche pour continuer." << std::endl;
  std::cin.ignore();
  std::cin.get();

  return EXIT_SUCCESS;
}
