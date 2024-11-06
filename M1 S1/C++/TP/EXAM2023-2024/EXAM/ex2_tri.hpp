
void triParSelection( Personne** tab, int n )
{
  for ( int k=0;k<n;++k )
    {
      Personne * pmin = tab[k];
      int idmin = k;
      for( int l=k+1;l<n;++l)
        {
          Personne * ptest = tab[l];
          if ( (*ptest)<(*pmin) )
            {
              pmin = ptest;
              idmin = l;
            }
        }
      if ( idmin != k )
        {
          Personne * tmp = tab[k];
          tab[k] = tab[idmin];
          tab[idmin] = tmp;
        }
    }
}
