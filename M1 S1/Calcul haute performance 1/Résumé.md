```cpp
int MPI_Send ( const void * buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm )

int MPI_Recv ( void * buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Status * status )

int MPI_Sendrecv ( const void * sendbuf , int sendcount , MPI_Datatype sendtype , int dest , int sendtag , void * recvbuf , int recvcount , MPI_Datatype recvtype , int source , int recvtag , MPI_Comm comm , MPI_Status * status )
```

