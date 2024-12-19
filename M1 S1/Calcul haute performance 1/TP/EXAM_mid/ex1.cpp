#include <mpi.h>
#include <iostream>
#include <cmath>
#include "randomnumber.hpp" // 假设提供 uniform() 函数生成[0,1]均匀分布随机数

class Vecteur3d
{
public :
    Vecteur3d( double x=0, double y=0, double z=0 )
    {
        M_val[0] = x;
        M_val[1] = y;
        M_val[2] = z;
    }
    Vecteur3d( Vecteur3d const& v )
    {
        M_val[0] = v.M_val[0];
        M_val[1] = v.M_val[1];
        M_val[2] = v.M_val[2];
    }

    double operator()(int i) const { return M_val[i]; }
    double& operator()(int i) { return M_val[i]; }
    double operator[](int i) const { return M_val[i]; }
    double& operator[](int i) { return M_val[i]; }

    double dot( Vecteur3d const& v ) const
    {
        double res=0;
        for (int d=0;d<3;++d )
            res += M_val[d]*v(d);
        return res;
    }

    double norm() const { return std::sqrt( this->dot( *this ) ); }

    void normalize()
    {
        double thenorm = this->norm();
        for (int d=0;d<3;++d )
            M_val[d] /= thenorm;
    }

private :
    double M_val[3];

};

Vecteur3d operator+( Vecteur3d const& u, Vecteur3d const& v )
{
    return Vecteur3d(u(0)+v(0),u(1)+v(1),u(2)+v(2));
}
Vecteur3d operator-( Vecteur3d const& u, Vecteur3d const& v )
{
    return Vecteur3d(u(0)-v(0),u(1)-v(1),u(2)-v(2));
}

Vecteur3d operator*( Vecteur3d const& u, double v ) { return Vecteur3d(u(0)*v,u(1)*v,u(2)*v); }
Vecteur3d operator*( double v, Vecteur3d const& u ) { return u*v; }
Vecteur3d operator/( Vecteur3d const& u, double v ) { return Vecteur3d(u(0)/v,u(1)/v,u(2)/v); }

std::ostream &
operator<<( std::ostream & o, Vecteur3d const& u ) { return o << "(" << u(0) << "," << u(1) << "," << u(2) << ")"; }

int main( int argc ,char ** argv)
{
    MPI_Init( &argc, &argv );
    int worldSize, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &worldSize );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // 定义已知参数
    Vecteur3d U(-0.801784, 0.267261, 0.534522);
    Vecteur3d V(-0.295128, -0.954827, 0.034721);
    Vecteur3d C(3.91923, 1.53929, 0.0446184);
    double R=0.025;
    Vecteur3d D(4,2,-1);
    Vecteur3d N(8,-2,13);
    double a=8,b=-2,c=13,d=-15; // 平面方程参数

    // 光源，仅在 rank 0 定义
    Vecteur3d O;
    if ( rank == 0 )
        O = Vecteur3d(3.9712, 1.5263, 0.129062);

    // 从命令行读取 Nrayon (仅 rank 0)
    int Nrayon = 0;
    if (rank == 0) {
        if (argc > 1) {
            Nrayon = atoi(argv[1]);
        } else {
            std::cerr << "Usage: mpiexec -n <procs> ./ex1 <Nrayon>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    // 广播 Nrayon 分配方案
    // 由于其他进程不会直接知道Nrayon，我们由0号进程计算每个进程的任务数并发送
    int localCount = 0;
    if (rank == 0) {
        // 简单均分。对于不能整除的情况，最后一个进程承担剩余任务。
        int base = Nrayon / worldSize;
        int rem = Nrayon % worldSize;
        // 将分配信息存于一个数组，然后发送给各进程
        int *counts = new int[worldSize];
        for (int i=0; i<worldSize; ++i) {
            counts[i] = base + (i < rem ? 1 : 0);
        }

        // 向每个进程发送任务数
        for (int i=1; i<worldSize; ++i) {
            MPI_Send(&counts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        localCount = counts[0];
        delete[] counts;
    } else {
        MPI_Recv(&localCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 定义Vecteur3d的MPI派生类型
    MPI_Datatype MPI_Vecteur3d;
    {
        // Vecteur3d内部是三个double连续存储
        MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_Vecteur3d);
        MPI_Type_commit(&MPI_Vecteur3d);
    }

    // 广播O给所有进程
    MPI_Bcast(&O, 1, MPI_Vecteur3d, 0, MPI_COMM_WORLD);

    // 每个进程根据localCount生成光线并计算与平面的交点
    // 生成光线步骤回顾：
    // 1. 随机产生 epsilon1, epsilon2 in [-R, R]
    // 2. P = epsilon1*U + epsilon2*V
    // 3. 若 ||P|| < R 则接受，否则重抽
    // 4. T = C + P
    // 5. 光线为OT，通过 O 和 T
    //
    // 求交点:
    // I = O + alpha * OT
    // alpha = -<O - D, N> / <OT, N>

    double sumX=0.0, sumY=0.0, sumZ=0.0; 
    // 用于计算重心的局部和
    int countValid = localCount; // 实际本地处理的光线数（都应有交点）

    // 初始化随机数生成器（假设 randomnumber.hpp 有初始化种子函数）
    RandomNumber<double> rng(-R, R); // 每个进程用不同种子初始化随机数

    // 先计算固定的分子与分母不变项以提高效率？
    // 不过 OT 不同光线不一样，所以还是对每条光线算
    Vecteur3d O_minus_D = O - D;
    double numerator = -(O_minus_D.dot(N));

    for (int i=0; i<localCount; ++i) {
        // 产生可接受的P
        Vecteur3d P;
        while (true) {
            double epsilon1 = rng(); // uniform()假设返回[0,1]之间随机数
            double epsilon2 = rng();
            P = U*epsilon1 + V*epsilon2;
            if (P.norm() < R) break;
        }

        Vecteur3d T = C + P;
        Vecteur3d OT = T - O; // 光线方向向量

        double denominator = OT.dot(N);
        double alpha = numerator / denominator;
        Vecteur3d I = O + OT*alpha; // 交点

        sumX += I(0);
        sumY += I(1);
        sumZ += I(2);
    }

    // 现在求所有点重心：先求全局和，然后除以总的光线数Nrayon
    double globalSumX=0.0, globalSumY=0.0, globalSumZ=0.0;
    MPI_Reduce(&sumX, &globalSumX, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumY, &globalSumY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumZ, &globalSumZ, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double baryX=0.0, baryY=0.0, baryZ=0.0;
    if (rank == 0) {
        baryX = globalSumX / Nrayon;
        baryY = globalSumY / Nrayon;
        baryZ = globalSumZ / Nrayon;
    }

    // 广播重心给所有进程
    MPI_Bcast(&baryX, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&baryY, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&baryZ, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Vecteur3d barycenter(baryX, baryY, baryZ);

    // 计算最小包围半径：对每个交点距离barycenter的距离求最大值
    // 我们不再重复计算交点，但可以重头再算一次或者在第一次就把交点都存下来？
    // 为了节省内存，Nrayon很大时存所有点不划算，所以我们再次遍历光线 (理论上不合适，但题目没特别要求性能)
    // 更好的方法是：第一次计算交点的时候就顺便计算其与barycenter距离的最大值（并存局部最大），但这要求知道barycenter。
    // barycenter必须在所有点计算完后才知道，所以必须进行两次过程。
    //
    // 简化起见，我们此处为了满足要求，只好再计算一次交点集合与barycenter的距离最大值。
    // （实际中应一次性存储在内存里避免重复计算，但题意并无明确性能要求。）

    // 再来一遍生成光线与交点，但这次只计算半径相关
    // 真实项目会有内存开销和性能问题，但这里为完成要求先这样实现。
    MPI_Barrier(MPI_COMM_WORLD); // 确保重心广播完成

    double localMaxDist = 0.0;
    // 再计算一次
    RandomNumber<double> rng2(-R, R); // 保持同样的随机序列
    for (int i=0; i<localCount; ++i) {
        Vecteur3d P;
        while (true) {
            double epsilon1 = rng2();
            double epsilon2 = rng2();
            P = U*epsilon1 + V*epsilon2;
            if (P.norm() < R) break;
        }
        Vecteur3d T = C + P;
        Vecteur3d OT = T - O;
        double denominator = OT.dot(N);
        double alpha = -( (O - D).dot(N) ) / denominator;
        Vecteur3d I = O + OT*alpha; // 交点

        double dist = (I - barycenter).norm();
        if (dist > localMaxDist) localMaxDist = dist;
    }

    double globalMaxDist=0.0;
    MPI_Reduce(&localMaxDist, &globalMaxDist, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 输出结果（仅0号进程）
    if (rank == 0) {
        std::cout << "Barycenter: (" << baryX << "," << baryY << "," << baryZ << ")" << std::endl;
        std::cout << "Minimum radius: " << globalMaxDist << std::endl;
    }

    MPI_Type_free(&MPI_Vecteur3d);
    MPI_Finalize();
    return 0;
}
