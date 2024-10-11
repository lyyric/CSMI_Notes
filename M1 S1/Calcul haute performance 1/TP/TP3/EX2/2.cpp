#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib> // 用于 rand()
#include <ctime>   // 用于 time()

// 定义 Point 类
class Point {
public:
    Point() : id(0) {
        coords[0] = coords[1] = coords[2] = 0.0;
    }

    Point(double x, double y, double z, int identifier) : id(identifier) {
        coords[0] = x;
        coords[1] = y;
        coords[2] = z;
    }

    // 获取坐标
    const double* getCoords() const { return coords; }

    // 获取标识符
    int getId() const { return id; }

    // 初始化随机坐标
    void initRandom(int identifier) {
        coords[0] = static_cast<double>(rand()) / RAND_MAX;
        coords[1] = static_cast<double>(rand()) / RAND_MAX;
        coords[2] = static_cast<double>(rand()) / RAND_MAX;
        id = identifier;
    }

    // 计算到参考点的距离
    double distanceTo(const double P[3]) const {
        double dx = coords[0] - P[0];
        double dy = coords[1] - P[1];
        double dz = coords[2] - P[2];
        return sqrt(dx*dx + dy*dy + dz*dz);
    }

    // 创建 MPI 派生类型
    MPI_Datatype create_MPI_Datatype() const {
        MPI_Datatype newtype;
        const int nItems = 2;
        int blockLengths[nItems] = {3, 1};
        MPI_Datatype types[nItems] = {MPI_DOUBLE, MPI_INT};
        MPI_Aint offsets[nItems];

        MPI_Aint baseAddress;
        MPI_Get_address(this, &baseAddress);
        MPI_Get_address(&coords, &offsets[0]);
        MPI_Get_address(&id, &offsets[1]);

        for (int i = 0; i < nItems; ++i) {
            offsets[i] -= baseAddress;
        }

        MPI_Type_create_struct(nItems, blockLengths, offsets, types, &newtype);
        MPI_Type_commit(&newtype);

        return newtype;
    }

private:
    double coords[3]; // x, y, z 坐标
    int id;           // 唯一标识符
};

int main(int argc, char* argv[]) {
    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);

    // 获取进程的秩和总数
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 定义参考点 P = (0.5, 0.5, 0.5)
    const double P[3] = {0.5, 0.5, 0.5};

    // 每个进程的点数 N，可以根据需要修改或从命令行参数获取
    const int N = 1000;

    // 初始化随机数种子
    srand(time(NULL) + rank); // 加上 rank 使得每个进程的种子不同

    // 定义 Point 的 MPI 数据类型
    Point tempPoint; // 临时对象用于创建 MPI 数据类型
    MPI_Datatype MPI_POINT = tempPoint.create_MPI_Datatype();

    // 进程 0 创建点云
    std::vector<Point> pointCloud;
    if (rank == 0) {
        int totalPoints = N * size; // 总点数
        pointCloud.resize(totalPoints);

        // 随机生成点云数据
        for (int i = 0; i < totalPoints; ++i) {
            pointCloud[i].initRandom(i);
        }
    }

    // 每个进程接收 N 个点
    std::vector<Point> localPoints(N);

    // 使用 MPI_Scatter 将点云分发给所有进程
    MPI_Scatter(
        pointCloud.data(), N, MPI_POINT,    // 发送缓冲区
        localPoints.data(), N, MPI_POINT,   // 接收缓冲区
        0, MPI_COMM_WORLD                   // 根进程和通信器
    );

    // 每个进程寻找其本地最近的点
    double minDist = -1.0;
    int localClosestId = -1;
    double localClosestCoords[3];

    for (int i = 0; i < N; ++i) {
        double dist = localPoints[i].distanceTo(P);
        if (minDist < 0 || dist < minDist) {
            minDist = dist;
            localClosestId = localPoints[i].getId();
            const double* coords = localPoints[i].getCoords();
            localClosestCoords[0] = coords[0];
            localClosestCoords[1] = coords[1];
            localClosestCoords[2] = coords[2];
        }
    }

    // 定义结构体来发送最小距离和对应的点信息
    struct ClosestPointInfo {
        double distance;
        int id;
        double coords[3];
    } localInfo, globalInfo;

    localInfo.distance = minDist;
    localInfo.id = localClosestId;
    localInfo.coords[0] = localClosestCoords[0];
    localInfo.coords[1] = localClosestCoords[1];
    localInfo.coords[2] = localClosestCoords[2];

    // 使用 MPI_Reduce 找到全局最近的点
    MPI_Datatype MPI_CLOSEST_POINT_INFO;
    int blockLengths[3] = {1, 1, 3};
    MPI_Aint offsets[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_DOUBLE};

    MPI_Aint baseAddress;
    MPI_Get_address(&localInfo, &baseAddress);
    MPI_Get_address(&localInfo.distance, &offsets[0]);
    MPI_Get_address(&localInfo.id, &offsets[1]);
    MPI_Get_address(&localInfo.coords, &offsets[2]);

    for (int i = 0; i < 3; ++i) {
        offsets[i] -= baseAddress;
    }

    MPI_Type_create_struct(3, blockLengths, offsets, types, &MPI_CLOSEST_POINT_INFO);
    MPI_Type_commit(&MPI_CLOSEST_POINT_INFO);

    // 自定义比较函数用于 MPI_Reduce
    MPI_Op minOp;
    MPI_Op_create([](void* a, void* b, int* len, MPI_Datatype* dtype) {
        ClosestPointInfo* infoA = static_cast<ClosestPointInfo*>(a);
        ClosestPointInfo* infoB = static_cast<ClosestPointInfo*>(b);
        if (infoA->distance < infoB->distance) {
            *infoB = *infoA;
        }
    }, 1, &minOp);

    MPI_Reduce(&localInfo, &globalInfo, 1, MPI_CLOSEST_POINT_INFO, minOp, 0, MPI_COMM_WORLD);

    // 释放自定义操作和类型
    MPI_Op_free(&minOp);
    MPI_Type_free(&MPI_CLOSEST_POINT_INFO);

    // 进程 0 输出全局最近的点信息
    if (rank == 0) {
        std::cout << "最近点的标识符: " << globalInfo.id << std::endl;
        std::cout << "坐标: (" << globalInfo.coords[0] << ", "
                  << globalInfo.coords[1] << ", " << globalInfo.coords[2] << ")" << std::endl;
        std::cout << "到参考点的距离: " << globalInfo.distance << std::endl;
    }

    // 清理 MPI 派生类型
    MPI_Type_free(&MPI_POINT);

    // 结束 MPI 环境
    MPI_Finalize();
    return 0;
}
