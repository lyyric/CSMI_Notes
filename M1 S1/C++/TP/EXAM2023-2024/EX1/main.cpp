#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define MAX_SIZE 10000


class Matrice2x2 {
private:
    double m[2][2];
public:
    Matrice2x2() {
        m[0][0] = m[0][1] = m[1][0] = m[1][1] = 0.0;
    }
    Matrice2x2(double a00, double a01, double a10, double a11) {
        m[0][0] = a00;
        m[0][1] = a01;
        m[1][0] = a10;
        m[1][1] = a11;
    }

    double get(int i, int j) const {
        return m[i][j];
    }

    void set(int i, int j, double value) {
        m[i][j] = value;
    }

    double determinant() const {
        return m[0][0]*m[1][1] - m[0][1]*m[1][0];
    }

    Matrice2x2 inverse() const {
        double det = determinant();
        Matrice2x2 inv;
        inv.set(0, 0,  m[1][1] / det);
        inv.set(0, 1, -m[0][1] / det);
        inv.set(1, 0, -m[1][0] / det);
        inv.set(1, 1,  m[0][0] / det);
        return inv;
    }

    void multiplyVector(const double* vec, double* result) const {
        result[0] = m[0][0]*vec[0] + m[0][1]*vec[1];
        result[1] = m[1][0]*vec[0] + m[1][1]*vec[1];
    }
};


class Matrice3x3 {
private:
    double m[3][3];
public:
    Matrice3x3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = 0.0;
    }
    void set(int i, int j, double value) {
        m[i][j] = value;
    }
    double get(int i, int j) const {
        return m[i][j];
    }
    double determinant() const {
        double det = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
        return det;
    }
    Matrice3x3 inverse() const {
        double det = determinant();
        Matrice3x3 inv;
        inv.set(0, 0, (m[1][1]*m[2][2] - m[1][2]*m[2][1]) / det);
        inv.set(0, 1, (m[0][2]*m[2][1] - m[0][1]*m[2][2]) / det);
        inv.set(0, 2, (m[0][1]*m[1][2] - m[0][2]*m[1][1]) / det);
        inv.set(1, 0, (m[1][2]*m[2][0] - m[1][0]*m[2][2]) / det);
        inv.set(1, 1, (m[0][0]*m[2][2] - m[0][2]*m[2][0]) / det);
        inv.set(1, 2, (m[0][2]*m[1][0] - m[0][0]*m[1][2]) / det);
        inv.set(2, 0, (m[1][0]*m[2][1] - m[1][1]*m[2][0]) / det);
        inv.set(2, 1, (m[0][1]*m[2][0] - m[0][0]*m[2][1]) / det);
        inv.set(2, 2, (m[0][0]*m[1][1] - m[0][1]*m[1][0]) / det);
        return inv;
    }
    void multiplyVector(const double* vec, double* result) const {
        for (int i = 0; i < 3; ++i) {
            result[i] = 0.0;
            for (int j = 0; j < 3; ++j) {
                result[i] += m[i][j] * vec[j];
            }
        }
    }
};


class Mesures {
public:
    double x[MAX_SIZE];
    double y[MAX_SIZE];
    double z[MAX_SIZE];
    int size;

    Mesures(const char* filename) {
        FILE* file = fopen(filename, "r");
        if (file == NULL) {
            printf("Impossible d'ouvrir le fichier :%s\n", filename);
            exit(1);
        }

        char line[256];
        size = 0;

        fgets(line, sizeof(line), file);

        while (fgets(line, sizeof(line), file) != NULL) {
            char* token = strtok(line, ",");
            x[size] = atof(token);

            token = strtok(NULL, ",");
            y[size] = atof(token);

            token = strtok(NULL, ",");
            z[size] = atof(token);

            size++;
        }

        fclose(file);
    }

    double correlation(double* a, double* b) {
        double sum_a = 0, sum_b = 0;

        for (int i = 0; i < size; ++i) {
            sum_a += a[i];
            sum_b += b[i];
        }

        double mean_a = sum_a / size;
        double mean_b = sum_b / size;

        double numerator = 0, denom_a = 0, denom_b = 0;

        for (int i = 0; i < size; ++i) {
            double diff_a = a[i] - mean_a;
            double diff_b = b[i] - mean_b;
            numerator += diff_a * diff_b;
            denom_a += diff_a * diff_a;
            denom_b += diff_b * diff_b;
        }

        return numerator / sqrt(denom_a * denom_b);
    }

    void analyse_y(double* a0, double* a1) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_x2 = 0.0;
        double sum_xy = 0.0;
        int N = size;

        for(int i = 0; i < N; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_x2 += x[i] * x[i];
            sum_xy += x[i] * y[i];
        }

        Matrice2x2 A(N, sum_x,
                     sum_x, sum_x2);
        double F[2];
        F[0] = sum_y;
        F[1] = sum_xy;

        Matrice2x2 A_inv = A.inverse();

        double U[2];
        A_inv.multiplyVector(F, U);

        *a0 = U[0];
        *a1 = U[1];
    }

    void analyse_z(double* a0, double* a1, double* a2) {
        double S0 = size;
        double S1 = 0.0, S2 = 0.0, S3 = 0.0, S4 = 0.0;
        double W0 = 0.0, W1 = 0.0, W2 = 0.0;

        for (int i = 0; i < size; ++i) {
            double xi = x[i];
            double zi = z[i];
            double xi2 = xi * xi;
            double xi3 = xi2 * xi;
            double xi4 = xi3 * xi;

            S1 += xi;
            S2 += xi2;
            S3 += xi3;
            S4 += xi4;

            W0 += zi;
            W1 += zi * xi;
            W2 += zi * xi2;
        }

        Matrice3x3 A;
        A.set(0, 0, S0);
        A.set(0, 1, S1);
        A.set(0, 2, S2);
        A.set(1, 0, S1);
        A.set(1, 1, S2);
        A.set(1, 2, S3);
        A.set(2, 0, S2);
        A.set(2, 1, S3);
        A.set(2, 2, S4);

        double F[3];
        F[0] = W0;
        F[1] = W1;
        F[2] = W2;

        Matrice3x3 A_inv = A.inverse();

        double U[3];
        A_inv.multiplyVector(F, U);

        *a0 = U[0];
        *a1 = U[1];
        *a2 = U[2];
    }
};


int main() {
    Mesures mesures("ex1_data.txt");

    double r_xy = mesures.correlation(mesures.x, mesures.y);
    double r_xz = mesures.correlation(mesures.x, mesures.z);

    printf("r(x, y) = %f\n", r_xy);
    printf("r(x, z) = %f\n", r_xz);

    double a0_y, a1_y;
    mesures.analyse_y(&a0_y, &a1_y);
    printf("Pour y: a0 = %f, a1 = %f\n", a0_y, a1_y);

    double a0_z, a1_z, a2_z;
    mesures.analyse_z(&a0_z, &a1_z, &a2_z);
    printf("Pour z: a0 = %f, a1 = %f, a2 = %f\n", a0_z, a1_z, a2_z);

    return 0;
}