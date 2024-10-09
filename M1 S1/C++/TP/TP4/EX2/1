#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>

class Complexe {
private:
    double re;  // 实部
    double im;  // 虚部

public:
    // 默认构造函数
    Complexe() : re(0.0), im(0.0) {}
    
    // 带参数的构造函数
    Complexe(double real, double imag) : re(real), im(imag) {}

    // 计算复数模的方法
    double module() const {
        return sqrt(re * re + im * im);
    }

    // 重载输出运算符 << 用于显示
    friend std::ostream& operator<<(std::ostream& os, const Complexe& c) {
        os << "(" << c.re << "," << c.im << ")";
        return os;
    }

    // 重载输入运算符 >> 用于从流中读取
    friend std::istream& operator>>(std::istream& is, Complexe& c) {
        char ch;  // 用于读取字符 '(' ',' 和 ')'
        is >> ch >> c.re >> ch >> c.im >> ch;
        return is;
    }
};

// 从文件中读取复数的函数
std::vector<Complexe> lireNombresComplexes(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Complexe> nombresComplexes;
    if (file.is_open()) {
        int taille;
        file >> taille;  // 读取复数列表的大小
        Complexe c;
        for (int i = 0; i < taille; ++i) {
            file >> c;  // 读取每个复数
            nombresComplexes.push_back(c);
        }
        file.close();
    } else {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
    }
    return nombresComplexes;
}

// 计算复数模的平均值的函数
double moyenneModules(const std::vector<Complexe>& nombresComplexes) {
    double sommeModules = 0.0;
    for (const auto& complexe : nombresComplexes) {
        sommeModules += complexe.module();
    }
    return sommeModules / nombresComplexes.size();
}

int main() {
    // 从文件中读取复数
    std::vector<Complexe> nombresComplexes = lireNombresComplexes("nombrecomplexe.txt");

    // 计算并显示模的平均值
    double moyenne = moyenneModules(nombresComplexes);
    std::cout << "复数模的平均值是: " << moyenne << std::endl;

    return 0;
}
