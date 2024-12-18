#include <iostream>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

// 模板类 EncodingTable
template <typename T>
class EncodingTable {
private:
    std::map<T, char> table;  // 字典：键与字符的映射关系

public:
    // 构造函数：读取文件并填充字典
    EncodingTable(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }

        T key;
        char value;
        while (file >> key >> value) {
            table[key] = value;
        }
    }

    // character 方法：返回对应的字符
    char character(T key) const {
        auto it = table.find(key);
        if (it == table.end()) {
            throw std::out_of_range("无效的键: " + std::to_string(key));
        }
        return it->second;
    }

    // 重载输出运算符 <<
    friend std::ostream& operator<<(std::ostream& os, const EncodingTable<T>& et) {
        for (const auto& pair : et.table) {
            os << pair.first << " -> " << pair.second << "\n";
        }
        return os;
    }
};

// 测试函数
int main() {
    try {
        // 读取 encoding_table.txt 文件并构建 EncodingTable 对象
        EncodingTable<int> et("encoding_table.txt");

        // 显示编码表
        std::cout << "编码表内容：" << std::endl;
        std::cout << et;

        // 测试 character 方法
        std::cout << "测试 character 方法：" << std::endl;
        std::cout << "键 65 的字符是: " << et.character(65) << std::endl; // 示例键
        std::cout << "键 66 的字符是: " << et.character(66) << std::endl;

        // 测试无效键
        std::cout << "尝试无效的键：" << std::endl;
        std::cout << "键 9999 的字符是: " << et.character(9999) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
    }

    return 0;
}


#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>

// 基类 DecodeChar
template <typename T>
class DecodeChar {
protected:
    T key;     // 键
    T graine;  // 种子

public:
    // 构造函数，初始化种子，键设为 0
    DecodeChar(T graine = 0) : key(0), graine(graine) {}

    // 获取当前键的值
    T getKey() const {
        return key;
    }

    // 获取种子的值
    T getGraine() const {
        return graine;
    }

    // 纯虚方法 update，需要派生类实现
    virtual void update(std::istream& is) = 0;

    // 虚析构函数
    virtual ~DecodeChar() = default;
};

// 子类 DecodeCharSomme：通过求和计算键
template <typename T>
class DecodeCharSomme : public DecodeChar<T> {
public:
    // 使用基类构造函数
    using DecodeChar<T>::DecodeChar;

    // 重写 update 方法，计算整数序列的和
    void update(std::istream& is) override {
        T n;  // 整数序列的长度
        is >> n;

        T sum = 0;
        T value;
        for (T i = 0; i < n; ++i) {
            is >> value;
            sum += value;
        }

        this->key = sum;  // 更新键的值
    }
};

// 子类 DecodeCharMax：通过取最大值计算键
template <typename T>
class DecodeCharMax : public DecodeChar<T> {
public:
    // 使用基类构造函数
    using DecodeChar<T>::DecodeChar;

    // 重写 update 方法，计算整数序列的最大值
    void update(std::istream& is) override {
        T n;  // 整数序列的长度
        is >> n;

        T maxValue = std::numeric_limits<T>::lowest();
        T value;
        for (T i = 0; i < n; ++i) {
            is >> value;
            if (value > maxValue) {
                maxValue = value;
            }
        }

        this->key = maxValue;  // 更新键的值
    }
};

// 测试函数
int main() {
    try {
        // 测试 DecodeCharSomme
        DecodeCharSomme<int> dcs;
        std::istringstream istrSomme("6 1 2 3 4 5 6");
        dcs.update(istrSomme);
        std::cout << "dcs.key() : " << dcs.getKey() << std::endl;

        // 测试 DecodeCharMax
        DecodeCharMax<int> dcm;
        std::istringstream istrMax("13 55 64 18 17 74 31 51 57 23 62 29 47 23");
        dcm.update(istrMax);
        std::cout << "dcm.key() : " << dcm.getKey() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
    }

    return 0;
}
