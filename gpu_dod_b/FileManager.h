#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <string>

class FileManager {
public:
    static bool loadMatrix(
        const std::string& fileName,
        int& N,
        float**& matrix,
        float*& data
    );

    static bool saveMatrix(
        const std::string& fileName,
        float** matrix,
        int N
    );
};

#endif // FILE_MANAGER_H
