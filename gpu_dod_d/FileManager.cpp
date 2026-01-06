#include "FileManager.h"
#include <fstream>
#include <iostream>

bool FileManager::loadMatrix(
    const std::string& fileName,
    int& N,
    float**& matrix,
    float*& data
) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Cannot open file!" << std::endl;
        return false;
    }

    file >> N;

    data = new float[N * N];
    matrix = new float*[N];
    for (int i = 0; i < N; i++)
        matrix[i] = data + i * N;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            file >> matrix[i][j];

    file.close();
    return true;
}

bool FileManager::saveMatrix(
    const std::string& fileName,
    float** matrix,
    int N
) {
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Cannot open output file!" << std::endl;
        return false;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file << matrix[i][j];
            if (j < N - 1) file << " ";
        }
        file << "\n";
    }

    file.close();
    return true;
}
