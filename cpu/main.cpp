#include <iostream>
#include <fstream>
using namespace std;

bool loadDataFromFile(const string& fileName, int& N, int& R, float**& matrix)
{
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Cannot open file!" << endl;
        return false;
    }

    // Read N and R
    file >> N >> R;

    // Allocate N x N matrix
    matrix = new float*[N];
    for (int i = 0; i < N; i++)
        matrix[i] = new float[N];

    // Read matrix values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            file >> matrix[i][j];
        }
    }

    file.close();
    return true;
}

bool saveMatrixToFile(const string& fileName, float** matrix, int N)
{
    ofstream file(fileName, ios::out | ios::trunc);
    if (!file.is_open())
    {
        cerr << "Cannot open output file!" << endl;
        return false;
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            file << matrix[i][j];

            if (j < N - 1)
                file << " ";
        }
        file << endl;
    }

    file.close();
    return true;
}

void freeMatrix(float** matrix, int N)
{
    for (int i = 0; i < N; i++)
        delete[] matrix[i];

    delete[] matrix;
}


float calculateElement(float ** TAB, int R, int x, int y){
    float sum = 0.0;
    for(int i = x-R; i<= x + R; i++){
        for(int j = y-R; j<= y + R; j++){
            sum += TAB[i][j];
        }
    }
    return sum;
}
int main() {

    int N, R;
    float** TAB = nullptr;
    float** OUT = nullptr;

    if (loadDataFromFile("dane.txt", N, R, TAB))
    {
        int M = N - 2*R; // rozmiar tablicy OUT
        if(M > 0){
            OUT = new float*[M];
            for (int i = 0; i < M; i++){
                OUT[i] = new float[M];
                for (int j = 0; j < M; j++) OUT[i][j] =  calculateElement(TAB, R, i+R, j+R);
            }
            saveMatrixToFile("odp.txt", OUT, M);
            freeMatrix(OUT, M);
        }
        else cerr << "Zmienna R jest za duża względem rozmiatu tablicy wejściowej!" << endl;
        freeMatrix(TAB, N);
    }


    return 0;
}