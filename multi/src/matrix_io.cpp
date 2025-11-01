#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <sstream> 

void generateAndSaveMatrices(int l, int n, int m, 
                             const std::string& fileA = "matrixA.txt", 
                             const std::string& fileB = "matrixB.txt") 
{
    std::ofstream outA(fileA);
    std::ofstream outB(fileB);

    if (!outA.is_open() || !outB.is_open()) {
        std::cerr << "Error opening file for writing.\n";
        exit(EXIT_FAILURE);
    }

    // Generate and save matrix A (l x n)
    for (int i = 0; i < l; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            outA << val << " ";
        }
        outA << "\n";
    }

    // Generate and save matrix B (n x m)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float val = static_cast<float>(rand()) / RAND_MAX;
            outB << val << " ";
        }
        outB << "\n";
    }

    outA.close();
    outB.close();

    std::cout << "Matrices saved to " << fileA << " and " << fileB << std::endl;
}

// Function to load a matrix from a file
std::vector<std::vector<float>> loadMatrix(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error opening file " << filename << " for reading.\n";
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<float>> matrix;
    std::string line;

    while (std::getline(in, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        float val;
        while (iss >> val) {
            row.push_back(val);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }

    in.close();
    return matrix;
}

// Helper function to print a matrix
void printMatrix(const std::vector<std::vector<float>>& mat) {
    for (const auto& row : mat) {
        for (auto val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}
