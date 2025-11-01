#pragma once
#include <vector>
#include <string>

void generateAndSaveMatrices(int l, int n, int m,
                             const std::string& fileA = "matrixA.txt",
                             const std::string& fileB = "matrixB.txt");

std::vector<std::vector<float>> loadMatrix(const std::string& filename);
