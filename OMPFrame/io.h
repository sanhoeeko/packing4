#pragma once

#include <string>
#include <iostream>
#include <fstream>

template<typename ty>
void writeArrayToFile(ty* ptr, size_t size, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(ptr), size * sizeof(ty));
        file.close();
    }
    else {
        std::cout << "Unable to open file" << std::endl;
    }
}

template<typename ty>
void readArrayFromFile(ty* ptr, size_t size, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(ptr), size * sizeof(ty));
        file.close();
    }
    else {
        std::cout << "Unable to open file" << std::endl;
    }
}