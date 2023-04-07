#ifndef LOAD_DIGIT_HPP_
#define LOAD_DIGIT_HPP_

#include <array>
#include <experimental/mdspan>
#include <vector>

namespace ml {

std::array<float, 28 * 28> LoadDigit(
    std::experimental::mdspan<const unsigned char,
                              std::experimental::extents<std::size_t, 28, 28>>
        image);

std::array<float, 28 * 28> LoadDigit(const char* filename);

std::vector<std::array<float, 28 * 28>> LoadSprites(const char* filename,
                                                    int size);

}  // namespace ml

#endif  // LOAD_DIGIT_HPP_
