#include "load_digit.hpp"

#include <CImg.h>

namespace ml {

std::array<float, 28 * 28> LoadDigit(
    std::experimental::mdspan<const unsigned char,
                              std::experimental::extents<std::size_t, 28, 28>>
        image) {
  std::array<float, 28 * 28> inputs;
  for (std::size_t y = 0; y < 28; y++) {
    for (std::size_t x = 0; x < 28; x++) {
      inputs[28 * y + x] = image(y, x) / 255.0f;
    }
  }
  return inputs;
}

std::array<float, 28 * 28> LoadDigit(const char* filename) {
  cimg_library::CImg<unsigned char> image(filename);
  image.autocrop();
  const int size = std::max(image.width(), image.height());
  image.resize(20 * image.width() / size, 20 * image.height() / size, 1, 1,
               /*cubic interpolation*/5);
  cimg_library::CImg<unsigned char> canvas(28, 28, 1, 1, 0);
  canvas.draw_image(14 - image.width() / 2, 14 - image.height() / 2, image);
  return LoadDigit(
      std::experimental::mdspan<
          const unsigned char, std::experimental::extents<std::size_t, 28, 28>>(
          canvas.data()));
}

}  // namespace ml
