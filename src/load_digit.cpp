#include "load_digit.hpp"

#include <CImg.h>

namespace ml {
namespace {

std::array<float, 28 * 28> Normalize(cimg_library::CImg<unsigned char> image) {
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

}  // namespace

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
  return Normalize(cimg_library::CImg<unsigned char>(filename));
}

std::vector<std::array<float, 28 * 28>> LoadSprites(const char* filename,
                                                    int size) {
  cimg_library::CImg<unsigned char> image(filename);
  if (image.width() % size != 0 || image.height() % size != 0) {
    throw std::runtime_error(
        "Image dimensions are not a multiple of the block size");
  }
  const int w = image.width() / size;
  const int h = image.height() / size;
  std::vector<std::array<float, 28 * 28>> sprites;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      cimg_library::CImg<unsigned char> copy(image);
      copy.crop(size * x, size * y, size * (x + 1), size * (y + 1));
      sprites.push_back(Normalize(copy));
    }
  }
  return sprites;
}

}  // namespace ml
