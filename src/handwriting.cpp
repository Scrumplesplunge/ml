#include <cmath>
#include <concepts>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

#include "io.hpp"
#include "load_digit.hpp"
#include "model.hpp"

namespace handwriting {
namespace {

using Network =
    decltype(ml::linear<28 * 28, 16> | ml::layer_norm<16> | ml::relu<16> |
             ml::linear<16, 16> | ml::layer_norm<16> | ml::relu<16> |
             ml::linear<16, 10> | ml::softmax<10>)::Type;

void Grade(const Network& network) {
  const ml::IdxFile<unsigned char, std::dynamic_extent, 28, 28> images(
      "data/mnist/t10k-images-idx3-ubyte");
  const ml::IdxFile<unsigned char, std::dynamic_extent> labels(
      "data/mnist/t10k-labels-idx1-ubyte");

  if (images.extent(0) != labels.extent(0)) {
    std::cerr << "Mismatch between number of images and number of labels.\n";
    std::exit(1);
  }

  const std::size_t n = images.extent(0);
  std::size_t correct = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (i % 1000 == 0) {
      std::cout << "\rGrading (" << i << "/" << n << ")..." << std::flush;
    }
    const std::array<float, 28 * 28> inputs =
        ml::LoadDigit(std::experimental::submdspan(
            images.contents(), i, std::experimental::full_extent,
            std::experimental::full_extent));
    float outputs[10];
    ml::Run(network, inputs, outputs);
    const int guess = ml::Select(outputs);
    if (guess == labels(i)) correct++;
  }
  std::cout << "\rGrading complete: " << 100.0f / n * correct << "% accuracy\n";
}

void Train(Network& network) {
  const ml::IdxFile<unsigned char, std::dynamic_extent, 28, 28> images(
      "data/mnist/train-images-idx3-ubyte");
  const ml::IdxFile<unsigned char, std::dynamic_extent> labels(
      "data/mnist/train-labels-idx1-ubyte");

  if (images.extent(0) != labels.extent(0)) {
    throw std::runtime_error(
        "Mismatch between number of images and number of labels.");
  }

  const std::size_t n = images.extent(0);
  std::cout << n << " images to train with.\n";

  Grade(network);

  constexpr int kMaxNumEpochs = 8;
  constexpr float kLearningRate = 0.1;
  for (int epoch = 0; epoch < kMaxNumEpochs; epoch++) {
    for (std::size_t i = 0; i < n; i++) {
      const std::size_t index = 1337 * i % n;
      const std::array<float, 28 * 28> inputs =
          ml::LoadDigit(std::experimental::submdspan(
              images.contents(), index, std::experimental::full_extent,
              std::experimental::full_extent));
      const auto label = labels(index);
      if (label > 9) throw std::runtime_error("label out of bounds");
      float expected_outputs[10] = {};
      expected_outputs[label] = 1.0f;
      ml::Train(network, inputs, expected_outputs, kLearningRate);
      if (i % 1000 == 0) {
        std::cout << "\rTraining (" << i << "/" << n << ")..." << std::flush;
      }
    }
    Grade(network);
  }

  std::cout << "Specializing on Agata's handwriting...\n";

  const auto agata = ml::LoadSprites("data/agata.png", 100);
  const auto w = agata.size() / 10;
  for (int epoch = 0; epoch < 5 * kMaxNumEpochs; epoch++) {
    for (int y = 0; y < 10; y++) {
      // Skip column 1 for grading.
      for (int x = 1; x < w; x++) {
        float expected_outputs[10] = {};
        expected_outputs[y] = 1.0f;
        ml::Train(network, agata[y * w + x], expected_outputs, kLearningRate);
      }
    }
  }

  Grade(network);
}

Network LoadOrTrain() {
  struct {
    std::ranlux48_base generator{std::random_device()()};
  } params;

  Network network(params);

  try {
    ml::MappedFile saved("build/network.bin");
    if (saved.size() != sizeof(network)) {
      throw std::runtime_error("size mismatch");
    }
    std::memcpy(&network, saved.data(), saved.size());
    std::cout << "Loaded pre-trained network.\n";
    Grade(network);
    return network;
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    Train(network);
    std::ofstream saved("build/network.bin", std::ios::binary);
    saved.write(reinterpret_cast<const char*>(&network), sizeof(network));
    if (!saved.good()) std::cerr << "Failed to save network.\n";
    return network;
  }
}

void Run() {
  const Network network = LoadOrTrain();
  std::cout << "Model size: " << sizeof(network) / sizeof(float)
            << " weights\n";
  std::cout << "Testing against custom inputs...\n";
  for (char c = '0'; c <= '9'; c++) {
    std::string filename("data/");
    filename.push_back(c);
    filename += ".png";
    const std::array<float, 28 * 28> inputs = ml::LoadDigit(filename.c_str());
    float outputs[10];
    ml::Run(network, inputs, outputs);
    const int guess = ml::Select(outputs);
    std::cout << filename << ": guessed " << guess
              << " (with p=" << outputs[guess] << ")\n";
  }
  const auto agata = ml::LoadSprites("data/agata.png", 100);
  const auto w = agata.size() / 10;
  int correct = 0;
  for (int y = 0; y < 10; y++) {
    float outputs[10];
    ml::Run(network, agata[y * w], outputs);
    const int guess = ml::Select(outputs);
    std::cout << "Agata '" << y << "': guessed " << guess << "\n";
    if (guess == y) correct++;
  }
  std::cout << "For Agata's handwriting, guessed " << 100.0f * correct / 10
            << "% correctly\n";
}

}  // namespace
}  // namespace ml

int main() { handwriting::Run(); }
