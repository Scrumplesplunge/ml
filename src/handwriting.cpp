#include <cmath>
#include <concepts>
#include <iostream>
#include <random>

#include "io.hpp"
#include "model.hpp"

namespace handwriting {
namespace {

using Network =
    decltype(ml::linear<28 * 28, 16> | ml::sigmoid<16> | ml::linear<16, 16> |
             ml::sigmoid<16> | ml::linear<16, 10> | ml::softmax<10>)::Type;

std::array<float, 28 * 28> LoadImage(
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
  int guesses[10] = {};
  int actual[10] = {};
  float correct_confidence = 0;
  float incorrect_confidence = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (i % 1000 == 0) {
      std::cout << "\rGrading (" << i << "/" << n << ")..." << std::flush;
    }
    const std::array<float, 28 * 28> inputs =
        LoadImage(std::experimental::submdspan(images.contents(), i,
                                               std::experimental::full_extent,
                                               std::experimental::full_extent));
    float outputs[10];
    ml::Run(network, inputs, outputs);
    const int guess = ml::Select(std::span<const float, 10>(outputs));
    (guess == labels(i) ? correct_confidence : incorrect_confidence) +=
        outputs[guess];
    if (guess == labels(i)) correct++;
    guesses[guess]++;
    actual[labels(i)]++;
  }
  std::cout << "\rGrading complete: " << correct << '/' << n << " correct\n";
  std::cout << "labels:         ";
  for (int i = 0; i < 10; i++) std::cout << i << '\t';
  std::cout << "\nactual:         ";
  for (int i = 0; i < 10; i++) std::cout << actual[i] << '\t';
  std::cout << "\nguesses:        ";
  for (int i = 0; i < 10; i++) std::cout << guesses[i] << '\t';
  std::cout << "\ncorrect_confidence: " << correct_confidence / correct << '\n';
  std::cout << "incorrect_confidence: " << incorrect_confidence / (n - correct)
            << '\n';
}

Network Train() {
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

  struct {
    std::ranlux48_base generator{std::random_device()()};
  } params;

  Network network(params);

  Grade(network);

  constexpr int kMaxNumEpochs = 5;
  for (int epoch = 0; epoch < kMaxNumEpochs; epoch++) {
    int correct = 0;
    for (std::size_t i = 0; i < n; i++) {
      const std::size_t index = 1337 * i % n;
      const std::array<float, 28 * 28> inputs =
          LoadImage(std::experimental::submdspan(
              images.contents(), index, std::experimental::full_extent,
              std::experimental::full_extent));
      const auto label = labels(index);
      if (label > 9) throw std::runtime_error("label out of bounds");
      float expected_outputs[10] = {};
      expected_outputs[label] = 1.0f;
      float outputs[10];
      ml::Run(network, inputs, outputs);
      if (ml::Select(std::span<const float, 10>(outputs)) == label) {
        correct++;
      }
      constexpr float kLearningRate = 0.01;
      ml::Train(network, inputs, expected_outputs, kLearningRate);
      if (i % 1000 == 0) {
        std::cout << "\rTraining (" << i << "/" << n << ")..." << std::flush;
      }
    }
    std::cout << "\rTraining accuracy: " << float(correct) / n << '\n';
    Grade(network);
  }

  return network;
}

void Run() {
  const Network network = Train();
}

}  // namespace
}  // namespace ml

int main() { handwriting::Run(); }
