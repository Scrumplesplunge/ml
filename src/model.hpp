#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <iostream>

#include <cmath>
#include <concepts>
#include <experimental/mdspan>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>

namespace ml {

template <typename T>
concept Model = requires () {
  { T::num_inputs } -> std::same_as<const std::size_t&>;
  { T::num_outputs } -> std::same_as<const std::size_t&>;
};

template <Model T>
struct Factory {
  using Type = T;
};

template <typename T, typename M>
concept Params = requires () {
  requires M::template params<T>;
};

template <Model A, Model B>
requires (A::num_outputs == B::num_inputs)
struct CompoundModel {
  static constexpr std::size_t num_inputs = A::num_inputs;
  static constexpr std::size_t num_outputs = B::num_outputs;

  template <typename T>
  static constexpr bool params = Params<T, A> && Params<T, B>;

  template <typename T>
  requires Params<T, CompoundModel>
  explicit CompoundModel(T&& params) : a(params), b(params) {}

  struct State {
    A::State a;
    float hidden[A::num_outputs];
    B::State b;
  };

  void Apply(State& state, std::span<const float, num_inputs> inputs,
             std::span<float, num_outputs> outputs) const noexcept {
    a.Apply(state.a, inputs, state.hidden);
    b.Apply(state.b, state.hidden, outputs);
  }

  void Backwards(State& state,
                 std::span<const float, num_inputs> inputs,
                 std::span<const float, num_outputs> output_loss_gradients,
                 std::span<float, num_inputs> input_loss_gradients) {
    float inner_gradients[A::num_outputs];
    b.Backwards(state.b, state.hidden, output_loss_gradients, inner_gradients);
    a.Backwards(state.a, inputs, inner_gradients, input_loss_gradients);
  }

  void Backpropagate(State& state,
                     std::span<const float, num_inputs> inputs,
                     std::span<const float, num_outputs> output_gradients,
                     std::span<float, num_inputs> input_gradients,
                     float learning_rate) {
    // TODO: This is somewhat inefficient since for a sequence of nested
    // CompoundModels it will stack up a bunch of arrays on the stack for each
    // set of intermediate gradients even though we only ever need one set at
    // a time. Fixing this seems to require flattening the sequence rather than
    // using this naive recursive approach.
    float inner_gradients[B::num_inputs];
    b.Backpropagate(state.b, state.hidden, output_gradients, inner_gradients,
                    learning_rate);
    a.Backpropagate(state.a, inputs, inner_gradients, input_gradients,
                    learning_rate);
  }

  A a;
  B b;
};

template <Model A, Model B>
consteval Factory<CompoundModel<A, B>> operator|(Factory<A>, Factory<B>) {
  return {};
}

template <Model T, Params<T> P>
constexpr T Create(Factory<T>, P&& params) { return T(params); }

template <std::size_t i, std::size_t o>
struct Linear {
  static constexpr std::size_t num_inputs = i;
  static constexpr std::size_t num_outputs = o;

  template <typename P>
  static constexpr bool params =
      requires(P p) { p.generator; } &&
      std::uniform_random_bit_generator<
          std::decay_t<decltype(std::declval<P>().generator)>>;

  struct State {};

  template <Params<Linear> P>
  explicit Linear(P&& params) {
    std::normal_distribution<float> f;
    for (auto& row : weights) {
      for (auto& cell : row) cell = f(params.generator);
    }
    for (auto& bias : biases) bias = f(params.generator);
  }

  void Apply(State& state,
             std::span<const float, num_inputs> inputs,
             std::span<float, num_outputs> outputs) const noexcept {
    for (std::size_t y = 0; y < num_outputs; y++) {
      float value = biases[y];
      const float* w = weights[y];
      for (std::size_t x = 0; x < num_inputs; x++) {
        value += inputs[x] * w[x];
      }
      outputs[y] = value;
    }
  }

  void Backwards(State& state,
                 std::span<const float, num_inputs> inputs,
                 std::span<const float, num_outputs> output_loss_gradients,
                 std::span<float, num_inputs> input_loss_gradients) {
    // Calculate the loss gradients with respect to the inputs.
    // dL/dx[i] = sum(j) of (dL/dy[j] * dy[j]/dx[i])
    //          = sum(j) of (output_gradients[j] * weights[j][i])
    for (std::size_t x = 0; x < num_inputs; x++) {
      float value = 0;
      for (std::size_t y = 0; y < num_outputs; y++) {
        value += output_loss_gradients[y] * weights[y][x];
      }
      input_loss_gradients[x] = value;
    }
  }

  void UpdateWeights(std::span<const float, num_inputs> inputs,
                     std::span<const float, num_outputs> output_gradients,
                     float learning_rate) {
    // Update the weights.
    for (std::size_t y = 0; y < num_outputs; y++) {
      for (std::size_t x = 0; x < num_inputs; x++) {
        // dL/dw[j][i] = dL/dy[j] * dy[j]/dw[i]
        //             = output_gradients[j] * state.inputs[i]
        weights[y][x] -= learning_rate * output_gradients[y] * inputs[x];
      }
    }

    // Update the biases.
    for (std::size_t y = 0; y < num_outputs; y++) {
      // dL/db[j] = dL/dy[j] * dy[j]/db[j]
      //          = output_gradients[j] * 1
      biases[y] -= learning_rate * output_gradients[y];
    }
  }

  void Backpropagate(State& state,
                     std::span<const float, num_inputs> inputs,
                     std::span<const float, num_outputs> output_gradients,
                     std::span<float, num_inputs> input_gradients,
                     float learning_rate) {
    // TODO: Experiment with updating the weights before calculating the
    // gradient vs. after calculating the gradient to see whether it makes
    // a difference.
    Backwards(state, inputs, output_gradients, input_gradients);
    UpdateWeights(inputs, output_gradients, learning_rate);
  }

  float weights[num_outputs][num_inputs];
  float biases[num_outputs];
};

template <std::size_t i, std::size_t o>
inline constexpr Factory<Linear<i, o>> linear;

template <std::size_t n>
struct Relu {
  static constexpr std::size_t num_inputs = n;
  static constexpr std::size_t num_outputs = n;
  template <typename T>
  static constexpr bool params = true;

  template <typename T>
  explicit Relu(T&&) {}

  struct State {};

  static void Apply(State& state, std::span<const float, n> inputs,
                    std::span<float, n> outputs) noexcept {
    for (std::size_t i = 0; i < n; i++) {
      outputs[i] = std::max<float>(0, inputs[i]);
    }
  }

  void Backwards(State& state, std::span<const float, n> inputs,
                 std::span<const float, n> output_loss_gradients,
                 std::span<float, n> input_loss_gradients) {
    for (std::size_t i = 0; i < n; i++) {
      input_loss_gradients[i] = inputs[i] > 0 ? output_loss_gradients[i] : 0.0f;
    }
  }

  void Backpropagate(State& state, std::span<const float, n> inputs,
                     std::span<const float, n> output_gradients,
                     std::span<float, n> input_gradients, float learning_rate) {
    Backwards(state, inputs, output_gradients, input_gradients);
  }
};

template <std::size_t n>
inline constexpr Factory<Relu<n>> relu;

template <std::size_t n>
struct LayerNorm {
  static constexpr std::size_t num_inputs = n;
  static constexpr std::size_t num_outputs = n;
  template <typename T>
  static constexpr bool params = true;

  template <typename T>
  explicit LayerNorm(T&&) {}

  struct State {};

  //        y[i] = (x[i] - mean(x)) / sqrt(variance(x) + epsilon)
  //    dL/dx[i] = sum(j) of (dL/dy[j] * dy[j]/dx[i])
  //     f[i](x) = x[i] - mean(x)
  //        s(x) = sqrt(variance(x) + epsilon)
  //        y[i] = f[i](x) / s(x)
  // df[i]/dx[i] = -1/n + 1
  // df[j]/dx[i] = -1/n
  //    ds/dx[i] = (x[i] - mean(x)) / ns(x)                       (*)
  // dy[j]/dx[i] = ((i == j) - 1/n) / s(x) -
  //               (x[i] - mean(x))(x[j] - mean(x)) / ns(x)^3
  //             = n * (i == j) / ns(x) -
  //               (x[i] - mean(x))(x[j] - mean(x)) / ns(x)s(x)^2
  //             = 1 / ns(x) * (n * (i == j) - y[i] * y[j])
  //    dL/dx[i] = sum(j) of (
  //                 output_loss_gradients[j] / ns(x) *
  //                 (n * (i == j) - y[i] * y[j]))
  //             = sum(j) of (output_loss_gradients[j] / ns(x) * n * (i == j)) -
  //               sum(j) of (output_loss_gradients[j] / ns(x) * y[i] * y[j])
  //             = output_loss_gradients[i] / s(x) -
  //               y[i] / ns(x) * sum(j) of (output_loss_gradients[j] * y[j])
  //
  // (*) The proof for this doesn't fit in this comment, but involves using the
  // chain rule for sqrt(u) and u = variance(x) + epsilon.

  struct Components {
    float mean;
    float factor;
  };

  static Components Analyze(std::span<const float, n> values) noexcept {
    float sum = 0;
    for (float x : values) sum += x;
    const float mean = sum / n;
    float variance_sum = 0;
    for (float x : values) variance_sum += (x - mean) * (x - mean);
    const float variance = variance_sum / n;
    constexpr float kEpsilon = 1e-3;
    const float factor = 1.0f / std::sqrt(variance + kEpsilon);
    return {.mean = mean, .factor = factor};
  }

  static void Apply(State& state, std::span<const float, n> inputs,
                    std::span<float, n> outputs) noexcept {
    const auto [mean, factor] = Analyze(inputs);
    for (std::size_t i = 0; i < n; i++) {
      outputs[i] = (inputs[i] - mean) * factor;
    }
  }

  void Backwards(State& state, std::span<const float, n> inputs,
                 std::span<const float, n> output_loss_gradients,
                 std::span<float, n> input_loss_gradients) {
    const auto [mean, factor] = Analyze(inputs);
    float total = 0;
    for (std::size_t i = 0; i < n; i++) {
      const float y = (inputs[i] - mean) * factor;
      input_loss_gradients[i] = y * (factor / n);
      total += output_loss_gradients[i] * y;
    }
    for (std::size_t i = 0; i < n; i++) {
      input_loss_gradients[i] =
          output_loss_gradients[i] * factor - input_loss_gradients[i] * total;
    }
  }

  void Backpropagate(State& state, std::span<const float, n> inputs,
                     std::span<const float, n> output_gradients,
                     std::span<float, n> input_gradients, float learning_rate) {
    Backwards(state, inputs, output_gradients, input_gradients);
  }
};

template <std::size_t n>
inline constexpr Factory<LayerNorm<n>> layer_norm;

template <std::size_t n>
struct Sigmoid {
  static constexpr std::size_t num_inputs = n;
  static constexpr std::size_t num_outputs = n;
  template <typename T>
  static constexpr bool params = true;

  template <typename T>
  explicit Sigmoid(T&&) {}

  struct State {};

  static void Apply(State& state, std::span<const float, n> inputs,
                    std::span<float, n> outputs) noexcept {
    for (std::size_t i = 0; i < n; i++) {
      outputs[i] = 1.0f / (1.0f + std::exp(-inputs[i]));
    }
  }

  void Backwards(State& state, std::span<const float, n> inputs,
                 std::span<const float, n> output_loss_gradients,
                 std::span<float, n> input_loss_gradients) {
    //     y = 1 / (1 + e^-x)
    // dy/dx = ((1 + e^-x) * 0 - 1 * (-e^-x)) / (1 + e^-x)^2
    //       =                         e^-x   / (1 + e^-x)^2
    for (std::size_t i = 0; i < n; i++) {
      const float x = std::exp(-inputs[i]);
      input_loss_gradients[i] =
          output_loss_gradients[i] * x / ((1 + x) * (1 + x));
    }
  }

  void Backpropagate(State& state, std::span<const float, n> inputs,
                     std::span<const float, n> output_gradients,
                     std::span<float, n> input_gradients, float learning_rate) {
    Backwards(state, inputs, output_gradients, input_gradients);
  }
};

template <std::size_t n>
inline constexpr Factory<Sigmoid<n>> sigmoid;

template <std::size_t n>
struct Softmax {
  static constexpr std::size_t num_inputs = n;
  static constexpr std::size_t num_outputs = n;
  template <typename T>
  static constexpr bool params = true;

  template <typename T>
  explicit Softmax(T&&) {}

  struct State {};

  //        y[i] = e^x[i] / (sum(j) of e^x[j])
  //                v                    du/dx             - u        dv/dx
  // dy[j]/dx[i] = ((sum(k) of e^x[k]) * (i == j) * e^x[i] - e^x[j] * e^x[i]) /
  //               (sum(k) of e^x[k])^2
  //             = (total * (i == j) - e^x[j]) * e^x[i] / (total * total)
  //    dL/dx[i] = sum(j) of (dL/dy[j] * dy[j]/dx[i])

  static void Apply(State& state, std::span<const float, n> inputs,
                    std::span<float, n> outputs) noexcept {
    // For numerical stability, offset everything by the maximum value. The
    // result is the same:
    //
    //   e^a / (e^a + e^b) = e^-x / e^-x * e^a / (e^a + e^b)
    //                     = e^(a - x) / (e^(a - x) + e^(b - x))
    //
    // This ensures that we only have unboundedly negative values, which will
    // saturate to `0` rather than `inf` and avoid producing `nan`.
    const float max = *std::max_element(inputs.begin(), inputs.end());
    float total = 0;
    for (std::size_t i = 0; i < n; i++) {
      outputs[i] = std::exp(inputs[i] - max);
      total += outputs[i];
    }
    const float factor = 1.0f / total;
    for (std::size_t i = 0; i < n; i++) outputs[i] *= factor;
  }

  void Backwards(State& state, std::span<const float, n> inputs,
                 std::span<const float, n> output_loss_gradients,
                 std::span<float, n> input_loss_gradients) {
    float e[n];
    float total = 0;
    for (std::size_t i = 0; i < n; i++) {
      e[i] = std::exp(inputs[i]);
      total += e[i];
    }
    const float factor = 1.0f / (total * total);
    for (std::size_t i = 0; i < n; i++) {
      float loss_gradient = 0;
      for (std::size_t j = 0; j < n; j++) {
        const float dldyj = output_loss_gradients[j];
        const float k = i == j ? e[i] : 1.0f;
        const float dyjdxi = (total * (i == j) - e[j]) * e[i] * factor;
        loss_gradient += dldyj * dyjdxi;
      }
      input_loss_gradients[i] = loss_gradient;
    }
  }

  void Backpropagate(State& state, std::span<const float, n> inputs,
                     std::span<const float, n> output_gradients,
                     std::span<float, n> input_gradients, float learning_rate) {
    Backwards(state, inputs, output_gradients, input_gradients);
  }
};

template <std::size_t n>
inline constexpr Factory<Softmax<n>> softmax;

template <Model M>
void Train(M& model, std::span<const float, M::num_inputs> inputs,
           std::span<const float, M::num_outputs> expected_outputs,
           float learning_rate) {
  float outputs[M::num_outputs];
  typename M::State state;
  model.Apply(state, inputs, outputs);

  float gradients[M::num_outputs];
  for (std::size_t i = 0; i < M::num_outputs; i++) {
    // Mean squared error:
    //   const float diff = outputs[i] - expected_outputs[i];
    //   gradients[i] = 2 * diff;
    // Cross entropy:
    //        L = sum(i) of -y[i] * log(x[i])
    // dL/dx[i] = -y[i] / x[i]
    gradients[i] = -expected_outputs[i] / outputs[i];
  }

  // TODO: Figure out if we can avoid the redundant gradient calculation at the
  // top level.
  float input_gradients[M::num_inputs];
  model.Backpropagate(state, inputs, gradients, input_gradients, learning_rate);
}

template <Model M>
void Run(const M& model, std::span<const float, M::num_inputs> inputs,
         std::span<float, M::num_outputs> outputs) {
  typename M::State state;
  model.Apply(state, inputs, outputs);
}

template <std::size_t n>
requires (n != 0)
std::size_t Select(std::span<const float, n> values) {
  if (values.empty()) throw std::logic_error("nothing to select from");
  return std::max_element(values.begin(), values.end()) - values.begin();
}

}  // namespace ml

#endif  // MODEL_HPP_
