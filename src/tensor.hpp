#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <concepts>
#include <experimental/mdspan>

namespace ml {

template <typename T, std::size_t... dimensions>
struct TensorView {
  explicit TensorView(T* data) : view(data) {}

  template <typename U, std::size_t n>
  requires std::convertible_to<U(&)[], T(&)[]> && (n == (dimensions * ...))
  TensorView(U (&data)[n]) : view(data) {}

  template <typename U, std::size_t n>
  requires std::convertible_to<U(&)[], T(&)[]> && (n == (dimensions * ...))
  TensorView(std::array<U, n>& x) : view(x.data()) {}

  template <typename U, std::size_t n>
  requires std::convertible_to<const U (&)[], T (&)[]> &&
           (n == (dimensions * ...))
  TensorView(const std::array<U, n>& x) : view(x.data()) {}

  template <typename U>
  requires std::convertible_to<U(&)[], T(&)[]>
  TensorView(TensorView<U, dimensions...> other) : view(other.view) {}

  template <typename... SizeTypes>
  decltype(auto) operator[](SizeTypes... indices) const {
    return view(indices...);
  }

  std::experimental::mdspan<
      T, std::experimental::extents<std::size_t, dimensions...>>
      view;
};

template <typename T, std::size_t n>
TensorView(T (&data)[n]) -> TensorView<T, n>;

template <typename T, std::size_t n>
TensorView(std::array<T, n>&) -> TensorView<T, n>;

}  // namespace ml

#endif  // TENSOR_HPP_
