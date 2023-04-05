#ifndef IO_HPP_
#define IO_HPP_

#include <cstdint>
#include <span>
#include <experimental/mdspan>

namespace ml {

class MappedFile {
 public:
  MappedFile(const char* filename);
  ~MappedFile() noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr const unsigned char* data() const noexcept { return data_; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr std::span<const unsigned char> bytes() const noexcept {
    return std::span<const unsigned char>(data_, size_);
  }

 private:
  const unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
};

class AnyIdxFile : public MappedFile {
 public:
  enum DataType : unsigned char {
    UBYTE = 0x8,
    BYTE = 0x9,
    SHORT = 0xB,
    INT = 0xC,
    FLOAT = 0xD,
    DOUBLE = 0xE,
  };

  AnyIdxFile(const char* filename);

  constexpr DataType type() const noexcept { return DataType(data()[2]); }
  constexpr unsigned char rank() const noexcept { return data()[3]; }

  constexpr std::size_t extent(unsigned char d) const noexcept {
    const unsigned char* x = data() + 4 + 4 * d;
    return (std::size_t)x[0] << 24 | (std::size_t)x[1] << 16 |
           (std::size_t)x[2] << 8 | (std::size_t)x[3];
  }

  std::span<const unsigned char> contents() const noexcept;
};

template <typename T>
struct idx_type;

template <>
struct idx_type<std::uint8_t> {
  static constexpr AnyIdxFile::DataType value = AnyIdxFile::UBYTE;
};

template <>
struct idx_type<std::int8_t> {
  static constexpr AnyIdxFile::DataType value = AnyIdxFile::BYTE;
};

template <>
struct idx_type<std::int16_t> {
  static constexpr AnyIdxFile::DataType value = AnyIdxFile::SHORT;
};

template <>
struct idx_type<std::int32_t> {
  static constexpr AnyIdxFile::DataType value = AnyIdxFile::INT;
};

template <>
struct idx_type<float> {
  static constexpr AnyIdxFile::DataType value = AnyIdxFile::FLOAT;
};

template <>
struct idx_type<double> {
  static constexpr AnyIdxFile::DataType value = AnyIdxFile::DOUBLE;
};

template <typename T>
static constexpr AnyIdxFile::DataType idx_type_v = idx_type<T>::value;

template <typename T, std::size_t... extents>
class IdxFile {
 public:
  using Extents = std::experimental::extents<std::size_t, extents...>;

  IdxFile(const char* filename) : file_(filename) {
    if (file_.type() != type()) {
      throw std::runtime_error(std::string(filename) +
                               " has incorrect element type");
    }
    if (file_.rank() != rank()) {
      throw std::runtime_error(std::string(filename) + " has incorrect rank");
    }
    extents_ =
        MakeExtents(filename, std::make_index_sequence<sizeof...(extents)>());
  }

  constexpr AnyIdxFile::DataType type() const noexcept { return idx_type_v<T>; }
  constexpr unsigned char rank() const noexcept { return sizeof...(extents); }
  constexpr std::size_t extent(unsigned char d) const noexcept {
    return extents_.extent(d);
  }

  std::experimental::mdspan<const T, Extents> contents() const noexcept {
    return std::experimental::mdspan<const T, Extents>(
        file_.data() + 4 * (1 + rank()), extents_);
  }

  template <typename... SizeTypes>
  const T& operator()(SizeTypes... indices) const {
    return contents()(indices...);
  }

 private:
  IdxFile(IdxFile&&) = delete;
  IdxFile& operator=(IdxFile&&) = delete;

  template <std::size_t index, std::size_t extent>
  void CheckExtent(const char* filename) const {
    if (extent != std::dynamic_extent && file_.extent(index) != extent) {
      throw std::runtime_error(std::string(filename) +
                               " has an extent mismatch for extent " +
                               std::to_string(index));
    }
  }

  template <std::size_t... indices>
  Extents MakeExtents(const char* filename,
                      std::index_sequence<indices...>) const {
    (CheckExtent<indices, extents>(filename), ...);
    return Extents(file_.extent(indices)...);
  }

  Extents extents_;
  AnyIdxFile file_;
};

}  // namespace ml

#endif  // IO_HPP_
