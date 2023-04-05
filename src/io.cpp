#include "io.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>
#include <utility>

namespace ml {
namespace {

bool IsValidType(unsigned char c) {
  return c == AnyIdxFile::UBYTE || c == AnyIdxFile::BYTE ||
         c == AnyIdxFile::SHORT || c == AnyIdxFile::INT ||
         c == AnyIdxFile::FLOAT || c == AnyIdxFile::DOUBLE;
}

size_t Size(AnyIdxFile::DataType t) {
  switch (t) {
    case AnyIdxFile::UBYTE:
    case AnyIdxFile::BYTE:
      return 1;
    case AnyIdxFile::SHORT:
      return 2;
    case AnyIdxFile::INT:
    case AnyIdxFile::FLOAT:
      return 4;
    case AnyIdxFile::DOUBLE:
      return 8;
  }
  throw std::logic_error("Invalid DataType");
}

}  // namespace

MappedFile::MappedFile(const char* filename) {
  const int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error(std::string("Could not open ") + filename);
  }
  struct stat info;
  if (fstat(fd, &info) < 0) {
    close(fd);
    throw std::runtime_error(std::string("Could not stat ") + filename);
  }
  data_ = (const unsigned char*)mmap(nullptr, info.st_size, PROT_READ,
                                     MAP_SHARED, fd, 0);
  size_ = info.st_size;
  close(fd);
  if (data_ == (void*)-1) {
    throw std::runtime_error(std::string("Could not map ") + filename);
  }
}

MappedFile::~MappedFile() noexcept {
  if (data_) munmap((void*)data_, size_);
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) return *this;
  if (data_) munmap((void*)data_, size_);
  data_ = std::exchange(other.data_, nullptr);
  size_ = std::exchange(other.size_, 0);
  return *this;
}

AnyIdxFile::AnyIdxFile(const char* filename) : MappedFile(filename) {
  const std::span<const unsigned char> contents = bytes();
  if (contents.size() < 4) {
    throw std::runtime_error(std::string(filename) +
                             " is not a valid IDX file (too short)");
  }
  if (contents[0] != 0 || contents[1] != 0 || !IsValidType(contents[2])) {
    throw std::runtime_error(std::string(filename) +
                             " is not a valid IDX file (bad magic number)");
  }
  const size_t header_size = 4 * (1 + rank());
  if (contents.size() < header_size) {
    throw std::runtime_error(std::string(filename) +
                             " is not a valid IDX file (truncated header)");
  }
  size_t num_values = 1;
  for (unsigned char i = 0; i < rank(); i++) {
    const unsigned char* cell = contents.data() + 4 + 4 * i;
    const size_t dimension_size = (size_t)cell[0] << 24 |
                                  (size_t)cell[1] << 16 | (size_t)cell[2] << 8 |
                                  (size_t)cell[3];
    num_values *= dimension_size;
  }
  const size_t data_size = num_values * Size(type());
  if (contents.size() != header_size + data_size) {
    throw std::runtime_error(std::string(filename) +
                             " is not a valid IDX file (size mismatch)");
  }
}

std::span<const unsigned char> AnyIdxFile::contents() const noexcept {
  const size_t header_size = 4 * (1 + rank());
  return bytes().subspan(header_size, size() - header_size);
}

}  // namespace ml
