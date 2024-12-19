HIPCC := hipcc
HIPCC_FLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-result

SRC_DIR = src
BUILD_DIR = build

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

TARGET = jane

all: $(TARGET)

$(TARGET): $(OBJ)
	$(HIPCC) $(HIPCC_FLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(HIPCC) $(HIPCC_FLAGS) -dc --hip-link -fgpu-rdc -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET)

.PHONY: all clean
