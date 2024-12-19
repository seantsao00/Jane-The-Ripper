HIPCC := hipcc
HIPCC_FLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-result -fgpu-rdc

SRC_DIR = src
BUILD_DIR = build

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

TARGET = jane

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling $@"
	$(HIPCC) $(HIPCC_FLAGS) $^ -o $@

# $(TARGET): $(OBJ)
# 	@echo "Linking $@"
# 	$(HIPCC) $(HIPCC_FLAGS) --hip-link $^ -o $@

# $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
# 	@mkdir -p $(BUILD_DIR)
# 	@echo "Compiling $<"
# 	$(HIPCC) $(HIPCC_FLAGS) -dc $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
