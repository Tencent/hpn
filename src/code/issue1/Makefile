CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra
TARGET := sm_budget_model
TEST_TARGET := test_runner

SRCS := main.cpp sm_budget_model.cpp bandwidth_model.cpp
TEST_SRCS := ../../test/issue1/test.cpp sm_budget_model.cpp bandwidth_model.cpp

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

clean:
	rm -f $(TARGET) $(TEST_TARGET)
