CXX := nvcc
CXXFLAGS := -std=c++17 -O3 -lineinfo -arch=sm_70
LDFLAGS := -lcudart

TARGET := phased_alltoall_sim
TEST_TARGET := test_runner

SRCS := main.cpp phased_alltoall.cpp workload_simulator.cpp
TEST_SRCS := ../../test/issue3/test.cpp phased_alltoall.cpp workload_simulator.cpp

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -x cu -o $@ $^ $(LDFLAGS)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) -x cu -I. -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TEST_TARGET)
