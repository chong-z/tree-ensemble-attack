CC=g++
COMMON_FLAGS = --std=c++17 -pthread -lboost_thread -lboost_system -lboost_filesystem
# DEBUG_CFLAGS = -g -D DEBUG $(COMMON_FLAGS)
DEBUG_CFLAGS = -g $(COMMON_FLAGS)
CFLAGS = -O2 $(COMMON_FLAGS)
DEPS = neighbor_attack.h decision_tree.h decision_forest.h bounding_box.h interval.h utility.h test.h timing.h nlohmann/json.hpp
OBJS = bounding_box.o decision_forest.o decision_tree.o interval.o neighbor_attack.o timing.o utility.o
DEBUG_OBJS = debug_bounding_box.o debug_decision_forest.o debug_decision_tree.o debug_interval.o debug_neighbor_attack.o debug_timing.o debug_utility.o

%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

debug_%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(DEBUG_CFLAGS)

all: lt_attack

debug: debug_lt_attack

test: decision_forest_test

decision_forest_test: $(OBJS) decision_forest_test.o
	$(CC) -o $@ $^ $(CFLAGS)

lt_attack: $(OBJS) lt_attack.o
	$(CC) -o $@ $^ $(CFLAGS)

debug_lt_attack: $(DEBUG_OBJS) debug_lt_attack.o
	$(CC) -o $@ $^ $(DEBUG_CFLAGS)

clean:
	rm -f *.o decision_forest_test lt_attack debug_lt_attack
