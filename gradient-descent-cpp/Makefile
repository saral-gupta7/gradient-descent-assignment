CXX = g++
STD = -std=c++17
FLAGS = -O2 -Wall

all: gd

gd: main.cpp model.h data.h loss.h train.h
	$(CXX) $(STD) $(FLAGS) main.cpp -o gd

run: gd
	./gd

clean:
	rm -f gd
