OGL_FLAGS=#-lGL -lGLU -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3
INCLUDES=-Iexternal/tinyobjloader
SCENE=teapot
CPP_FLAGS=-std=c++17 -ltbb -DDEBUG

all: update-git-submodules rt rt_dbg genbvh genbvh_dbg

.PHONY: update-git-submodules
update-git-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then\
		git submodule update --init;\
	fi

rt: rt.cpp
	g++ $(CPP_FLAGS) -fopenmp -lGL -lGLU -I/usr/include $(INCLUDES) -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3 $^ -o $@

rt_dbg: rt.cpp
	g++ $(CPP_FLAGS) -Wall -g -O0 $^ -o $@

genbvh: genbvh.cpp
	g++ $(CPP_FLAGS) -fopenmp -lGL -lGLU -I/usr/include $(INCLUDES) -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3 $^ -o $@

genbvh_dbg: genbvh.cpp
	g++ $(CPP_FLAGS) -fopenmp -lGL -lGLU -ggdb3 -I/usr/include $(INCLUDES) -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3 $^ -o $@

rt_acc: rt.cpp
	pgc++ $(CPP_FLAGS) -Minfo=accel -ta=tesla:lineinfo -Minfo $^ -O2 ${OGL_FLAGS} -o $@

.PHONY: test
test: test-genbvh

.PHONY: test-genbvh
test-genbvh: genbvh
	./genbvh teapot.obj

.PHONY: memcheck
memcheck: genbvh
	valgrind --error-exitcode=1 ./genbvh teapot.obj

.PHONY: clean
clean:
	rm -f rt rt_dbg rt_acc
