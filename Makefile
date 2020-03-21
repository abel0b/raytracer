OGL_FLAGS=#-lGL -lGLU -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3
INCLUDES=-Iexternal/tinyobjloader
SCENE=teapot

all: rt rt_dbg genbvh

rt: rt.cpp
	g++ -fopenmp -lGL -lGLU -I/usr/include $(INCLUDES) -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3 rt.cpp -o rt

genbvh: genbvh.cpp
	g++ -fopenmp -lGL -lGLU -I/usr/include $(INCLUDES) -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3 genbvh.cpp -o genbvh

rt_dbg: rt.cpp
	g++ -Wall -g -O0 rt.cpp -o rt_dbg

rt_acc: rt.cpp
	pgc++ -Minfo=accel -ta=tesla:lineinfo -Minfo rt.cpp -o rt_acc  -O2 ${OGL_FLAGS} -o rt_acc

.PHONY: test
test: memcheck

.PHONY: memcheck
memcheck: rt
	valgrind --error-exitcode=1 ./rt $(SCENE)

.PHONY: clean
clean:
	rm -f rt rt_dbg rt_acc
