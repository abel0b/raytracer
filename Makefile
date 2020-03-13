OGL_FLAGS=#-lGL -lGLU -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3


rt: rt.cpp
	g++ rt.cpp -fopenmp -lGL -lGLU -I/usr/include/ -L/usr/lib/x86_64-linux-gnu/ -lglut -Wall -O3 -o rt

dbg: rt.cpp
	g++ -Wall -g -O0 rt.cpp -o rt_dbg

acc: rt.cpp
	pgc++ -Minfo=accel -ta=tesla:lineinfo -Minfo rt.cpp -o rt_acc  -O2 ${OGL_FLAGS} -o rt_acc


clean:
	rm -f rt rt_dbg rt_acc

run:
	./a.out sponza
