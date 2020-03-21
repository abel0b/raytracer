#include <iostream>
#include <algorithm>
#include <limits>
#include<iostream>
#include<fstream>
#include <functional>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct float3 {
    float3() { }
    float3(float xx, float yy, float zz) { x = xx; y = yy; z = zz; }
    float3(float xyz) { x = xyz; y = xyz; z = xyz; }

    float3 operator+(const float3 &f2) const { 
        return float3(x+f2.x, y+f2.y, z+f2.z); 
    }
    float3 operator-(const float3 &f2) const { 
        return float3(x-f2.x, y-f2.y, z-f2.z); 
    }
    float3 operator*(float f) const { 
      return float3(x*f, y*f, z*f); 
    }
    float3 operator*(const float3 &f2) const { 
        return float3(x*f2.x, y*f2.y, z*f2.z); 
    }
    float x, y, z;
};

struct Triangle {
    float p[3][4]; // extra float pad after each vertex
    int32_t id;
    float area();
};

float Triangle::area() {
    float a = sqrt((p[0][0]-p[1][0])*(p[0][0]-p[1][0]) + (p[0][1]-p[1][1])*(p[0][1]-p[1][1] + (p[0][2]-p[1][2])*(p[0][1]-p[1][2])));
    float b = sqrt((p[1][0]-p[2][0])*(p[1][0]-p[2][0]) + (p[1][1]-p[1][1])*(p[1][1]-p[2][1] + (p[1][2]-p[2][2])*(p[1][1]-p[2][2])));
    float c = sqrt((p[2][0]-p[0][0])*(p[2][0]-p[0][0]) + (p[2][1]-p[0][1])*(p[2][1]-p[0][1] + (p[2][2]-p[0][2])*(p[2][1]-p[0][2])));
    float s = a + b + c;
    return sqrt(s*(s-a)*(s-b)*(s-c));
}

struct LinearBVHNode {
    float bounds[2][3];
    int32_t offset;     // primitives for leaf, second child for interior
    uint8_t n_primitives;
    uint8_t split_axis;
    LinearBVHNode();
};

LinearBVHNode::LinearBVHNode() {

}

struct BVH {
    constexpr static float Ttri = 4.0; // time tocompute a ray-triangle intersection
    constexpr static float Taabb = 1.0; // time to test a ray and an AABB for intersection
    std::vector<LinearBVHNode> nodes;
    std::vector<Triangle> triangles;
    BVH(std::vector<Triangle> triangles);
    void write(std::string filename);
};

void BVH::write(std::string filename) {
    std::ofstream wf(filename, std::ios::out | std::ios::binary);

    unsigned int n_nodes = nodes.size();
    wf.write((char*)&n_nodes, sizeof(unsigned int));
    for(auto node: nodes) {
        wf.write((char*)&node.bounds[0][0], sizeof(float));
        wf.write((char*)&node.bounds[0][1], sizeof(float));
        wf.write((char*)&node.bounds[0][2], sizeof(float));
        wf.write((char*)&node.bounds[1][0], sizeof(float));
        wf.write((char*)&node.bounds[1][1], sizeof(float));
        wf.write((char*)&node.bounds[1][2], sizeof(float));
        wf.write((char*)&node.offset, sizeof(float));
        wf.write((char*)&node.n_primitives, sizeof(uint8_t));
        wf.write((char*)&node.split_axis, sizeof(uint8_t));
        uint16_t pad_dummy;
        wf.write((char*)&pad_dummy, sizeof(uint16_t));
    }

    unsigned int n_tris = triangles.size();
    wf.write((char*)&n_tris, sizeof(unsigned int));
    for (unsigned int i = 0; i < n_tris; ++i) {
        wf.write((char*)&triangles[i].p[0][0], sizeof(float));
        wf.write((char*)&triangles[i].p[0][1], sizeof(float));
        wf.write((char*)&triangles[i].p[0][2], sizeof(float));
        wf.write((char*)&triangles[i].p[1][0], sizeof(float));
        wf.write((char*)&triangles[i].p[1][1], sizeof(float));
        wf.write((char*)&triangles[i].p[1][2], sizeof(float));
        wf.write((char*)&triangles[i].p[2][0], sizeof(float));
        wf.write((char*)&triangles[i].p[2][1], sizeof(float));
        wf.write((char*)&triangles[i].p[2][2], sizeof(float));
    }

    wf.close();
}

// Axis-aligned minimum bounding box
struct AABB {
    float3 bb_min;
    float3 bb_max;
	AABB();
	AABB(float3& bb_min, float3& bb_max);
    void insert(Triangle& tri);
    float area();
};
	
AABB::AABB() : bb_min(float3(std::numeric_limits<float>::max())), bb_max(float3(std::numeric_limits<float>::min())) {


}

AABB::AABB(float3& bb_min, float3& bb_max) : bb_min(bb_min), bb_max(bb_max) {

}

void AABB::insert(Triangle& tri) {
    for (int vertex=0; vertex<3; vertex++) {
        bb_min.x = std::min(bb_min.x, tri.p[vertex][0]);
        bb_min.y = std::min(bb_min.y, tri.p[vertex][1]);
        bb_min.z = std::min(bb_min.z, tri.p[vertex][2]);
        bb_max.x = std::max(bb_max.x, tri.p[vertex][0]);
        bb_max.y = std::max(bb_max.y, tri.p[vertex][1]);
        bb_max.z = std::max(bb_max.z, tri.p[vertex][2]);
    }
}

float AABB::area() {
    return 2*((bb_max.x-bb_min.x + bb_max.z-bb_min.z)*(bb_max.y-bb_min.y) + (bb_max.x-bb_min.x)*(bb_max.z-bb_min.z));
}

// TODO: quick sort
template<typename T, typename Comparator>
void sort(T * arr, int start, int end, Comparator cmp) {
	for (int i = start; i < end; i++) {
		for (int j = start; j < end; j++) {
			if (cmp(&arr[j], &arr[i])) {
				T tmp = arr[i];
				arr[i] = arr[j];
				arr[j] = tmp;
			}
		}
	}
}

// Surface Area Heuristic
float SAH(int ns1, int ns2, float left_area, float right_area, float total_area) {
    return 2*BVH::Taabb + left_area/total_area*ns1*BVH::Ttri + right_area/total_area*ns2*BVH::Ttri;
}

// Implements algorithm described in https://graphics.stanford.edu/~boulos/papers/togbvh.pdf
void partition_sweep(std::vector<LinearBVHNode>& nodes, Triangle * triangles, int start, int end) {
    float best_cost = BVH::Ttri*(end-start);
    float best_axis = -1;
    float best_event = -1;

    std::cout << "start=" << start << " end=" << end << std::endl; 

    AABB overall_box;

    for (int axis = 0; axis<3; axis++) {
        sort(
            triangles,
            start,
            end,
            [axis](const Triangle * tri1, const Triangle * tri2) -> bool {
                return (tri1->p[0][axis]+tri1->p[1][axis]+tri1->p[2][axis]) > (tri2->p[0][axis]+tri2->p[1][axis]+tri2->p[2][axis]); 
            }
        );

        AABB left_area[end-start];
        AABB right_area[end-start];

        {
            AABB box;
            for(int i = 0; i<end-start; ++i) {
                left_area[i] = box;
                box.insert(triangles[start+i]);
                overall_box.insert(triangles[start+i]);
            }
        }

        {
            AABB box;
            for(int i = end-start-1; i>=0; --i) {
                right_area[i] = box;
                box.insert(triangles[i]);
                float this_cost = SAH(i, end-start-i, left_area[i].area(), right_area[i].area(), overall_box.area());
                if (this_cost < best_cost) {
                    best_cost = this_cost;
                    best_event = start+i;
                    best_axis = axis;
                }
            }
        }
    }

    LinearBVHNode node;
    node.bounds[0][0] = overall_box.bb_min.x;
    node.bounds[0][1] = overall_box.bb_min.y;
    node.bounds[0][2] = overall_box.bb_min.z;
    node.bounds[1][0] = overall_box.bb_max.x;
    node.bounds[1][1] = overall_box.bb_max.y;
    node.bounds[1][2] = overall_box.bb_max.z;
    node.split_axis = best_axis;
    int node_num = nodes.size();

    if (best_axis == -1) {
        // Make a leaf node
        node.n_primitives = end-start;
        node.offset = start; // primitive offset
        nodes.push_back(node);
    }
    else {
        // Make an inner node
        int axis = best_axis;
        sort(
            triangles,
            start,
            end,
            [axis](const Triangle * tri1, const Triangle * tri2) -> bool {
                return (tri1->p[0][axis]+tri1->p[1][axis]+tri1->p[2][axis]) > (tri2->p[0][axis]+tri2->p[1][axis]+tri2->p[2][axis]);
            }
        );
    
        node.n_primitives = 0;
        nodes.push_back(node);

        partition_sweep(nodes, triangles, start, best_event);
        nodes[node_num].offset = nodes.size();
        partition_sweep(nodes, triangles, best_event, end);
    }
    
}

BVH::BVH(std::vector<Triangle> triangles): triangles(triangles) {
    partition_sweep(nodes, triangles.data(), 0, (int)triangles.size());
    std::cout << "bvh nodes=" << nodes.size() << std::endl;
}

int main(int argc, char * argv[]) {
    if (argc < 2) {
        std::cout << "Error: invalid arguments" << std::endl;
        std::cout << "Usage: " << argv[0] << " <obj_file>" << std::endl;
        exit(1);
    }

    std::cout << "Loading obj" << std::endl;

    std::string inputfile(argv[1]);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    std::vector<Triangle> triangles;

    int32_t next_id = 1;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        std::cout << "Nb faces " << shapes[s].mesh.num_face_vertices.size() << std::endl;
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            assert(fv == 3); // Triangle only

            Triangle tri;

            // Loop over vertices in the face.
            for (int v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
                tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
                tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
                // tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
                // tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
                // tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];
                // tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
                // tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
                // Optional: vertex colors
                // tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
                // tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
                // tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
                tri.p[v][0] = vx;
                tri.p[v][1] = vy;
                tri.p[v][2] = vz;
            }
            tri.id = next_id++;
            triangles.push_back(tri);

            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    std::cout << "Loaded obj" << std::endl;
    BVH bvh(triangles);
    bvh.write("out.bvh");

    return EXIT_SUCCESS;
}