#define MAX(a,b) ((a)<(b)?(b):(a))
#define MIN(a,b) ((a)<(b)?(a):(b))

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <stdint.h>

#include <GL/gl.h>
#include <GL/glut.h>   // freeglut.h might be a better alternative, if available.
 
#ifdef M_PI
#undef M_PI
#endif
#define M_PI 3.1415926535f

#define NAO_SAMPLES 8

#define _NOPADDING_NOPUDDING

/* TODO LIST :
 * - Placement caméra par scène, sinon, ça veut rien dire...
 * - Nouvelles scènes ?
 * - Réactiver le padding, sauf si debug
 * - Eclairage : définir une couleur de lampe 
 * - Nombre d'échantillons AO en paramètre
 */

/* Min et max coord de la scène pour chaque direction */
float max_x = -10000000.0;
float max_y = -10000000.0;
float max_z = -10000000.0;
float min_x = 10000000.0;
float min_y = 10000000.0;
float min_z = 10000000.0;

/* Norme de la diagonale de la scène. Utilisée pour adimensioner les calculs d'occlusion*/
float norm_diag = 0.0f;


float camera2world[4][4], raster2camera[4][4];

/* light[1] := position en x de la lumière */
/* light[2] := position en y de la lumière */
/* light[3] := position en z de la lumière */
/* light[4] := intensité de la lumière */
/* C'est quand même mieux commenté que camera2world et raster2camera... */
float light[4]; 

// Just enough of a float3 class to do what we need in this file.
struct float3 {
    float3() { }
    float3(float xx, float yy, float zz) { x = xx; y = yy; z = zz; }

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
    // float pad;  // match padding/alignment of ispc version 
};

struct Isect {
    float      t;
    float3        p;
    float3        n;
    int        hit; 
};

struct Ray {
    float3 origin, dir, invDir;
    float3 hit_n, hit_p;
    Isect i;
    unsigned int dirIsNeg[3];
    float mint, maxt;
    int hitId;
};

// Declare these in a namespace so the mangling matches
namespace ispc {
    struct Triangle {
        float p[3][4]; // extra float pad after each vertex
        int32_t id;
#ifndef _NOPADDING_NOPUDDING
        int32_t pad[3]; // make 16 x 32-bits
#endif
    };

    struct LinearBVHNode {
        float bounds[2][3];
        int32_t offset;     // primitives for leaf, second child for interior
        uint8_t nPrimitives;
        uint8_t splitAxis;
#ifndef _NOPADDING_NOPUDDING
        uint16_t pad;
#endif
    };

    struct Light {
        Light() : position(), intensity() {}
        Light(const float3 &p, const float &i) : position(p), intensity(i) {}
        float3 position;
        float intensity;
    };

}

using namespace ispc;

typedef unsigned int uint;


inline float Dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline void vnormalize(float3 &v) {
    float len2 = Dot(v, v);
    float invlen = 1.f / sqrtf(len2);
    v = v * invlen;
}


inline float3 Cross(const float3 &v1, const float3 &v2) {
    float v1x = v1.x, v1y = v1.y, v1z = v1.z;
    float v2x = v2.x, v2y = v2.y, v2z = v2.z;
    float3 ret;
    ret.x = (v1y * v2z) - (v1z * v2y);
    ret.y = (v1z * v2x) - (v1x * v2z);
    ret.z = (v1x * v2y) - (v1y * v2x);
    return ret;
}

static void generateRay(const float raster2camera[4][4], 
                        const float camera2world[4][4],
                        float x, float y, Ray &ray) {
    ray.mint = 0.f;
    ray.maxt = 1e30f;

    ray.hitId = 0;

    // transform raster coordinate (x, y, 0) to camera space
    float camx = raster2camera[0][0] * x + raster2camera[0][1] * y + raster2camera[0][3];
    float camy = raster2camera[1][0] * x + raster2camera[1][1] * y + raster2camera[1][3];
    float camz = raster2camera[2][3];
    float camw = raster2camera[3][3];
    camx /= camw;
    camy /= camw;
    camz /= camw;

    ray.dir.x = camera2world[0][0] * camx + camera2world[0][1] * camy +
        camera2world[0][2] * camz;
    ray.dir.y = camera2world[1][0] * camx + camera2world[1][1] * camy +
        camera2world[1][2] * camz;
    ray.dir.z = camera2world[2][0] * camx + camera2world[2][1] * camy +
        camera2world[2][2] * camz;

    vnormalize(ray.dir);

    ray.origin.x = camera2world[0][3] / camera2world[3][3];
    ray.origin.y = camera2world[1][3] / camera2world[3][3];
    ray.origin.z = camera2world[2][3] / camera2world[3][3];

    ray.invDir.x = 1.f / ray.dir.x;
    ray.invDir.y = 1.f / ray.dir.y;
    ray.invDir.z = 1.f / ray.dir.z;

    ray.dirIsNeg[0] = (ray.invDir.x < 0) ? 1 : 0;
    ray.dirIsNeg[1] = (ray.invDir.y < 0) ? 1 : 0;
    ray.dirIsNeg[2] = (ray.invDir.z < 0) ? 1 : 0;
}


static inline bool BBoxIntersect(const float bounds[2][3], 
                                 const Ray &ray) {
    float3 bounds0(bounds[0][0], bounds[0][1], bounds[0][2]);
    float3 bounds1(bounds[1][0], bounds[1][1], bounds[1][2]);
    float t0 = ray.mint, t1 = ray.maxt;

    float3 tNear = (bounds0 - ray.origin) * ray.invDir;
    float3 tFar  = (bounds1 - ray.origin) * ray.invDir;
    if (tNear.x > tFar.x) {
        float tmp = tNear.x;
        tNear.x = tFar.x;
        tFar.x = tmp;
    }
    t0 = std::max(tNear.x, t0);
    t1 = std::min(tFar.x, t1);

    if (tNear.y > tFar.y) {
        float tmp = tNear.y;
        tNear.y = tFar.y;
        tFar.y = tmp;
    }
    t0 = std::max(tNear.y, t0);
    t1 = std::min(tFar.y, t1);

    if (tNear.z > tFar.z) {
        float tmp = tNear.z;
        tNear.z = tFar.z;
        tFar.z = tmp;
    }
    t0 = std::max(tNear.z, t0);
    t1 = std::min(tFar.z, t1);
    
    return (t0 <= t1);
}



inline bool TriIntersect(const Triangle &tri, Ray &ray) {
    float3 p0(tri.p[0][0], tri.p[0][1], tri.p[0][2]);
    float3 p1(tri.p[1][0], tri.p[1][1], tri.p[1][2]);
    float3 p2(tri.p[2][0], tri.p[2][1], tri.p[2][2]);
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;


    float3 s1 = Cross(ray.dir, e2);
    float divisor = Dot(s1, e1);

    if (divisor == 0.)
        return false;
    float invDivisor = 1.f / divisor;

    // Compute first barycentric coordinate
    float3 d = ray.origin - p0;
    float b1 = Dot(d, s1) * invDivisor;
    if (b1 < 0. || b1 > 1.)
        return false;

    // Compute second barycentric coordinate
    float3 s2 = Cross(d, e1);
    float b2 = Dot(ray.dir, s2) * invDivisor;
    if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Compute _t_ to intersection point
    float t = Dot(e2, s2) * invDivisor;
    if (t < ray.mint || t > ray.maxt)
        return false;

    //fprintf(stderr, "e1=(%f,%f,%f), e2=(%f,%f,%f)\n",
    //        e1.x, e1.y, e1.z,
    //        e2.x, e2.y, e2.z);
 
    ray.maxt = t;
    ray.hitId = tri.id;

    ray.i.n = float3(0,0,0);
    ray.i.p = float3(0,0,0);

    ray.hit_p = ray.origin + ray.dir*t;
    ray.hit_n = Cross(e1, e2);
    /* 
     *   Ici, on s'assure que la normale ne sort pas de la boiboite.
     *   Astuce de sioux : on essaye de s'assurer que les vecteurs p et n pointent
     *   dans des sens opposés (si on est à l'extérieur d'une boite, ça marche aussi...)
     */
        float dirInv = Dot(ray.dir,ray.hit_n);
        if(dirInv > 0.f) 
	  // Si le produit scalaire est positif, les vecteurs vont dans le meme sens
	  // On s'assure de vérifier l'inverse
          ray.hit_n = Cross(e2, e1);

    return true;
}


bool BVHIntersect(const LinearBVHNode nodes[], const Triangle tris[], 
                  Ray &r) {
    Ray ray = r;
    bool hit = false;
    // Follow ray through BVH nodes to find primitive intersections
    int todoOffset = 0, nodeNum = 0;
    int todo[64];

    while (true) {
        // Check ray against BVH node
        const LinearBVHNode &node = nodes[nodeNum];
        if (BBoxIntersect(node.bounds, ray)) {
            unsigned int nPrimitives = node.nPrimitives;
            if (nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                unsigned int primitivesOffset = node.offset;
                for (unsigned int i = 0; i < nPrimitives; ++i) {
                    if (TriIntersect(tris[primitivesOffset+i], ray))
                    {
                        hit = true;
                    }
                }
                if (todoOffset == 0) 
                    break;
                nodeNum = todo[--todoOffset];
            }
            else {
                // Put far BVH node on _todo_ stack, advance to near node
                if (r.dirIsNeg[node.splitAxis]) {
                   todo[todoOffset++] = nodeNum + 1;
                   nodeNum = node.offset;
                }
                else {
                   todo[todoOffset++] = node.offset;
                   nodeNum = nodeNum + 1;
                }
            }
        }
        else {
            if (todoOffset == 0)
                break;
            nodeNum = todo[--todoOffset];
        }
    }
    r.maxt = ray.maxt;
    r.hitId = ray.hitId;
    r.i = ray.i;
    r.i.hit = hit;
    r.hit_p = ray.hit_p;
    r.hit_n = ray.hit_n;

    return hit;
}

static inline void orthoBasis(float3 basis[3], const float3 &n) {
    basis[2] = n;
    basis[1].x = 0.0;
    basis[1].y = 0.0;
    basis[1].z = 0.0;

    if ((n.x < 0.6f) && (n.x > -0.6f)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    basis[0] = Cross(basis[1], basis[2]);
    vnormalize(basis[0]);

    basis[1] = Cross(basis[2], basis[0]);
    vnormalize(basis[1]);
}

void make_ray(float3 p, float3 dir, Ray  *ray)
{
            ray->origin = p;
            ray->dir = dir;

	    ray->invDir.x = 1.f / ray->dir.x;
	    ray->invDir.y = 1.f / ray->dir.y;
	    ray->invDir.z = 1.f / ray->dir.z;

	    ray->dirIsNeg[0] = (ray->invDir.x < 0) ? 1 : 0;
	    ray->dirIsNeg[1] = (ray->invDir.y < 0) ? 1 : 0;
	    ray->dirIsNeg[2] = (ray->invDir.z < 0) ? 1 : 0;

	    ray->mint = 0.f;
	    ray->maxt = 1e30f;
}

static float ambient_occlusion(const LinearBVHNode nodes[], const Triangle tris[], Ray r) {
    float eps = 0.001f;
    float3 p, n;
    float3 basis[3];
    float occlusion = 0.0;

    /* On s'éloigne un peu du point d'impact le long de la normale en ce point */
    n = r.hit_n;
    vnormalize(n);

    p = r.hit_p + n * eps; 


    orthoBasis(basis, n);


    static const int ntheta = NAO_SAMPLES;
    static const int nphi = NAO_SAMPLES;

    for (int j = 0; j < ntheta; j++) {
        for (int i = 0; i < nphi; i++) {
            Ray ray;

            float theta = sqrtf(drand48());
            float phi = 2.0f * M_PI * drand48();
            float x = cosf(phi) * theta;
            float y = sinf(phi) * theta;
            float z = sqrtf(1.0f - theta * theta);

            // local . global
            float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
            float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
            float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

	    float3 dir(rx,ry,rz);
	    make_ray(p, dir, &ray);

	    bool hit = BVHIntersect(nodes, tris, ray);
            if (hit && sqrtf(ray.maxt) < 0.005*norm_diag){
	      occlusion += 1.0f;
	      //printf("HIT, occlusion = %f\n", occlusion);
	    }
        }
    }

    occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
    return occlusion;
}

float light_and_shadows(const LinearBVHNode nodes[], const Triangle tris[], Ray r)
{
    Light L;
    float lightning = 0.0;

    L.position.x = light[0];
    L.position.y = light[1];
    L.position.z = light[2];
    L.intensity  = light[3];
    
    for(int i=0; i<1; i++){
      float3 point = r.hit_p;
      float3 light_dir = L.position - point;
      float light_distance = sqrtf(Dot(light_dir,light_dir));
      float3 n = r.hit_n;
      vnormalize(light_dir);
      vnormalize(n);

      // checking if the point lies in the shadow of the lights[i]
      float3 shadow_orig = (Dot(light_dir,n) < 0) ? point - n*1e-3 : point + n*1e-3;
      float3 shadow_pt, shadow_N;
      Ray ray;
      make_ray(shadow_orig,light_dir,&ray);
      float dotlight = 1.0f;
      float shadowing = 1.0f;
      if ( BVHIntersect(nodes, tris, ray)){
	float3 dist = ray.hit_p-shadow_orig;
	if(sqrtf(Dot(dist,dist)) < light_distance){
	  shadowing = 0.15;
	}
      }
      dotlight = Dot(light_dir,n);
      if(dotlight>=0)
	lightning += L.intensity*dotlight*shadowing;
    }
    lightning  = std::min(1.0f, lightning);
    return lightning;
}

void raytrace_serial(int width, int height, int baseWidth, int baseHeight,
                     const float raster2camera[4][4], 
                     const float camera2world[4][4],
                     float image[],
                     int id[],
                     const LinearBVHNode nodes[], int nnodes,
                     const Triangle triangles[], int ntriangles) {
    float widthScale = float(baseWidth) / float(width);
    float heightScale = float(baseHeight) / float(height);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#else
#pragma acc parallel loop copyin(triangles[0:ntriangles], nodes[0:nnodes]) copy(image[0:width*height], id[0:width*height]) copyin(raster2camera[0:4][0:4], camera2world[0:4][0:4]) tile(8,8)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
                Ray ray;
                generateRay(raster2camera, camera2world, x * widthScale,
                            y * heightScale, ray);
                bool hit = BVHIntersect(nodes, triangles, ray);
                float light = 0.0;
                float occ = 1.0f;

                if (hit) {
     		        occ = ambient_occlusion(nodes, triangles, ray);
		            light = light_and_shadows(nodes, triangles, ray);
                }

                int offset = y * width + x;
                image[offset] = light*occ;
                id[offset] = ray.hitId;
        }
    }
}

extern void raytrace_serial(int width, int height, int baseWidth, int baseHeight,
                            const float raster2camera[4][4], 
                            const float camera2world[4][4], float image[],
                            int id[], const LinearBVHNode nodes[], int nnodes,
                            const Triangle triangles[], int ntriangles);

Triangle *triangles;// = new Triangle[nTris];
uint nTris;

static void writeImage(int *idImage, float *depthImage, int width, int height,
                       const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror(filename);
        exit(1);
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // use the bits from the object id of the hit object to make a
            // random color
            int id = idImage[y * width + x];
            float t = depthImage[y * width + x];
            unsigned char r = 0, g = 0, b = 0;
	    r = 0;
	    g = 0;
	    b = 0;
	    if(id){
	        r = 255.0*(t);
	        g = 255.0*(t);
	        b = 255.0*(t);
	    }
            fputc(r, f);
            fputc(g, f);
            fputc(b, f);
        }
    }            
    fclose(f);
    printf("Wrote image file %s\n", filename);
}

static void usage() {
    fprintf(stderr, "rt <scene name base> [--scale=<factor>] [ispc iterations] [tasks iterations] [serial iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    float scale = 1.f;
    const char *filename = NULL;
    int poslight=0;
    if (argc < 2) usage();
    filename = argv[1];
    if (argc > 2) {
        if (strncmp(argv[2], "--scale=", 8) == 0) {
            scale = atof(argv[2] + 8);
        }
        if (strncmp(argv[2], "--poslight", 10) == 0) {
	    poslight=1;
        }
    }

#define READ(var, n)                                            \
    if (fread(&(var), sizeof(var), n, f) != (unsigned int)n) {  \
        fprintf(stderr, "Unexpected EOF reading scene file\n"); \
        return 1;                                               \
    } else /* eat ; */                                                     

    //
    // Read the camera specification information from the camera file
    //
    char fnbuf[1024];
    sprintf(fnbuf, "%s.camera", filename);
    FILE *f = fopen(fnbuf, "rb");
    if (!f) {
        perror(fnbuf);
        return 1;
    }

    //
    // Nothing fancy, and trouble if we run on a big-endian system, just
    // fread in the bits
    //
    int baseWidth, baseHeight;
    READ(baseWidth, 1);
    READ(baseHeight, 1);
    READ(camera2world[0][0], 16);
    READ(raster2camera[0][0], 16);

    //
    // Read in the serialized BVH 
    //
    sprintf(fnbuf, "%s.bvh", filename);
    f = fopen(fnbuf, "rb");
    if (!f) {
        perror(fnbuf);
        return 1;
    }

    // The BVH file starts with an int that gives the total number of BVH
    // nodes
    uint nNodes;
    READ(nNodes, 1);

    printf("nNodes = %d\n", nNodes);

    LinearBVHNode *nodes = new LinearBVHNode[nNodes];
    for (unsigned int i = 0; i < nNodes; ++i) {
        // Each node is 6x floats for a boox, then an integer for an offset
        // to the second child node, then an integer that encodes the type
        // of node, the total number of int it if a leaf node, etc.
        float b[6];
	uint16_t pad_dummy;
        READ(b[0], 6);
        nodes[i].bounds[0][0] = b[0];
        nodes[i].bounds[0][1] = b[1];
        nodes[i].bounds[0][2] = b[2];
        nodes[i].bounds[1][0] = b[3];
        nodes[i].bounds[1][1] = b[4];
        nodes[i].bounds[1][2] = b[5];
        READ(nodes[i].offset, 1);
        READ(nodes[i].nPrimitives, 1);
        READ(nodes[i].splitAxis, 1);
#ifndef _NOPADDING_NOPUDDING
        READ(nodes[i].pad, 1);
#else
        READ(pad_dummy, 1);

#endif
    }

    // And then read the triangles 
    // uint nTris;
    READ(nTris, 1);
    /*Triangle * */triangles = new Triangle[nTris];
    for (uint i = 0; i < nTris; ++i) {
        // 9x floats for the 3 vertices
        float v[9];
        READ(v[0], 9);
        float *vp = v;
        for (int j = 0; j < 3; ++j) {
            triangles[i].p[j][0] = *vp++;
            triangles[i].p[j][1] = *vp++;
            triangles[i].p[j][2] = *vp++;
        }
        // And create an object id
        triangles[i].id = i+1;
    }

    char tmp;
    assert(fread(&tmp, sizeof(char), 1, f) != 1);

    fclose(f);


    uint i;
    for (i = 0; i < nTris; i++) {
        max_x = MAX(max_x, triangles[i].p[0][0]);
        max_x = MAX(max_x, triangles[i].p[0][1]);
        max_x = MAX(max_x, triangles[i].p[0][2]);

        max_y = MAX(max_y, triangles[i].p[1][0]);
        max_y = MAX(max_y, triangles[i].p[1][1]);
        max_y = MAX(max_y, triangles[i].p[1][2]);

        max_z = MAX(max_z, triangles[i].p[2][0]);
        max_z = MAX(max_z, triangles[i].p[2][1]);
        max_z = MAX(max_z, triangles[i].p[2][2]);

        min_x = MIN(min_x, triangles[i].p[0][0]);
        min_x = MIN(min_x, triangles[i].p[0][1]);
        min_x = MIN(min_x, triangles[i].p[0][2]);

        min_y = MIN(min_y, triangles[i].p[1][0]);
        min_y = MIN(min_y, triangles[i].p[1][1]);
        min_y = MIN(min_y, triangles[i].p[1][2]);

        min_z = MIN(min_z, triangles[i].p[2][0]);
        min_z = MIN(min_z, triangles[i].p[2][1]);
        min_z = MIN(min_z, triangles[i].p[2][2]);
    }

    float3 diag(max_x-min_x, max_y - min_y, max_z - min_z);
    fprintf(stderr, "MIN ET MAX X %f %f\n", min_x, max_x);
    fprintf(stderr, "MIN ET MAX Y %f %f\n", min_y, max_y);
    fprintf(stderr, "MIN ET MAX Z %f %f\n", min_z, max_z);
    
    norm_diag = sqrtf(Dot(diag, diag));
    printf("Longueur de la diagonale de la scène : %f\n", norm_diag);
 
    int height = int(baseHeight * scale);
    int width = int(baseWidth * scale);

    /* Par defaut, on met la lumière au milieu de la boiboite et l'intensité à 1*/
    light[0] = (max_x - min_x)/2.;
    light[1] = (max_y - min_y)/2.;
    light[2] = (max_z - min_z)/2.;
    light[3] = 1.0f;

    // allocate images; one to hold hit object ids, one to hold depth to
    // the first interseciton
    int *id = new int[width*height];
    float *image = new float[width*height];

    memset(id, 0, width*height*sizeof(int));
    memset(image, 0, width*height*sizeof(float));

    // reset_and_start_timer();
    do{
        if(poslight){
	        fprintf(stderr, "normalized light pos x y z and intensity?\n");
	        scanf("%f %f %f %f", &light[0], &light[1], &light[2], &light[3]);
        }
        raytrace_serial(width, height, baseWidth, baseHeight, raster2camera, 
                        camera2world, image, id, nodes, nNodes, triangles, nTris);
        writeImage(id, image, width, height, "rt-serial.ppm");
    } while(poslight);

    return 0;
}



#if 0
double rotate_by_key=0;

double rotate_x=0.5;

// angle of rotation for the camera direction
float angle=0.0;
// actual vector representing the camera's direction
float lx=0.0f;
float ly=1.0f;
float lz=0.0f;
// XZ position of the camera
float x=0.0f;
float z=0.0f;
float y = -20.0f;

void reshape ( int width, int height ) {

    /* define the viewport transformation */
    glViewport(0,0,width,height);

}

void processNormalKeys(unsigned char key, int x, int y) {
    if (key == 27)
        exit(0);
}

void display() {  // Display function will draw the image.
 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1.5,1.5,-1.5,1.5,1.0,100.0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 1.0f);



    fprintf(stderr, "UP %f %f %f\n", camera2world[0][1], camera2world[1][1], camera2world[2][1]);
    
    // Set the camera
    gluLookAt(	x, y, z,
		x + lx, y + ly, z + lz,
		camera2world[0][1], camera2world[1][1], camera2world[2][1]);
    
    /* future matrix manipulations should affect the modelview matrix */
    glMatrixMode(GL_MODELVIEW);
    
    
    //glPushMatrix();
    int i;
    for (i = 0; i < nTris; i++)
      {
    	glBegin(GL_TRIANGLES);
        float rel_y;
        rel_y = (triangles[i].p[0][1] - min_y)/(max_y - min_y);
    	glColor3f( rel_y, rel_y, rel_y ); // red
    	glVertex3f( 1.0*triangles[i].p[0][0], 1.0*triangles[i].p[0][1], 1.0*triangles[i].p[0][2]);
	
        rel_y = (triangles[i].p[1][1] - min_y)/(max_y - min_y);
    	glColor3f( rel_y, rel_y, rel_y ); // red
    	glVertex3f( 1.0*triangles[i].p[1][0], 1.0*triangles[i].p[1][1], 1.0*triangles[i].p[1][2]);
	
        rel_y = (triangles[i].p[2][1] - min_y)/(max_y - min_y);
    	glColor3f( rel_y, rel_y, rel_y ); // red
    	glVertex3f( 1.0*triangles[i].p[2][0], 1.0*triangles[i].p[2][1], 1.0*triangles[i].p[2][2]);
    	glEnd(); 
      }
    
    /// glPopMatrix();
    glFlush();
    
    fprintf(stderr, "MAX X %f %f\n", min_x, max_x);
    fprintf(stderr, "MAX Y %f %f\n", min_y, max_y);
    fprintf(stderr, "MAX Z %f %f\n", min_z, max_z);
    fprintf(stderr, "POS %f,%f,%f VERS %f,%f,%f \n", x, y, z, x + lx, y + ly, z + lz);
    
    // lx = camera2world[0][2] - x;

    glutSwapBuffers(); // Required to copy color buffer onto the screen.
 
}



void processSpecialKeys(int key, int xx, int yy) {

	float fraction = .1f;
	float fraction_angle = 2*3.14/100.0f;

	switch (key) {
		case GLUT_KEY_LEFT :
			//angle -= fraction_angle;
			//lx = sinf(angle);
			//lz = cosf(angle);
                        y += fraction;
			break;
		case GLUT_KEY_RIGHT :
			//angle += fraction_angle;
			//lx = sinf(angle);
			//lz = cosf(angle);
                        y -= fraction;
			break;
		case GLUT_KEY_UP :
			//x += lx * fraction;
			//z += lz * fraction;
                        x += fraction;
			break;
		case GLUT_KEY_DOWN :
			//x -= lx * fraction;
			//z -= lz * fraction;
                        x -= fraction;
			break;
                 case GLUT_KEY_PAGE_DOWN:
                        z += fraction;
			break;
	
                 case GLUT_KEY_PAGE_UP:
                        z -= fraction;
			break;
	}
	fprintf(stderr, "X  %f Y %f Z %f angle %f\n", x, y, z, angle);

	display();
}
#endif

/*
  Cornell : 280. 475. 200. 1.0
  Teapot  : 18. 10. 6. 1.
  Sponza  : -4. 0. 10. 1.
 */
