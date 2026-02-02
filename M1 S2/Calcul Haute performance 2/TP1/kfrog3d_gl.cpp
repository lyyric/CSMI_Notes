// kfrog3d_gl.cpp
// 3D wave equation (leapfrog) + OpenMP + OpenGL (GLFW+GLEW) 2D slice visualization

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// --------- OpenGL / GLFW / GLEW ----------
#include <GL/glew.h>
#include <GLFW/glfw3.h>

static void die(const char* msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(1);
}

// -----------------------------
// Simulation parameters
// -----------------------------
static const double L = 1.0;
static const double c = 1.0;
static const double cfl = 0.35;

// absorption-ish parameter (kept similar spirit; 3D boundary handling simplified)
static const double r0 = 1.0;
static const double gam = (1.0 - r0) / (1.0 + r0);

// grid (you can pass via argv)
static int nx = 128, ny = 128, nz = 128;

static double dx, dy, dz, dt;
static double bx, by, bz;

inline int idx3(int i, int j, int k) {
    return (i * ny + j) * nz + k;
}

// peak function (same as your python idea)
inline double peak(double x) {
    const double eps = 0.2;
    const double r2  = x * x;
    const double eps2 = eps * eps;
    if (r2 / eps2 < 1.0) return std::pow(1.0 - r2 / eps2, 4);
    return 0.0;
}

// initial condition: radial bump centered at (0.5,0.5,0.5)
inline double exact_sol(double x, double y, double z) {
    double r = std::sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5));
    return peak(r);
}

inline double source(double, double, double, double) { return 0.0; }

void init_sol(std::vector<double>& un, std::vector<double>& unm1) {
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                double x = i * dx, y = j * dy, z = k * dz;
                double u0 = exact_sol(x, y, z);
                un[idx3(i,j,k)]   = u0;
                unm1[idx3(i,j,k)] = u0;
            }
}

// 3D leapfrog step with 6-neighbor stencil.
// Boundary: "mirror-like" clamp to inner cell (similar to your python fallback to 1/n-2)
void leapfrog_step_3d_omp(const std::vector<double>& un,
                         const std::vector<double>& unm1,
                         std::vector<double>& unp1,
                         double t)
{
    // OpenMP strategy:
    // - collapse(2): parallelize tiles of (i,j), keep k innermost for contiguous memory along k
    // - or collapse(3) if you want maximum granularity (sometimes more overhead)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {

            for (int k = 0; k < nz; ++k) {

                // boundary coefficient a (a very light version of your 2D idea)
                // If any index on boundary, damp slightly with bx/by/bz. (TP-style "absorbing-ish".)
                double a = 1.0;
                if (i == 0 || i == nx-1) a = std::min(a, 1.0 / (1.0 + bx * gam));
                if (j == 0 || j == ny-1) a = std::min(a, 1.0 / (1.0 + by * gam));
                if (k == 0 || k == nz-1) a = std::min(a, 1.0 / (1.0 + bz * gam));

                // neighbor indices with clamp to interior
                int iL = (i-1 < 0)   ? 1      : i-1;
                int iR = (i+1 >= nx) ? nx-2   : i+1;
                int jL = (j-1 < 0)   ? 1      : j-1;
                int jR = (j+1 >= ny) ? ny-2   : j+1;
                int kL = (k-1 < 0)   ? 1      : k-1;
                int kR = (k+1 >= nz) ? nz-2   : k+1;

                const int id  = idx3(i,j,k);
                const double uC = un[id];

                const double ux = un[idx3(iL,j,k)] + un[idx3(iR,j,k)];
                const double uy = un[idx3(i,jL,k)] + un[idx3(i,jR,k)];
                const double uz = un[idx3(i,j,kL)] + un[idx3(i,j,kR)];

                const double s = source(i*dx, j*dy, k*dz, t);

                // 3D leapfrog:
                // u^{n+1} = (1-2a)u^{n-1} + 2a(1-bx^2-by^2-bz^2)u^n + a*bx^2*(ux) + a*by^2*(uy) + a*bz^2*(uz) - dt^2 * a*s
                unp1[id] =
                    (1.0 - 2.0 * a) * unm1[id]
                    + 2.0 * a * (1.0 - bx*bx - by*by - bz*bz) * uC
                    + a * bx*bx * ux
                    + a * by*by * uy
                    + a * bz*bz * uz
                    - dt*dt * a * s;
            }
        }
    }
}

// -----------------------------
// OpenGL helpers
// -----------------------------
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, 2048, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
        die("shader compile failed");
    }
    return s;
}

GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, 2048, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
        die("program link failed");
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// simple full-screen quad (triangle strip)
void createFullscreenQuad(GLuint& vao, GLuint& vbo) {
    float verts[] = {
        // pos      // uv
        -1.f, -1.f, 0.f, 0.f,
         1.f, -1.f, 1.f, 0.f,
        -1.f,  1.f, 0.f, 1.f,
         1.f,  1.f, 1.f, 1.f
    };
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));

    glBindVertexArray(0);
}

// map slice values to bytes (grayscale); auto-scale using a robust range
void slice_to_bytes_k(const std::vector<double>& un, int k0,
                      std::vector<uint8_t>& img, double& out_min, double& out_max)
{
    img.resize(nx * ny);

    // find min/max on slice
    double mn =  1e300;
    double mx = -1e300;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            double v = un[idx3(i,j,k0)];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
    // avoid zero range
    if (std::abs(mx - mn) < 1e-12) { mx = mn + 1e-12; }

    out_min = mn;
    out_max = mx;

    // normalize to [0,255]
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            double v = un[idx3(i,j,k0)];
            double x = (v - mn) / (mx - mn);
            x = std::clamp(x, 0.0, 1.0);
            img[i*ny + j] = static_cast<uint8_t>(std::lround(255.0 * x));
        }
}

int main(int argc, char** argv)
{
    // usage: ./kfrog3d_gl nx ny nz
    if (argc >= 2) nx = std::atoi(argv[1]);
    if (argc >= 3) ny = std::atoi(argv[2]);
    if (argc >= 4) nz = std::atoi(argv[3]);

    dx = L / (nx - 1);
    dy = L / (ny - 1);
    dz = L / (nz - 1);

    // 3D CFL: dt ~ cfl * sqrt(dx^2+dy^2+dz^2)/c (similar style as your 2D code)
    dt = cfl * std::sqrt(dx*dx + dy*dy + dz*dz) / c;

    bx = c * dt / dx;
    by = c * dt / dy;
    bz = c * dt / dz;

    std::vector<double> unm1(nx*ny*nz, 0.0);
    std::vector<double> un  (nx*ny*nz, 0.0);
    std::vector<double> unp1(nx*ny*nz, 0.0);

    init_sol(un, unm1);

    // --------- GLFW init ----------
    if (!glfwInit()) die("glfwInit failed");

    // OpenGL 3.3 core (works well on mac if supported)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(900, 900, "3D Leapfrog Wave: 2D Slice", nullptr, nullptr);
    if (!win) die("glfwCreateWindow failed");
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    // --------- GLEW init ----------
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) die("glewInit failed");

    const char* vs = R"GLSL(
        #version 330 core
        layout(location=0) in vec2 aPos;
        layout(location=1) in vec2 aUV;
        out vec2 vUV;
        void main() {
            vUV = aUV;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )GLSL";

    const char* fs = R"GLSL(
        #version 330 core
        in vec2 vUV;
        out vec4 FragColor;
        uniform sampler2D uTex;
        void main() {
            float g = texture(uTex, vUV).r; // [0,1]
            FragColor = vec4(g, g, g, 1.0);
        }
    )GLSL";

    GLuint prog = makeProgram(vs, fs);
    GLuint vao=0, vbo=0;
    createFullscreenQuad(vao, vbo);

    // texture for slice (ny as width, nx as height)
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // allocate immutable size (R8)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, ny, nx, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

    int k0 = nz / 2;               // visualize this slice
    double t = 0.0;
    double tmax = 1.0;             // you can ignore for realtime

    // controls
    int steps_per_frame = 2;       // increase for faster simulation

    std::vector<uint8_t> img;
    double mn=0.0, mx=1.0;

    auto lastPrint = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        // basic key controls:
        // Up/Down to move slice, Left/Right to change speed
        if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS)   k0 = std::min(k0+1, nz-1);
        if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) k0 = std::max(k0-1, 0);
        if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) steps_per_frame = std::min(steps_per_frame+1, 50);
        if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS)  steps_per_frame = std::max(steps_per_frame-1, 1);

        // --- simulate a few steps ---
        for (int s = 0; s < steps_per_frame; ++s) {
            leapfrog_step_3d_omp(un, unm1, unp1, t);
            unm1.swap(un);
            un.swap(unp1);
            t += dt;
            if (t > tmax) t = 0.0; // loop time (optional)
        }

        // --- upload slice to texture ---
        slice_to_bytes_k(un, k0, img, mn, mx);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ny, nx, GL_RED, GL_UNSIGNED_BYTE, img.data());

        // --- draw ---
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(win);

        // print status occasionally
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - lastPrint).count() > 1.0) {
#ifdef _OPENMP
            int nt = omp_get_max_threads();
#else
            int nt = 1;
#endif
            std::cout << "t=" << t << "  slice k=" << k0
                      << "  range=[" << mn << "," << mx << "]"
                      << "  steps/frame=" << steps_per_frame
                      << "  threads=" << nt << "\n";
            lastPrint = now;
        }
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
