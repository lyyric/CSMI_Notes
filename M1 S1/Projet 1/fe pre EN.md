
### **Page 1: Title Slide (1 minute)**

**Script:**

- Hello everyone, I’m Yehua. I’m delighted to present my project on solving a Poisson-type partial differential equation (PDE) in two dimensions using the Finite Element Method (FEM).
- Today’s presentation will cover the project’s motivation, mathematical modeling, numerical implementation, and performance analysis.
- Through this project, I aim to demonstrate the importance of combining mathematical methods and programming techniques in modern scientific computing.

---

### **Page 2: Outline (30 seconds)**

**Script:**

- Here is the outline for today’s presentation. We will proceed as follows:
  1. Introduce the project background and objectives.
  2. Explain the mathematical model of the Poisson-type equation and its weak formulation.
  3. Present the C++ implementation details.
  4. Build & CI & docker
  5. Discuss the results and performance analysis.
  6. Conclude and talk about future work.

---

### **Page 3: Motivation and Objectives (1 minute 30 seconds)**

**Script:**

- First, why did we choose to solve the Poisson-type PDE?
  - The Poisson PDE serves as a basic model for many real-world phenomena, such as heat conduction, electric potential distribution, and fluid dynamics. 

- What are the specific objectives?
  - **First**, we aim to implement a 2D FEM solver in a modular fashion using C++. This includes building sparse linear algebra computation modules.
  - **Second**, we adopt modern software engineering practices, such as using CMake for build management, Docker to ensure a consistent environment, and GitHub Actions for continuous integration (CI).
  - **Third**, we wish to validate the effectiveness of the FEM in solving the Poisson equation and evaluate its performance.


---

### **Pages 4–6 (Estimated 3 minutes total)**

---

### **Page 4: Mathematical Formulation - Poisson-type PDE (1 minute)**

**Script:**

- Let’s move on to the mathematical modeling part.
- The equation we need to solve is the standard **Poisson-type PDE** given by:

this

- This equation describes the distribution of a physical quantity (such as temperature or electric potential) under a given source and boundary conditions.
- Directly solving this equation is challenging; hence, we transform it into its **weak formulation**.

- The **weak formulation** is as follows:

this

---

### **Page 5: Finite Element Discretization (1 minute)**

**Script:**

- After obtaining the weak formulation, we discretize it using the **Finite Element Method (FEM)**.
- The main steps in FEM include:

  1. **Mesh Generation**: Partitioning the domain $\Omega$ into small triangular elements.
  2. **Basis Functions**: Using **P1 Lagrange elements**, which are piecewise linear functions.
  3. **Assembly**: Constructing the global linear system via the following formulas:
 $$
 A_{ij} = \int_{\Omega} \nabla \phi_j \cdot \nabla \phi_i \, dx, \quad F_i = \int_{\Omega} f\, \phi_i \, dx.
 $$

  4. **Applying Boundary Conditions**: Modifying the stiffness matrix $A$ and the load vector $F$ according to the Dirichlet conditions.
  5. **Solving the Linear System**: Solving the system $A\mathbf{U} = \mathbf{F}$.

- Since the stiffness matrix $A$ is a **sparse matrix**, we use the **Eigen** library’s sparse linear algebra tools for efficient solving.

- This process effectively simulates the solution of the Poisson equation over a two-dimensional domain.

---

### **Page 6: Local Element Computations (1 minute)**

**Script:**

- Next, we look at the computations on each **local element (triangle)**.
- In the FEM, the global stiffness matrix and load vector are assembled from the **local matrices** and **local vectors** computed over each triangle.

- The **local stiffness matrix** is given by:

$$
A^K_{ij} = \int_K \nabla \phi_j \cdot \nabla \phi_i \, dx.
$$

- The **local load vector** is:

$$
F^K_i = \int_K f \, \phi_i \, dx.
$$

- **Key Points**:
  - **Simple Gradient Calculation**: Because the basis functions are linear, their gradients are constant.
  - **Local Computations**: Assembly of the global system is simplified by computing matrices and vectors element by element.

- By summing up the contributions from each element, we form the global stiffness matrix and load vector.

---

### **Pages 7–8 (Estimated 2 minutes total)**

---

### **Page 7: C++ Implementation - Project Structure (1 minute)**

**Script:**

- Now, let me introduce the **C++ implementation** of the project.
- To ensure the code is maintainable and extensible, I adopted a **modular design** that divides different functionalities into independent modules.

- **Project Directory Structure:**

|  **File/Folder**         | **Description**                                |
| ------------------------- | ------------------------------------------------ |
| `CMakeLists.txt`        | CMake build configuration file for the project |
| `Dockerfile`            | Docker container configuration for environment consistency |
| `meshes/`               | Contains meshes generated by Gmsh              |
| `results/`              | Stores the results (VTK files)                 |
| `src/`                  | Directory for C++ source code                  |
| ├── `main.cpp`          | Main program controlling overall flow          |
| ├── `Mesh.cpp/.hpp`     | Mesh data structure and geometric computations |
| ├── `PoissonSolver.cpp/.hpp` | FEM Poisson equation solver              |
| ├── `MshReader.cpp/.hpp`| Gmsh mesh file reader                          |
| ├── `VtkExporter.cpp/.hpp` | VTK file export utility                     |
| └── `Quadrature.cpp/.hpp` | Numerical integration module                |

---

### **Page 8: Key Modules (1 minute)**

**Script:**

- Next, I will introduce the **key modules** in the project, which work together from reading the mesh to solving the Poisson equation.

---

**1. Geometry and Data Processing Module**

1. **`Point`, `Triangle`, and `Segment`**:
   - **Point**: Stores the coordinates $(x, y)$ of the 2D nodes.
   - **Triangle**: Represents a triangular element using three node indices.
   - **Segment**: A boundary segment defined by two node indices and a physical tag (for differentiating boundary conditions).

2. **`Mesh`**:
   - Stores all nodes, triangles, and boundary segments in the mesh.
   - Provides functions to **compute the total area** and **boundary length**:
     - `computeTotalArea()`: Sums the area of all triangles.
     - `computeBoundaryLength()`: Sums the length of all boundary segments.

3. **`MshReader`**:
   - Offers a static method `readMsh()` to read mesh data from a Gmsh `.msh` file and populate a `Mesh` instance.
   - Supports reading nodes, triangular elements, and boundary information.

4. **`VtkExporter`**:
   - Provides two static methods:
     - `exportMesh()`: Exports a VTK file containing only geometric information.
     - `exportCompleteVTK()`: Exports a VTK file that includes node scalar fields for visualization.

---

**2. Numerical Computation and Equation Solving Module**

5. **`Quadrature`**:
   - Provides a static method `integrateOnMesh()` to perform **numerical integration** on the mesh for a given function.
   - Supports various orders of quadrature (e.g., **ORDER1**, **ORDER2**, **ORDER3**) for enhanced accuracy.

6. **`PoissonProblem`**:
   - Describes the Poisson problem, including:
     - The **source term** $f(x, y)$ (for example, $f(x, y) = 1$).
     - The **Dirichlet boundary condition** $g(x, y)$ (for example, $u = 0$).

7. **`PoissonSolver`**:
   - The core module responsible for:
     - **Assembling the global stiffness matrix** and the **load vector**.
     - **Applying Dirichlet boundary conditions** (modifying the matrix and vector).
     - Solving the linear system $A\mathbf{U} = \mathbf{F}$ using **Eigen**’s **SparseLU** solver.

---

**Summary**:
- The **Mesh**, **MshReader**, and **VtkExporter** modules manage the mesh data’s reading, storage, and visualization export.
- The **Quadrature** module handles the numerical integration.
- The **PoissonSolver**, in conjunction with **PoissonProblem**, performs the numerical solution of the Poisson equation.
- This **modular design** enhances the code’s **maintainability** and **extensibility** by keeping the functionalities both independent and closely collaborative.
- Next, I will explain how to use **CMake** and **Docker** to build and manage the project.

---

### **Pages 13–15 (Estimated 3 minutes total)**

---

### **Page 13: Meshes and Geometry Check (1 minute)**

**Script:**

- Next, I will show the results of the geometric computations on different meshes.
- The project uses **Gmsh** to generate various 2D meshes, which are used for computation and verification.

---

**Example Meshes:**

| **Filename**                  | **Description**                     | **Number of Triangles** |
|-------------------------------|-------------------------------------|-------------------------|
| `square2d_M0.msh`             | Coarse square mesh                  | 1,474                   |
| `square2d_M1.msh`             | Moderate-density square mesh        | 5,824                   |
| `square2d_M2.msh`             | Fine square mesh                    | 23,252                  |
| `square2d_perforated.msh`     | Square mesh with multiple circular holes | 42,184          |

---

**Geometric Property Calculations:**

- **Area**: Calculated via `Mesh::computeTotalArea()`.
- **Boundary Length**: Calculated via `Mesh::computeBoundaryLength()`.

**Summary:**

- These calculations verify the **correctness** of the mesh reading and geometry calculations.
- Different mesh densities are used to test the **convergence** and **efficiency** of the FEM solver.

---

### **Page 14: Poisson Results (1 minute)**

**Script:**

- Next, we present the numerical solution of the Poisson equation.

- **Test Problem:**

$$
-\Delta u = 1, \quad u = 0 \text{ on the boundary}.
$$

- This implies that the source term on the right-hand side is constant (1) and the boundary condition is a **homogeneous Dirichlet condition** ($u = 0$).

---

**Comparison of Maximum $u$:**

| **Mesh** | **Maximum $u$** | **Minimum $u$** |
|----------|-------------------|-------------------|
| M0       | 0.07358           | 0.0000            |
| M1       | 0.07365           | 0.0000            |
| M2       | 0.07367           | 0.0000            |

- **Observations**:
  - The **maximum value** stabilizes with mesh refinement, indicating good convergence.
  - The **minimum value** remains 0, in accordance with the boundary condition $u = 0$.

---

### **Page 15: Performance (1 minute)**

**Script:**

- Now let’s analyze the program’s **performance** on meshes of various densities.

---

**Assembly Time and Solve Time Comparison:**

| **Mesh**           | **Assembly Time (seconds)** | **Solve Time (seconds)** |
|--------------------|-----------------------------|--------------------------|
| M0                 | 0.0011                      | 0.078                    |
| M1                 | 0.0049                      | 0.469                    |
| M2                 | 0.0254                      | 2.562                    |
| Perforated         | 0.0545                      | 4.142                    |

---

**Observations:**

- **Assembly Time**:
  - Grows linearly with refinement as the time to compute the stiffness matrix and load vector increases.
- **Solve Time**:
  - Increases at a higher rate, indicating that solving the **sparse linear system** is the more computationally demanding part.
  - The SparseLU solver from the **Eigen** library performs well for moderately sized problems.

---

### **Pages 16–17 (Estimated 2 minutes total)**

---

### **Page 16: ParaView - Visualization of FEM Solution (1 minute)**

**Script:**

- Next, I will demonstrate the visualization of the numerical solution for the Poisson equation in **ParaView**.

---

- **Boundary Condition**: $u = 0$ (Dirichlet)
- **Source Term**: $f(x, y) = 1$

---

**Observations:**

- By loading the exported **VTK** file in **ParaView** and applying a **Rainbow Uniform** color map, we can see that the solution $u(x, y)$ achieves its maximum in the center of the domain and then gradually drops to 0 along the boundaries, consistent with the physical characteristics of the Poisson equation.

  - The image shows a color transition from blue (lowest) to red (highest).
  - The **red region** in the center represents the maximum value, while the **blue region** along the boundary corresponds to the minimum value (0).

---

### **Page 17: ParaView - Complex Domain Visualization (1 minute)**

**Script:**

- Next, I will show the solution for a more complex domain—a square mesh with multiple circular holes
- This mesh tests the program’s capability to solve problems with complex geometries.

- **Mesh Characteristics**:
  - Contains 30 evenly distributed circular holes.

---

**Observations:**

- The ParaView visualization reveals significant local variations near the hole boundaries.
- At each hole’s boundary, the condition $u = 0$ is met, and the solution exhibits local “collapse” behavior.
- The overall solution distribution is more complex compared to a simple square domain.

---

### **Pages 18–20 (Estimated 3 minutes total)**

---

### **Page 18: Conclusions (1 minute)**

**Script:**

- Now, let me summarize the entire project.

---

**Key Achievements:**

1. **Successfully implemented a 2D FEM solver for the Poisson equation**:
   - The entire process—from mesh reading and equation assembly to numerical solving—has been implemented and tested in C++.
   - The **Eigen** library is used for sparse matrix storage and solving the linear system, ensuring computational efficiency.

2. **Integration of Modern Software Development Tools**:
   - **CMake**: Automates the build process.
   - **Docker**: Creates a consistent development and runtime environment, simplifying deployment.
   - **GitHub Actions**: Provides Continuous Integration (CI) for automated testing and building, ensuring code quality.

3. **Validation of Numerical Results**:
   - The geometric computations (area and boundary length) are accurate.
   - The numerical solution exhibits good **convergence** and **stability** across various mesh densities.
   - Performance analysis indicates that the program runs efficiently even with large meshes.

4. **Result Visualization**:
   - Using **ParaView** to visualize the results provides an intuitive display of the solution’s spatial distribution.
   - Whether for a simple mesh or a perforated mesh, the solution distribution conforms to physical expectations.

---

**Project Highlights:**

- **Modular Design**: Each functionality is independent, making the code easy to maintain and extend.
- **Automated Workflow**: From development and testing to deployment, the process is fully automated, improving development efficiency.

---

**Summary:**

- This project effectively demonstrates the integration of **numerical computation** and **software engineering practices**.
- It ensures high numerical accuracy while maintaining high code quality and maintainability.

---

### **Page 19: Future Work (1 minute)**

**Script:**

- Although the current solver is robust and feature-rich, there are many avenues for further improvement and expansion.

---

**Future Directions:**

1. **Extension to More Complex PDEs**:
   - For example, tackling nonlinear Poisson equations, elasticity problems, heat conduction problems, etc.
   - Adding other types of boundary conditions (e.g., Neumann boundary conditions).

2. **Adaptive Mesh Refinement (AMR)**:
   - Refining the mesh adaptively based on the error distribution to improve accuracy while reducing computational cost.

3. **Performance Optimization and Parallel Computing**:
   - Utilizing multithreading or GPU acceleration (via **OpenMP**, **CUDA**) to enhance computational speed.
   - Incorporating more efficient solvers (e.g., Conjugate Gradient, Multigrid methods).

4. **Post-Processing and Error Analysis**:
   - Incorporating an **error estimator** to evaluate the accuracy of the numerical solution.
   - Computing additional data such as gradient fields or stress fields for more advanced post-processing.

5. **Enhanced User Interaction and Visualization**:
   - Developing a simple GUI or command-line interface for easier user interaction and parameter setting.
   - Integrating real-time visualization tools to dynamically display the solving process.

---

**Summary:**

- These future enhancements will further increase the solver’s **versatility** and **computational efficiency**, enabling it to address a broader range of engineering and scientific problems.

---

### **Page 20: Q & A (1 minute)**

**Script:**

- This concludes my presentation.

- Thank you very much for your attention!

- **If you have any questions about the project, the numerical methods, or implementation details, please feel free to ask.**

---

**Closing Remarks:**

- Thank you once again!
- If you have more questions, I’d be happy to continue the discussion after the presentation.

---



---

### **Pages 10–12 (Estimated 3 minutes total)**

---

### **Page 10: CMake Build System (1 minute)**

**Script:**

- Now, I will introduce the project’s **build system**.
- To efficiently manage the project's build process, I used **CMake**.

- **CMake** is a cross-platform build tool that can automatically detect dependencies, generate Makefiles, and support multi-platform compilation.

---

**Main Content of CMakeLists.txt:**

1. **Set the C++ Standard and Project Name**:
   ```cmake
   cmake_minimum_required(VERSION 3.10)
   project(MyFemProject CXX)
   set(CMAKE_CXX_STANDARD 17)
   ```

2. **Locate and Link Dependencies**:
   - Use **Eigen3** for sparse matrix computations:
     ```cmake
     find_package(Eigen3 REQUIRED)
     ```

3. **Define the Source Files**:
   ```cmake
   set(SRCS
       src/Mesh.cpp
       src/MshReader.cpp
       src/PoissonSolver.cpp
       src/Quadrature.cpp
       src/VtkExporter.cpp
       src/main.cpp
   )
   ```

4. **Generate the Executable**:
   ```cmake
   add_executable(MyFemExec ${SRCS})
   target_link_libraries(MyFemExec Eigen3::Eigen)
   ```

---

**Build Steps:**

```bash
mkdir build && cd build   # Create a build directory
cmake ..                  # Generate the Makefile
make                      # Compile to create the executable
./MyFemExec               # Run the program
```

- This procedure allows you to compile and run the entire FEM solver successfully.

---

### **Page 11: GitHub Actions - CI Workflow (1 minute)**

**Script:**

- To ensure the code’s **stability** and **quality**, I integrated **GitHub Actions** for **Continuous Integration (CI)**.
- Every time there is a code update or a pull request, the CI pipeline automatically performs the following steps:

---

**CI Workflow:**

1. **Code Checkout**:
   - Use `actions/checkout` to pull the code repository.

2. **Environment Setup**:
   - Automatically install dependencies like **CMake**, **Eigen3**, and **cppcheck**:
     ```bash
     sudo apt-get update
     sudo apt-get install -y cmake build-essential libeigen3-dev cppcheck
     ```

3. **Project Build**:
   - Execute CMake and Make to compile the project:
     ```bash
     mkdir build && cd build
     cmake ..
     make
     ```

4. **Static Code Analysis**:
   - Use `cppcheck` to analyze code quality and detect potential issues:
     ```bash
     cppcheck --enable=warning,style,performance,portability --inconclusive --std=c++17 .
     ```

---

**Benefits:**

- **Automation**: Every code update automatically triggers building and testing.
- **Quality Assurance**: Static analysis helps identify potential bugs and style issues.
- **Continuous Delivery**: Reduces human errors and speeds up the development process.

- This way, the code quality and stability are maintained throughout development.

---

### **Page 12: Docker Integration (1 minute)**

**Script:**

- To ensure consistent **development environments** and facilitate **deployment**, I used **Docker**.

- **Docker** packages the program and its runtime environment into a container, ensuring the same results across different machines.

---

**Main Content of Dockerfile:**

1. **Setting the Base Image**:
   - Use Ubuntu as the base image:
     ```dockerfile
     FROM ubuntu:22.04
     ```

2. **Installing Dependencies**:
   - Install the C++ development environment and required libraries:
     ```dockerfile
     RUN apt-get update && apt-get install -y \
         build-essential \
         cmake \
         git \
         libeigen3-dev \
         cppcheck
     ```

3. **Copying and Building the Project**:
   ```dockerfile
   WORKDIR /app
   COPY . /app
   RUN mkdir build && cd build && cmake .. && make
   ```

4. **Set the Default Command**:
   ```dockerfile
   CMD ["./build/MyFemExec"]
   ```

---

**Building and Running the Docker Container:**

1. **Build the Docker Image**:
   ```bash
   docker build -t fem_project:latest .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run fem_project:latest
   ```

---

**Advantages of Docker:**

- **Environment Consistency**: The same result regardless of the development environment.
- **Easy Deployment**: Containers can be directly deployed to servers or cloud platforms.
- **CI Integration**: GitHub Actions can automatically build and push Docker images.

---

**Summary:**

- Using the combination of **CMake**, **GitHub Actions**, and **Docker**, we achieve an automated workflow from development to deployment.
- This ensures the project’s **portability**, **stability**, and **efficiency**.
- Next, I will show the numerical results and performance analysis.

---

### Time measurement

In the project's PoissonSolver implementation, time measurement is implemented using the C++ standard library's `<chrono>` facilities, specifically employing the high-resolution clock. Below is an explanation of the process and the related code snippets:

1. **Include the Chrono Header**  
    The code begins by including the `<chrono>` header, which provides the time measurement tools:
    
    ```cpp
    #include <chrono>
    ```
    
2. **Start Timing**  
    Right before the section of code you want to measure (for example, the assembly phase), the code records the start time using `std::chrono::high_resolution_clock::now()`:
    
    ```cpp
    auto t_start_assemble = std::chrono::high_resolution_clock::now();
    ```
    
3. **Perform the Task**  
    The target code section is executed. For instance, this might involve assembling the global stiffness matrix and load vector.
    
4. **Stop Timing**  
    Immediately after the task finishes, the code records the end time:
    
    ```cpp
    auto t_end_assemble = std::chrono::high_resolution_clock::now();
    ```
    
5. **Calculate the Duration**  
    The duration of the task is computed by taking the difference between the start and end time points. The `std::chrono::duration<double>` is used to express the time difference in seconds:
    
    ```cpp
    double assembleTime = std::chrono::duration<double>(t_end_assemble - t_start_assemble).count();
    ```
    
    This value, `assembleTime`, now holds the total time (in seconds) taken by the assembly phase.
    
6. **Repeat for Other Sections**  
    A similar approach is used to measure other phases, such as the solving phase (e.g., using the Eigen SparseLU solver). For instance:
    
    ```cpp
    auto t_start_solve = std::chrono::high_resolution_clock::now();
    // Call the solver here, e.g., solver.solve();
    auto t_end_solve = std::chrono::high_resolution_clock::now();
    double solveTime = std::chrono::duration<double>(t_end_solve - t_start_solve).count();
    ```
    
7. **Output the Timing Information**  
    Finally, the measured durations are typically printed out to the console with statements like:
    
    ```cpp
    std::cout << "[Timing] Assembly Time: " << assembleTime << " s" << std::endl;
    std::cout << "[Timing] Solve Time: " << solveTime << " s" << std::endl;
    ```
    

---

#### **Summary**

- **Chrono Library Usage:** The code utilizes `std::chrono::high_resolution_clock` to capture precise time points before and after key operations.
- **Duration Calculation:** `std::chrono::duration<double>` converts the time difference into seconds.
- **Output:** The computed timings are output to the console for performance analysis.

This method provides a precise, cross-platform way to measure the performance of critical sections of the code, aiding in debugging and optimizing the application's efficiency.


## 收敛

After obtaining the solution vector, the function `computeMaxValue` is called to extract the maximum value of \( u \) for that mesh. Here’s a detailed explanation of this step:

1. **Obtaining the Solution Vector**  
   After solving the Poisson equation via `PoissonSolver::solve()`, the function returns a `std::vector<double>` where each entry represents the computed numerical solution \( u \) at a node in the mesh. For example, the vector might look like:
   ```cpp
   std::vector<double> solution = { u_0, u_1, u_2, ..., u_N };
   ```

2. **Purpose of computeMaxValue**  
   The goal of `computeMaxValue` is to traverse the solution vector and determine the maximum value among all \( u \) values. This maximum value is used as a simple convergence indicator. By comparing these maximum values from different meshes, one can observe whether the solution stabilizes (i.e., converges) as the mesh is refined.

3. **Using std::max_element**  
   Inside `computeMaxValue`, the Standard Template Library (STL) function `std::max_element` is used. Here’s how it works step by step:
   
   - **Iterators:**  
     `std::max_element` requires two iterators: one pointing to the beginning of the vector (`solution.begin()`) and one to the end (`solution.end()`).

   - **Traversal:**  
     The function then iterates through the entire vector, comparing elements based on the default "less-than" operator. The algorithm finds the iterator that points to the largest element in the range.
     
   - **Dereferencing:**  
     Once the maximum element is identified, the iterator is dereferenced (using the `*` operator) to obtain the actual value.
     
   - **Example Implementation:**  
     ```cpp
     double computeMaxValue(const std::vector<double>& solution) {
         return *std::max_element(solution.begin(), solution.end());
     }
     ```
     In this code:
     - `solution.begin()` and `solution.end()` specify the range of elements to search.
     - `std::max_element` returns an iterator pointing to the maximum value.
     - The dereference operator (`*`) extracts that maximum value, which is then returned.

4. **Significance and Application**  
   - **Convergence Analysis:**  
     By comparing the maximum \( u \) values across different mesh densities, you can check for convergence. If, as you refine the mesh, the maximum \( u \) value approaches a constant, it indicates that the numerical solution is converging.
     
   - **Quality Check:**  
     The maximum value can also serve as a quick check for unexpected anomalies in the solution. For example, if the maximum value varies significantly between meshes, it might indicate issues in the discretization or solver.

---

**Summary:**

- **Step 1:** The solution vector is computed and returned by `PoissonSolver::solve()`.
- **Step 2:** The `computeMaxValue` function uses `std::max_element` to iterate over the vector.
- **Step 3:** The iterator returned by `std::max_element` is dereferenced to obtain the maximum \( u \) value.
- **Purpose:** This maximum value is then used to assess convergence as different meshes yield increasingly refined approximations of the solution.

This detailed process ensures that convergence is monitored effectively by using a simple yet powerful STL algorithm.