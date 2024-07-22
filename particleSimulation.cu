#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

const int screenWidth = 1920;
const int screenHeight = 1080;
const int numParticles = 10000;
const int numColors = 5;

struct Particle
{
    float3 position;
    float3 velocity;
    int color;
};

__global__ void initParticles(Particle *particles, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);

        particles[idx].position = make_float3(curand_uniform(&state) * 2.0f - 1.0f, curand_uniform(&state) * 2.0f - 1.0f, curand_uniform(&state) * 2.0f - 1.0f);
        particles[idx].velocity = make_float3(0.0f, 0.0f, 0.0f);
        particles[idx].color = idx % numColors;
    }
}

__global__ void initColors(float4 *colors, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numColors)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);

        colors[idx] = make_float4(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state), 1.0f);
    }
}

__device__ float length(float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float3 normalize(float3 v)
{
    float len = length(v);
    if (len > 0)
    {
        v.x /= len;
        v.y /= len;
        v.z /= len;
    }
    return v;
}

__device__ float calculateForce(float distance, float attractionFactor, float beta)
{
    if (distance < beta)
    {
        return (distance / beta) - 1;
    }
    else if (distance < 1.0f)
    {
        return attractionFactor * (1 - (2 * distance - 1 - beta) / (1 - beta));
    }
    else
    {
        return 0.0f;
    }
}

__global__ void updateParticles(Particle *particles, float *attractionMatrix, int numParticles, float deltaTime, int numColors, float beta, float frictionFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
    {
        float3 force = make_float3(0, 0, 0);
        Particle &self = particles[idx];

        for (int j = 0; j < numParticles; j++)
        {
            if (j != idx)
            {
                Particle &other = particles[j];
                float3 diff = make_float3(other.position.x - self.position.x, other.position.y - self.position.y, other.position.z - self.position.z);
                float distance = length(diff);

                if (distance < 1.0f)
                {
                    int color_i = self.color;
                    int color_j = other.color;
                    float attractionFactor = attractionMatrix[color_i * numColors + color_j];

                    float forceMagnitude = calculateForce(distance, attractionFactor, beta);
                    force = make_float3(force.x + forceMagnitude * normalize(diff).x, force.y + forceMagnitude * normalize(diff).y, force.z + forceMagnitude * normalize(diff).z);
                }
            }
        }

        self.velocity = make_float3(self.velocity.x * frictionFactor, self.velocity.y * frictionFactor, self.velocity.z * frictionFactor);

        float3 acceleration = force;
        self.velocity = make_float3(self.velocity.x + acceleration.x * deltaTime, self.velocity.y + acceleration.y * deltaTime, self.velocity.z + acceleration.z * deltaTime);
        self.position = make_float3(self.position.x + self.velocity.x * deltaTime, self.position.y + self.velocity.y * deltaTime, self.position.z + self.velocity.z * deltaTime);

        if (self.position.x > 1.0f)
            self.position.x = -1.0f;
        if (self.position.x < -1.0f)
            self.position.x = 1.0f;
        if (self.position.y > 1.0f)
            self.position.y = -1.0f;
        if (self.position.y < -1.0f)
            self.position.y = 1.0f;
        if (self.position.z > 1.0f)
            self.position.z = -1.0f;
        if (self.position.z < -1.0f)
            self.position.z = 1.0f;
    }
}

const char *vertexShaderSource = "#version 330 core\n"
                                 "layout (location = 0) in vec3 aPos;\n"
                                 "layout (location = 1) in int aColorIndex;\n"
                                 "out vec4 ourColor;\n"
                                 "uniform vec4 colors[3];\n"
                                 "void main()\n"
                                 "{\n"
                                 "   gl_Position = vec4(aPos, 1.0);\n"
                                 "   ourColor = colors[aColorIndex];\n"
                                 "}\0";

const char *fragmentShaderSource = "#version 330 core\n"
                                   "in vec4 ourColor;\n"
                                   "out vec4 FragColor;\n"
                                   "void main()\n"
                                   "{\n"
                                   "   FragColor = ourColor;\n"
                                   "}\n\0";

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "Particle Simulation", NULL, NULL);
    if (window == NULL)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glViewport(0, 0, 800, 600);

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                  << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    Particle *particles;
    int size = sizeof(Particle) * numParticles;
    particles = (Particle *)malloc(size);

    Particle *d_particles;
    cudaMalloc((void **)&d_particles, size);

    float4 *d_colors;
    cudaMalloc((void **)&d_colors, numColors * sizeof(float4));

    unsigned int seed = static_cast<unsigned int>(time(NULL));
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    initParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, seed);
    initColors<<<1, numColors>>>(d_colors, seed);

    float attractionMatrix[numColors * numColors];
    for (int i = 0; i < numColors; i++)
    {
        for (int j = 0; j < numColors; j++)
        {
            attractionMatrix[i * numColors + j] = (i == j) ? 1.0f : -1.0f;
            // attractionMatrix[i * numColors + j] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2.0f)));
        }
    }

    float *d_attractionMatrix;
    cudaMalloc((void **)&d_attractionMatrix, numColors * numColors * sizeof(float));
    cudaMemcpy(d_attractionMatrix, attractionMatrix, numColors * numColors * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size, particles, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribIPointer(1, 1, GL_INT, sizeof(Particle), (void *)(offsetof(Particle, color)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    glUseProgram(shaderProgram);

    glPointSize(5.0f);

    float deltaTime = 0.0004f;
    float beta = 0.1f;
    float frictionFactor = 0.99f;

    int colorLoc = glGetUniformLocation(shaderProgram, "colors");

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        updateParticles<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_attractionMatrix, numParticles, deltaTime, numColors, beta, frictionFactor);

        cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, size, particles);

        float4 colors[numColors];
        cudaMemcpy(colors, d_colors, numColors * sizeof(float4), cudaMemcpyDeviceToHost);

        glUseProgram(shaderProgram);
        glUniform4fv(colorLoc, numColors, (float *)colors);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, numParticles);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    cudaFree(d_particles);
    cudaFree(d_colors);
    cudaFree(d_attractionMatrix);
    free(particles);

    glfwTerminate();
    return 0;
}
