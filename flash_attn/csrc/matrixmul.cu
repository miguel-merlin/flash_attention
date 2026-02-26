#include <string>
#include "matrixmul.h"
#include "file_io.h"
#define TILE_WIDTH 16

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

int main(int argc, char **argv)
{

    Matrix M;
    Matrix N;
    Matrix P;
    int errorM = 0, errorN = 0;

    srand(5672);

    if (argc != 5 && argc != 4)
    {
        M = AllocateMatrix(rand() % 1024, rand() % 1024, 1);
        N = AllocateMatrix(M.width, rand() % 1024, 1);
        P = AllocateMatrix(M.height, N.width, 0);
    }
    else
    {
        int *params = NULL;
        unsigned int data_read = 3;
        readFile(argv[1], &params, &data_read);
        if (data_read != 3)
        {
            printf("Error reading parameter file\n");
            return 1;
        }

        M = AllocateMatrix(params[0], params[1], 0);
        N = AllocateMatrix(params[1], params[2], 0);
        P = AllocateMatrix(params[0], params[2], 0);
        errorM = ReadMatFile(&M, argv[2]);
        errorN = ReadMatFile(&N, argv[3]);
        if (errorM || errorN)
        {
            printf("Error reading input files %d, %d\n", errorM, errorN);
            return 1;
        }
    }

    MatrixMulOnDevice(M, N, P);

    printf("GPU computation complete\n");
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);

    printf("CPU computation complete\n");
    int res = compareData(reference.elements, P.elements, P.height * P.width, 0.001f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    if (argc == 5)
    {
        WriteMatFile(P, argv[4]);
    }
    else if (argc == 2)
    {
        WriteMatFile(P, argv[1]);
    }

    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
    return 0;
}

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N.width + TILE_WIDTH - 1) / TILE_WIDTH, (M.height + TILE_WIDTH - 1) / TILE_WIDTH);
    MatrixMulKernel<<<gridSize, blockDim>>>(Md, Nd, Pd);

    CopyFromDeviceMatrix(P, Pd);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    // don't allocate memory on option 2
    if (init == 2)
        return M;

    M.elements = (float *)malloc(size * sizeof(float));

    for (unsigned int i = 0; i < M.height * M.width; i++)
    {
        M.elements[i] = (init == 0) ? (0.0f) : (rand() * 3 / (float)RAND_MAX);
    }
    return M;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
               cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,
               cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix *M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix *M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is
//  equals M.height * M.width, and 1 otherwise
int ReadMatFile(Matrix *M, char *file_name)
{
    unsigned int data_read = M->height * M->width;
    readFile(file_name, &(M->elements), &data_read);
    return (data_read != (M->height * M->width));
}

// Write a floating point matrix to file
void WriteMatFile(Matrix M, char *file_name)
{
    writeFile(file_name, M.elements, M.width * M.height, 0.0001f);
}
