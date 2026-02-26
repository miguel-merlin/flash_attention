#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

typedef struct
{
    unsigned int width;
    unsigned int height;
    float *elements;
} Matrix;

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadMatFile(Matrix *M, char *file_name);
void WriteMatFile(Matrix M, char *file_name);
void FreeDeviceMatrix(Matrix *M);
void FreeMatrix(Matrix *M);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);

#endif // _MATRIXMUL_H_
