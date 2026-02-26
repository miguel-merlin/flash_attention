#ifndef _FILE_IO_H_
#define _FILE_IO_H_

/*
 * Read Matrix File For *Reduction*
 * @param filename of the source file
 * @param M pointer to data to place read file into
 * @return int success message
 * -2 File name was NULL
 * -1 Buffer was not filled
 * 0 Error opening item file
 * 1 If succesfully read
 */
int readFile(char *file_name, float *M);

/*
 * Read Matrix File For *Matrix Multiplication*
 * @param filename of the source file
 * @param data pointer to data to write to
 * @param len number of data elements in data
 * @return bool succeful
 * -3 If len of supposed elements is not found in file
 * -2 File name was NULL
 * -1 Buffer was not filled
 * 0 Error opening item file
 * 1 If succesfully read
 */
int readFile(const char *file_name, float **data, unsigned int *len);

/*
 * Read File For *Matrix Multiplication*,
 * read integer data
 * check if len is equal before copping over memory
 * @param filename of the source file
 * @param data pointer to data to write to
 * @param len number of data elements in data
 * @return bool succeful
 * -3 If len of supposed elements is not found in file
 * -2 File name was NULL
 * -1 Buffer was not filled
 * 0 Error opening item file
 * 1 If succesfully read
 */
int readFile(const char *file_name, int **data, unsigned int *len);

/*
 * Write Matrix File For *Matrix Multiplication*
 * @param filename of the source file
 * @param data pointer to data to write from
 * @param len number of data elements in data
 * @param epsilon to account for floating point precision
 * @return bool succeful
 * exit - 2, if filenae/len == NULL
 * return 1/true if succesfull
 */
int writeFile(const char *filename, const float *data, unsigned int len, const float epsilon);

/*
 * Compare Data For *Matrix Multiplication*
 * @param reference to matrix data calculated by a gold standard
 * @param data pointer to matrix data to compare
 * @param len number of data elements in data
 * @return bool succeful
 * return 1/true if equal within given epsilon
 * return 0/false if not equal within given episilon
 */
int compareData(const float *reference, const float *data, const unsigned int len, const float epsilon);

#endif