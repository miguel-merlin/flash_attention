#include <cstddef>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>

#include "file_io.h"

/*
 * Read File For *Reduction*, read contents of file into data
 * @param filename of the source file
 * @param M pointer to data to place read file into
 * @return int success message
 * -2 File name was NULL
 * -1 Buffer was not filled
 * 0 Error opening item file
 * 1 If succesfully read
 */
int readFile(char *file_name, float *data)
{
    // Check file name
    if (file_name == NULL)
    {
        perror("Error file name is null\n");
        return -2;
    }

    // Open file
    FILE *file;
    file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening item file in readFile\n");
        return -1;
    }

    // Find the end of the file
    fseek(file, 0, SEEK_END);
    long file_end = ftell(file);
    rewind(file);

    // Create a buffer and fill buffer with contents of file
    char *buffer = (char *)malloc((file_end + 1) * sizeof(char));

    // Read into buffer
    size_t ret_code = fread(buffer, file_end, sizeof(char), file);

    // Error handle fread
    if (ret_code > 0)
    {
        printf("Array read successfully\n");
    }
    else if (ret_code == 0)
    {
        printf("If ret_code is 0 then the contents of the array and the state of the stream remain unchanged.\n");
    }
    else
    { // error handling
        if (feof(file))
            printf("Error reading file: unexpected end of file\n");
        else if (ferror(file))
        {
            perror("Error reading file\n");
        }
    }

    // Check if buffer has been allocated/read into
    if (buffer == NULL)
    {
        return 0;
    }
    buffer[file_end] = '\0';

    // Close file
    fclose(file);

    // Read buffer into float array
    int element = 0;
    char *token = strtok(buffer, " ");
    while (token != NULL)
    {
        data[element++] = atof(token);
        token = strtok(NULL, " ");
    }

    // Success
    return 1;
}

/*
 * Read File For *Matrix Multiplication*,
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
int readFile(const char *file_name, float **data, unsigned int *len)
{
    // Check file name
    if (file_name == NULL)
    {
        perror("Error file name is null\n");
        return -2;
    }

    // Open file
    FILE *file;
    file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening item file in readFile\n");
        return -1;
    }

    // Find the end of the file
    fseek(file, 0, SEEK_END);
    long file_end = ftell(file);
    rewind(file);

    // Create a buffer and fill buffer with contents of file
    char *buffer = (char *)malloc((file_end + 1) * sizeof(char));

    // Read into buffer
    size_t ret_code = fread(buffer, file_end, sizeof(char), file);

    // Error handle fread
    if (ret_code > 0)
    {
        printf("Array read successfully\n");
    }
    else if (ret_code == 0)
    {
        printf("If ret_code is 0 then the contents of the array and the state of the stream remain unchanged.\n");
    }
    else
    { // error handling
        if (feof(file))
            printf("Error reading file: unexpected end of file\n");
        else if (ferror(file))
        {
            perror("Error reading file\n");
        }
    }

    // Check if buffer has been allocated/read into
    if (buffer == NULL)
    {
        return 0;
    }
    buffer[file_end] = '\0';

    // Close file
    fclose(file);

    // Read data into a vector of floats
    std::vector<float> data_read;
    char *token = strtok(buffer, " ");
    while (token != NULL)
    {
        data_read.push_back(atof(token));
        token = strtok(NULL, " ");
    }

    // Check size
    if (data_read.size() != *len)
    {
        perror("Found more elements in file than expected\n");
        *len = static_cast<unsigned int>(data_read.size());
        return -3;
    }

    // Success, copy vector into data
    *data = (float *)malloc(sizeof(float) * data_read.size());
    std::memcpy(*data, &data_read.front(), sizeof(float) * data_read.size());

    return 1;
}

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
int readFile(const char *file_name, int **data, unsigned int *len)
{
    // Check file name
    if (file_name == NULL)
    {
        perror("Error file name is null\n");
        return -2;
    }

    // Open file
    FILE *file;
    file = fopen(file_name, "r");
    if (file == NULL)
    {
        perror("Error opening item file in readFile\n");
        return -1;
    }

    // Find the end of the file
    fseek(file, 0, SEEK_END);
    long file_end = ftell(file);
    rewind(file);

    // Create a buffer and fill buffer with contents of file
    char *buffer = (char *)malloc((file_end + 1) * sizeof(char));

    // Read into buffer
    size_t ret_code = fread(buffer, file_end, sizeof(char), file);

    // Error handle fread
    if (ret_code > 0)
    {
        printf("Array read successfully\n");
    }
    else if (ret_code == 0)
    {
        printf("If ret_code is 0 then the contents of the array and the state of the stream remain unchanged.\n");
    }
    else
    { // error handling
        if (feof(file))
            printf("Error reading file: unexpected end of file\n");
        else if (ferror(file))
        {
            perror("Error reading file\n");
        }
    }

    // Check if buffer has been allocated/read into
    if (buffer == NULL)
    {
        return 0;
    }
    buffer[file_end] = '\0';

    // Close file
    fclose(file);

    // Read data into a vector of floats
    std::vector<int> data_read;
    char *token = strtok(buffer, " ");
    while (token != NULL)
    {
        data_read.push_back(atoi(token));
        token = strtok(NULL, " ");
    }

    // Check size
    if (data_read.size() != *len)
    {
        perror("Found more elements in file than expected\n");
        *len = static_cast<unsigned int>(data_read.size());
        return -3;
    }

    // Success, copy vector into data
    *data = (int *)malloc(sizeof(int) * data_read.size());
    std::memcpy(*data, &data_read.front(), sizeof(int) * data_read.size());

    return 1;
}

/*
 * Write Matrix File For *Matrix Multiplication*
 * @param filename of the source file
 * @param data pointer to data to write from
 * @param len number of data elements in data
 * @param epsilon ?
 * @return bool succeful
 * exit - 2, if filenae/len == NULL
 * return 1/true if succesfull
 */
int writeFile(const char *filename, const float *data, unsigned int len, const float epsilon)
{
    if (filename == NULL || data == NULL)
        exit(2);

    // open file for writing
    std::fstream fileStreamOut(filename, std::fstream::out);

    // check if filestream is valid
    if (!fileStreamOut.good())
    {
        perror("Writing File: Opening file failed.\n");
        return false;
    }

    // first write epsilon
    fileStreamOut << "# " << epsilon << "\n";

    // write data
    for (unsigned int i = 0; (i < len) && (fileStreamOut.good()); ++i)
    {
        fileStreamOut << data[i] << ' ';
    }

    // Check if writing succeeded
    if (!fileStreamOut.good())
    {
        perror("Writing file failed.\n");
    }

    // file ends with nl
    fileStreamOut << std::endl;

    return 1;
}

/*
 * Compare Data For *Matrix Multiplication*
 * @param reference to matrix data calculated by a gold standard
 * @param data pointer to matrix data to compare
 * @param len number of data elements in data
 * @return bool succeful
 * return 1/true if equal within given epsilon
 * return 0/false if not equal within given episilon
 */
int compareData(const float *reference, const float *data, const unsigned int len, const float epsilon = 0.001f)
{
    // Threshold is percent allowed to be wrong
    const float threshold = 0.0f;

    unsigned int error_count = 0;

    for (unsigned int i = 0; i < len; ++i)
    {
        float diff = reference[i] - data[i];
        bool comp = (diff <= epsilon) && (diff >= -epsilon);
        if (comp == false)
        {
            error_count += 1;
            printf("Element %u: %f != %f (not gold)\n", i, data[i], reference[i]);
        }
    }
    if (error_count)
        printf("%4.2f(%%) of bytes mismatched (count=%d)\n", (float)error_count * 100 / (float)len, error_count);
    // printf("Acceptable errors:%f\n", len*threshold);
    // printf("Error Count:%u\n", error_count);
    return (len * threshold >= error_count) ? true : false;
}