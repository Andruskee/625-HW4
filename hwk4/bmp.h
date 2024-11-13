//
// bmp.h
//
#define MAX_ROW 800
#define MAX_COL 800
#define NUM_COLORS 3
#define BMP_HDR_SIZE 54

typedef unsigned char byte;

int read_bmp(char*, byte*, int*, int*, int*, int*);
int write_bmp(char*, byte*, int*, int, int);
