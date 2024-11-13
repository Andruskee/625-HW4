//
// bmp.cu
//
#include <stdio.h>
#include <stdlib.h>
#include "bmp.h"

int read_bmp(char *filename, byte *bmp_hdr, int *a, int *s, int *w, int *h)
{
  unsigned char z;
  unsigned char *sp = (unsigned char *)s;
  unsigned char *wp = (unsigned char *)w;
  unsigned char *hp = (unsigned char *)h;
  int odd_width = 0;
  int buffer_byte = 0;
  long i = 0;
  long x = 0;
  long y = 0;
  long c = 0;
  FILE *fp_in;

  fp_in = fopen(filename,"rb+");
  while (!feof(fp_in))
  {
    if (fread(&z,1,1,fp_in) > 0)
    {
      i++;
    }
    switch(i-1)
    {
      case 0: if (z!='B')
                printf("Not a bitmap file. Magic B missing.\n");
              break;
      case 1: if (z!='M')
                printf("Not a bitmap file. Magic M missing.\n");
              break;
      case 2:
      case 3:
      case 4:
      case 5: *sp++ = z;
	      break;
      case 18: odd_width = z%2;
      case 19:
      case 20:
      case 21: *wp++ = z;
               break;
      case 22:
      case 23:
      case 24:
      case 25: *hp++ = z;
               break;
    }
    if (i<=0x36)
    {
      bmp_hdr[i-1]=z;
    }
    else
    {
      if (!buffer_byte)
      {
        a[x*(*w)*3+y*3+c]=z;
        c++;
        if (c>2)
        {
          c=0;
          y++;
          if (y>=(*w))
          {
            y=0;
            x++;
            if (odd_width)
              buffer_byte = 1;
          }
        }
      }
      else
      {
        buffer_byte=0;
      }
    }
  }
  printf("\n");
  printf("Size: %d\n", (*s));
  printf("Width: %d\n", (*w));
  printf("Height: %d\n", (*h));
  fseek(fp_in,0x16,SEEK_SET);
  fclose(fp_in);
  return 0;
}

int write_bmp(char *filename, byte *bmp_hdr, int *b, int width, int height)
{
  int x, y, c;
  int *ptr;
  int odd_width = width%2;
  unsigned char zero = 0;
  FILE *fp_out;

  fp_out = fopen(filename,"wb+");

  ptr = (int *) &bmp_hdr[18];
  *ptr = width;
  ptr = (int *) &bmp_hdr[22];
  *ptr = height;

  fwrite(bmp_hdr,1,BMP_HDR_SIZE,fp_out);

  for (x=0; x<height; x++)
  {
    for (y=0; y<width; y++)
      for (c=0; c<3; c++)
        fwrite(&b[x*width*3+y*3+c],1,1,fp_out);
    if (odd_width)
      fwrite(&zero,1,1,fp_out);
  }
  fclose(fp_out);
  return 0;
}
