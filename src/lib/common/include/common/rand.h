/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/

#ifndef __KIVATI_RAND_H__
#define __KIVATI_RAND_H__

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL         /* Least significant 31 bits */

void init_genrand64(unsigned long long seed);
void init_by_array64(unsigned long long init_key[],
                     unsigned long long key_length);
unsigned long long genrand64_int64();

// Example code
//
// int main(void)
// {
//     int i;
//     unsigned long long init[4]={0x12345ULL, 0x23456ULL, 0x34567ULL,
//     0x45678ULL}, length=4;
//     init_by_array64(init, length);
//     printf("1000 outputs of genrand64_int64()\n");
//     for (i=0; i<1000; i++) {
//       printf("%20llu ", genrand64_int64());
//       if (i%5==4) printf("\n");
//     }
//     printf("\n1000 outputs of genrand64_real2()\n");
//     for (i=0; i<1000; i++) {
//       printf("%10.8f ", genrand64_real2());
//       if (i%5==4) printf("\n");
//     }
//     return 0;
// }

#endif  // __KIVATI_RAND_H__
