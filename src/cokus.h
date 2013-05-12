#ifndef __COKUS_H__
#define __COKUS_H__

#ifdef __cplusplus
extern "C" {
#endif

//
// uint32 must be an unsigned integer type capable of holding at least 32
// bits; exactly 32 should be fastest, but 64 is better on an Alpha with
// GCC at -O3 optimization so try your options and see what's best for you
//

typedef unsigned long uint32;

void seedMT(uint32 seed);
uint32 reloadMT(void);
uint32 randomMT(void);

#ifdef __cplusplus
}
#endif

#endif // __COKUS_H__