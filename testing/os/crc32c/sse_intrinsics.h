
/*  Copyright (C) 2011, 2012 Nicholas Cardell, Jake Stine
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do
 *  so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

// --------------------------------------------------------------------------------------
// WARNING: Visual Studio and PMOVSX/PMOVZX  (SSE4.1+)   [SSE_INTRIN_PMOV_HACK]
//
// All known versions of Visual Studio make alignment assumptions when PMOVSX/ZX are used
// to fetch contents from memory, which will generally break most intended uses of these
// intrinsics.  (result is an invalid access crash caused by a preceding movqda).
// The bug occurs most frequently when optimizations are disabled, however can also 
// happen when optimizations are enabled in specific (rare) circumstances.
//
// Various forms of pointer typecast appear to convince the optimizer to use indirect-memory
// forms of the PMOVSX/ZX instructions; however debug builds ALWAYS generate code that uses
// movdqa.  The only reliable fix for this at this time is to manually use movdqu to load
// data into an xmm prior to using it in PMOVSX/ZX.  Ex:
//
//   __m128 temp, result;
//   i_movdqa( temp, ptr );
//   i_movsxbd( result, temp );
//
// The trade-off is that this will generate an extra mov instruction even when the optimizer
// is enabled.  (one would think the optimizer could strip out the excess mov, but alas
// I tested and it does not. --jstine)
// --------------------------------------------------------------------------------------


// --------------------------------------------------------------------------------------
// SIMD_STRICT_MEMORY_TYPES  (define/config)
//
// When defined, loads and stores (movaps, etc) will require explicit typecasting to
// or from their base type, suchs as (__m128*) and (__m128i*).  When disabled all memory
// references are treated as void instead.
// --------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------
// SSE_INTRIN_NO_OPTIMIZE  (define/config)
//
// Option to disable the header file's automatic enabling of optimization and inlining.
// Highly recommended this is left at its default (0).  Tracing intrinsic code via the
// debugger should be a lot more gratifying in all cases barring perhaps troubleshooting
// bugs in the intrinsic wrapper itself.
// --------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------
// SSE_INTRIN_PMOV_HACK=[0,1]  (define/config)
//
// Controls whether or not the PMOVSX/ZX instructions are generated with preceding movdqu
// (unaligned) loads.  This hack must be enabled for Visual Studio non-optimized compiles,
// otherwise resulting code may crash with an unaligned read exception.
// --------------------------------------------------------------------------------------

#pragma once

#include <xmmintrin.h>
#include <emmintrin.h>

#if (_MSC_VER >= 1600)
#	include <immintrin.h>	// VS2010 supports AVX
#endif	// _MSC_VER >= 1600

#if defined(__GNUC__)

	// gcc provides a single header that includes all intrinsics based on the compiler's
	// target machine configuration.  If you want AVX support then you need to tell GCC to
	// use i7-avx as its target.

#	include <x86intrin.h>
#	if defined(__PCLMUL__)	// only include wmmintrin if pclmul support is enabled by compiler.
#		include <wmmintrin.h>
#	endif
#endif

#include <stdint.h>

#if !defined(SSE_INTRIN_PMOV_HACK)
#	if defined(_MSC_VER) && defined(_DEBUG)
#		define SSE_INTRIN_PMOV_HACK		1
#	else
#		define SSE_INTRIN_PMOV_HACK		0
#	endif
#endif


#if defined(__GNUC__)
#	if !defined(__ssi_always_inline__)
#		define __ssi_always_inline__			__attribute__((always_inline, unused))
#	endif
#	if !defined( __ssi_static_inline__ )
#		define __ssi_static_inline__			static __attribute__((always_inline, unused))
#	endif
#elif defined(_MSC_VER)
#	if !defined( __ssi_always_inline__ )
#		define __ssi_always_inline__			__forceinline
#	endif
#	if !defined( __ssi_static_inline__ )
#		define __ssi_static_inline__			static __forceinline
#	endif
#else
#	if !defined( __ssi_always_inline__ )
#		define __ssi_always_inline__			inline
#	endif
#	if !defined( __ssi_static_inline__ )
#		define __ssi_static_inline__			static inline
#	endif
#endif


// _M_AMD64 for msvc, __x86_64__ for gcc.
#if !defined(_M_AMD64) && !defined(__x86_64__)

#	include <assert.h>
#	define _mm_cvtsi64_ss(a, b) (assert(0), a) // only available in x64
	uint64_t _mm_crc32_u64 (uint64_t crc, uint64_t v)	{ assert(0); return 0; } // only available in x64
	int64_t  _mm_popcnt_u64(uint64_t v)					{ assert(0); return 0; } // only available in x64

#endif

#define FtoD(x) _mm_castps_pd(x)
#define DtoF(x) _mm_castpd_ps(x)
#define FtoI(x) _mm_castps_si128(x)
#define ItoF(x) _mm_castsi128_ps(x)
#define FtoF(x) x

#define StInl	__ssi_static_inline__
#define tmplInl	__ssi_always_inline__

// Implementation note for strict memory typedefs: intentionally opting for exclusion of the
// ptr(*) in the typedefs.  I find code more clear when the * is present in the function
// prototypes and associated casts. --jstine
#if defined(SIMD_STRICT_MEMORY_TYPES)
	typedef const	__m128i		memsrc_m128i;
	typedef	const	__m128		memsrc_m128;
	typedef	const	__m128d		memsrc_m128d;
	typedef	const	float		memsrc_float;
	typedef	const	double		memsrc_double;
	typedef	const	uint16_t	memsrc_u16;
	typedef	const	uint32_t	memsrc_u32;
	typedef	const	uint64_t	memsrc_u64;
	typedef	const	char		memsrc_char;

	typedef			__m128i		memdst_m128i;
	typedef			__m128		memdst_m128;
	typedef			__m128d		memdst_m128d;
	typedef			float		memdst_float;
	typedef			double		memdst_double;
	typedef			uint16_t	memdst_u16;
	typedef			uint32_t	memdst_u32;
	typedef			uint64_t	memdst_u64;
	typedef			char		memdst_char;

#else

	typedef const	void		memsrc_m128i;
	typedef	const	void		memsrc_m128;
	typedef	const	void		memsrc_m128d;
	typedef	const	void		memsrc_float;
	typedef	const	void		memsrc_double;
	typedef	const	void		memsrc_u16;
	typedef	const	void		memsrc_u32;
	typedef	const	void		memsrc_u64;
	typedef	const	void		memsrc_char;

	typedef			void		memdst_m128i;
	typedef			void		memdst_m128;
	typedef			void		memdst_m128d;
	typedef			void		memdst_float;
	typedef			void		memdst_double;
	typedef			void		memdst_u16;
	typedef			void		memdst_u32;
	typedef			void		memdst_u64;
	typedef			char		memdst_char;
#endif


// --------------------------------------------------------------------------------------
// SSE_INTRIN_OPTIMIZE_BEGIN / SSE_INTRIN_OPTIMIZE_END
//
// Force optimization AND inlining on Visual Studio in ALL BUILDS (including unoptimized
// debug builds) by default.  This helps improve both performance and step-tracing of
// SIMD intrinsics.  It is also highly recommended to enable intrinsic inlines in your
// debug build target project options.
//
#if (_MSC_VER > 1600) && !defined(SSE_INTRIN_NO_OPTIMIZE)
#	define SSE_INTRIN_OPTIMIZE_BEGIN	__pragma( optimize( "gtb", on ) )
#	define SSE_INTRIN_OPTIMIZE_END		__pragma( optimize( "", on ) )
#else
#	define SSE_INTRIN_OPTIMIZE_BEGIN
#	define SSE_INTRIN_OPTIMIZE_END
#endif
// --------------------------------------------------------------------------------------


SSE_INTRIN_OPTIMIZE_BEGIN

// ADD / SUB
StInl void i_addps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_add_ps(FtoF(a), FtoF(b))); }
StInl void i_addss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_add_ss(FtoF(a), FtoF(b))); }
StInl void i_addpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_add_pd(FtoD(a), FtoD(b))); }
StInl void i_addsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_add_sd(FtoD(a), FtoD(b))); }
StInl void i_subps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_sub_ps(FtoF(a), FtoF(b))); }
StInl void i_subss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_sub_ss(FtoF(a), FtoF(b))); }
StInl void i_subpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_sub_pd(FtoD(a), FtoD(b))); }
StInl void i_subsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_sub_sd(FtoD(a), FtoD(b))); }

// ADDSUB
StInl void i_addsubps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_addsub_ps(FtoF(a), FtoF(b))); }
StInl void i_addsubpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_addsub_pd(FtoD(a), FtoD(b))); }

// AND / ANDN / OR / XOR
StInl void i_andps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_and_ps(FtoF(a), FtoF(b))); }
StInl void i_andpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_and_pd(FtoD(a), FtoD(b))); }
StInl void i_andnps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_andnot_ps(FtoF(a), FtoF(b))); }
StInl void i_andnpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_andnot_pd(FtoD(a), FtoD(b))); }
StInl void i_orps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_or_ps(FtoF(a), FtoF(b))); }
StInl void i_orpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_or_pd(FtoD(a), FtoD(b))); }
StInl void i_xorps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_xor_ps(FtoF(a), FtoF(b))); }
StInl void i_xorpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_xor_pd(FtoD(a), FtoD(b))); }

// single-param versions of xor, which provide warning-free method of zeroing an uninitialized register.
StInl void i_xorps		(__m128& dest)						{ dest = FtoF(_mm_xor_ps(FtoF(dest), FtoF(dest))); }
StInl void i_xorpd		(__m128& dest)						{ dest = DtoF(_mm_xor_pd(FtoD(dest), FtoD(dest))); }

// BLEND
template<int mask> tmplInl void i_blendps_(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_blend_ps(FtoF(a), FtoF(b), mask & 3)); }
template<int mask> tmplInl void i_blendpd_(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_blend_pd(FtoD(a), FtoD(b), mask & 3)); }
#define i_blendps(dest, a, b, mask) i_blendps_<mask>(dest, a, b) // mask needs to be a constant expression / integer literal
#define i_blendpd(dest, a, b, mask) i_blendpd_<mask>(dest, a, b) // mask needs to be a constant expression / integer literal

// BLENDV
StInl void i_blendvps	(__m128& dest, __m128 a, __m128 b, __m128 c)		{ dest = FtoF(_mm_blendv_ps(FtoF(a), FtoF(b), FtoF(c))); }
StInl void i_blendvpd	(__m128& dest, __m128 a, __m128 b, __m128 c)		{ dest = DtoF(_mm_blendv_pd(FtoD(a), FtoD(b), FtoD(c))); }

// CMPPD
StInl void i_cmpeqpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpeq_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpltpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmplt_pd(FtoD(a), FtoD(b))); }
StInl void i_cmplepd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmple_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpgtpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpgt_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpgepd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpge_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpneqpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpneq_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpnltpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpnlt_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpnlepd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpnle_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpngtpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpngt_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpngepd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpnge_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpordpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpord_pd(FtoD(a), FtoD(b))); }
StInl void i_cmpunordpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpunord_pd(FtoD(a), FtoD(b))); }

// CMPSD
StInl void i_cmpeqsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpeq_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpltsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmplt_sd(FtoD(a), FtoD(b))); }
StInl void i_cmplesd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmple_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpgtsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpgt_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpgesd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpge_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpneqsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpneq_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpnltsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpnlt_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpnlesd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpnle_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpngtsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpngt_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpngesd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpnge_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpordsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpord_sd(FtoD(a), FtoD(b))); }
StInl void i_cmpunordsd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cmpunord_sd(FtoD(a), FtoD(b))); }

// CMPPS
StInl void i_cmpeqps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpeq_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpltps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmplt_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpleps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmple_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpgtps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpgt_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpgeps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpge_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpneqps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpneq_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpnltps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpnlt_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpnleps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpnle_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpngtps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpngt_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpngeps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpnge_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpordps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpord_ps(FtoF(a), FtoF(b))); }
StInl void i_cmpunordps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpunord_ps(FtoF(a), FtoF(b))); }

// CMPSS
StInl void i_cmpeqss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpeq_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpltss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmplt_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpless	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmple_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpgtss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpgt_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpgess	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpge_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpneqss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpneq_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpnltss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpnlt_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpnless	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpnle_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpngtss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpngt_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpngess	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpnge_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpordss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpord_ss(FtoF(a), FtoF(b))); }
StInl void i_cmpunordss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cmpunord_ss(FtoF(a), FtoF(b))); }

// COMISD
StInl int i_comieqsd	(__m128 a, __m128 b)				{ return (_mm_comieq_sd(FtoD(a), FtoD(b))); }
StInl int i_comiltsd	(__m128 a, __m128 b)				{ return (_mm_comilt_sd(FtoD(a), FtoD(b))); }
StInl int i_comilesd	(__m128 a, __m128 b)				{ return (_mm_comile_sd(FtoD(a), FtoD(b))); }
StInl int i_comigtsd	(__m128 a, __m128 b)				{ return (_mm_comigt_sd(FtoD(a), FtoD(b))); }
StInl int i_comigesd	(__m128 a, __m128 b)				{ return (_mm_comige_sd(FtoD(a), FtoD(b))); }
StInl int i_comineqsd	(__m128 a, __m128 b)				{ return (_mm_comineq_sd(FtoD(a), FtoD(b))); }

// COMISS
StInl int i_comieqss	(__m128 a, __m128 b)				{ return (_mm_comieq_ss(FtoF(a), FtoF(b))); }
StInl int i_comiltss	(__m128 a, __m128 b)				{ return (_mm_comilt_ss(FtoF(a), FtoF(b))); }
StInl int i_comiless	(__m128 a, __m128 b)				{ return (_mm_comile_ss(FtoF(a), FtoF(b))); }
StInl int i_comigtss	(__m128 a, __m128 b)				{ return (_mm_comigt_ss(FtoF(a), FtoF(b))); }
StInl int i_comigess	(__m128 a, __m128 b)				{ return (_mm_comige_ss(FtoF(a), FtoF(b))); }
StInl int i_comineqss	(__m128 a, __m128 b)				{ return (_mm_comineq_ss(FtoF(a), FtoF(b))); }

// UCOMISD
StInl int i_ucomieqsd	(__m128 a, __m128 b)				{ return (_mm_ucomieq_sd(FtoD(a), FtoD(b))); }
StInl int i_ucomiltsd	(__m128 a, __m128 b)				{ return (_mm_ucomilt_sd(FtoD(a), FtoD(b))); }
StInl int i_ucomilesd	(__m128 a, __m128 b)				{ return (_mm_ucomile_sd(FtoD(a), FtoD(b))); }
StInl int i_ucomigtsd	(__m128 a, __m128 b)				{ return (_mm_ucomigt_sd(FtoD(a), FtoD(b))); }
StInl int i_ucomigesd	(__m128 a, __m128 b)				{ return (_mm_ucomige_sd(FtoD(a), FtoD(b))); }
StInl int i_ucomineqsd	(__m128 a, __m128 b)				{ return (_mm_ucomineq_sd(FtoD(a), FtoD(b))); }

// UCOMISS
StInl int i_ucomieqss	(__m128 a, __m128 b)				{ return (_mm_ucomieq_ss(FtoF(a), FtoF(b))); }
StInl int i_ucomiltss	(__m128 a, __m128 b)				{ return (_mm_ucomilt_ss(FtoF(a), FtoF(b))); }
StInl int i_ucomiless	(__m128 a, __m128 b)				{ return (_mm_ucomile_ss(FtoF(a), FtoF(b))); }
StInl int i_ucomigtss	(__m128 a, __m128 b)				{ return (_mm_ucomigt_ss(FtoF(a), FtoF(b))); }
StInl int i_ucomigess	(__m128 a, __m128 b)				{ return (_mm_ucomige_ss(FtoF(a), FtoF(b))); }
StInl int i_ucomineqss	(__m128 a, __m128 b)				{ return (_mm_ucomineq_ss(FtoF(a), FtoF(b))); }

// CVTDQ2PD
StInl void i_cvtdq2pd	(__m128& dest, __m128 a)			{ dest = DtoF(_mm_cvtepi32_pd(FtoI(a))); }
// CVTDQ2PS
StInl void i_cvtdq2ps	(__m128& dest, __m128 a)			{ dest = FtoF(_mm_cvtepi32_ps(FtoI(a))); }
// CVTPD2DQ
StInl void i_cvtpd2dq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtpd_epi32(FtoD(a))); }
// CVTPD2PI
StInl void i_cvtpd2pi	(__m64&  dest, __m128 a)			{ dest =     (_mm_cvtpd_pi32(FtoD(a))); }
// CVTPD2PS
StInl void i_cvtpd2ps	(__m128& dest, __m128 a)			{ dest = FtoF(_mm_cvtpd_ps(FtoD(a))); }
// CVTPI2PD
StInl void i_cvtpi2pd	(__m128& dest, __m64  a)			{ dest = DtoF(_mm_cvtpi32_pd(a)); }
// CVTPI2PS
StInl void i_cvtpi2ps	(__m128& dest, __m128 a, __m64  b)	{ dest = FtoF(_mm_cvt_pi2ps(FtoF(a), b)); }
// CVTPS2DQ
StInl void i_cvtps2dq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtps_epi32(FtoF(a))); }
// CVTPS2PD
StInl void i_cvtps2pd	(__m128& dest, __m128 a)			{ dest = DtoF(_mm_cvtps_pd(FtoF(a))); }
// CVTPS2PI
StInl void i_cvtps2pi	(__m64&  dest, __m128 a)			{ dest =     (_mm_cvt_ps2pi(FtoF(a))); }
// CVTSD2SI
StInl int  i_cvtsd2si	(__m128 a)							{ return     (_mm_cvtsd_si32(FtoD(a))); }
// CVTSD2SS
StInl void i_cvtsd2ss	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_cvtsd_ss(FtoF(a), FtoD(b))); }
// CVTSI2SD
StInl void i_cvtsi2sd	(__m128& dest, __m128 a, int b)		{ dest = DtoF(_mm_cvtsi32_sd(FtoD(a),  b)); }
// CVTSI2SS
StInl void i_cvtsi2ss	(__m128& dest, __m128 a, int b)		{ dest = FtoF(_mm_cvt_si2ss (FtoF(a),  b)); }
StInl void i_cvtsi2ss	(__m128& dest, __m128 a, int64_t b)	{ dest = FtoF(_mm_cvtsi64_ss(FtoF(a),  b)); }
// CVTSS2SD
StInl void i_cvtss2sd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_cvtss_sd(FtoD(a), FtoF(b))); }
// CVTSS2SI
StInl int  i_cvtss2si	(__m128 a)							{ return     (_mm_cvt_ss2si(FtoF(a))); }
// CVTTPD2DQ
StInl void i_cvttpd2dq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvttpd_epi32(FtoD(a))); }
// CVTTPD2PI
StInl void i_cvttpd2pi	(__m64&  dest, __m128 a)			{ dest =     (_mm_cvttpd_pi32(FtoD(a))); }
// CVTTPS2DQ
StInl void i_cvttps2dq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvttps_epi32(FtoF(a))); }
// CVTTPS2PI
StInl void i_cvttps2pi	(__m64&  dest, __m128 a)			{ dest =     (_mm_cvtt_ps2pi(FtoF(a))); }
// CVTTSD2SI
StInl int  i_cvttsd2si	(__m128 a)							{ return     (_mm_cvttsd_si32(FtoD(a))); }
// CVTTSS2SI
StInl int  i_cvttss2si	(__m128 a)							{ return     (_mm_cvtt_ss2si(FtoF(a))); }

// DIV
StInl void i_divps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_div_ps(FtoF(a), FtoF(b))); }
StInl void i_divss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_div_ss(FtoF(a), FtoF(b))); }
StInl void i_divpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_div_pd(FtoD(a), FtoD(b))); }
StInl void i_divsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_div_sd(FtoD(a), FtoD(b))); }

// DP
template<int mask> tmplInl void i_dpps_(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_dp_ps(FtoF(a), FtoF(b), mask)); }
template<int mask> tmplInl void i_dppd_(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_dp_pd(FtoD(a), FtoD(b), mask)); }
#define i_dpps(dest, a, b, mask) i_dpps_<mask>(dest, a, b) // mask needs to be a constant expression / integer literal
#define i_dppd(dest, a, b, mask) i_dppd_<mask>(dest, a, b) // mask needs to be a constant expression / integer literal

// EXTRACTPS
template<int ndx> tmplInl int  i_extractps_	(__m128 a)		{ return     (_mm_extract_ps(FtoF(a), ndx)); }
#define i_extractps(a, ndx) i_extractps_<ndx>(a) // ndx needs to be a constant expression / integer literal

// INSERTPS
template<int ndx> tmplInl void i_insertps_(__m128& dest, __m128 dst, __m128 src)		{ dest = FtoF(_mm_insert_ps(FtoF(dst), FtoF(src), ndx)); }
template<int ndx> tmplInl void i_insertps_(__m128& dest, __m128 dst, memsrc_m128* src)	{ dest = FtoF(_mm_insert_ps(FtoF(dst), *(__m128*)src, ndx)); }
#define i_insertps(dest, dst, src, ndx) i_insertps_<ndx>(dest, dst, src) // ndx needs to be a constant expression / integer literal

// HADD / HSUB
StInl void i_haddps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_hadd_ps(FtoF(a), FtoF(b))); }
StInl void i_haddpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_hadd_pd(FtoD(a), FtoD(b))); }
StInl void i_hsubps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_hsub_ps(FtoF(a), FtoF(b))); }
StInl void i_hsubpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_hsub_pd(FtoD(a), FtoD(b))); }

// LDDQU 
StInl void i_lddqu		(__m128& dest, memsrc_m128i* src)	{ dest = ItoF(_mm_lddqu_si128((__m128i*)src)); }

// MASKMOVDQU
// Uses strict const char* type for source memory since it is referencing non-SIMD data.
StInl void i_maskmovdqu	(__m128 src, __m128 mask, const char* p)	{ _mm_maskmoveu_si128(FtoI(src), FtoI(mask), (char*)p); }

// MAX / MIN
StInl void i_maxps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_max_ps(FtoF(a), FtoF(b))); }
StInl void i_maxss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_max_ss(FtoF(a), FtoF(b))); }
StInl void i_maxpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_max_pd(FtoD(a), FtoD(b))); }
StInl void i_maxsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_max_sd(FtoD(a), FtoD(b))); }
StInl void i_minps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_min_ps(FtoF(a), FtoF(b))); }
StInl void i_minss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_min_ss(FtoF(a), FtoF(b))); }
StInl void i_minpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_min_pd(FtoD(a), FtoD(b))); }
StInl void i_minsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_min_sd(FtoD(a), FtoD(b))); }

// MOVAPS / MOVAPD / MOVDQA / MOVSS / MOVSD / MOVQ
StInl void i_movaps		(memdst_float*  dest, __m128 src)	{ _mm_store_ps		((float*)  dest, FtoF(src)); }
StInl void i_movapd		(memdst_double* dest, __m128 src)	{ _mm_store_pd		((double*) dest, FtoD(src)); }
StInl void i_movss		(memdst_float*  dest, __m128 src)	{ _mm_store_ss		((float*)  dest, FtoF(src)); }
StInl void i_movsd		(memdst_double* dest, __m128 src)	{ _mm_store_sd		((double*) dest, FtoD(src)); }
StInl void i_movq		(memdst_m128i*	dest, __m128 src)	{ _mm_storel_epi64	((__m128i*)dest, FtoI(src)); }
StInl void i_movdqa		(memdst_m128i*	dest, __m128 src)	{ _mm_store_si128	((__m128i*)dest, FtoI(src)); }
StInl void i_movaps		(__m128& dest, memsrc_float*   src)	{ dest = FtoF(_mm_load_ps		((float*)  src)); }
StInl void i_movapd		(__m128& dest, memsrc_double*  src)	{ dest = DtoF(_mm_load_pd		((double*) src)); }
StInl void i_movss_zx	(__m128& dest, memsrc_float*   src)	{ dest = FtoF(_mm_load_ss		((float*)  src)); }
StInl void i_movsd_zx	(__m128& dest, memsrc_double*  src)	{ dest = DtoF(_mm_load_sd		((double*) src)); }
StInl void i_movq_zx	(__m128& dest, memsrc_m128i* src)	{ dest = ItoF(_mm_loadl_epi64	((__m128i*)src)); }
StInl void i_movdqa		(__m128& dest, memsrc_m128i* src)	{ dest = ItoF(_mm_load_si128	((__m128i*)src)); }
StInl void i_movss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_move_ss(FtoF(a), FtoF(b))); }
StInl void i_movsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_move_sd(FtoD(a), FtoD(b))); }
StInl void i_movq_zx	(__m128& dest, __m128 src)			{ dest = ItoF(_mm_move_epi64(FtoI(src))); }
//StInl void i_movaps	(__m128& dest, __m128 src)			{ dest = ItoF(_mm_shuffle_epi32(FtoI(src), 0xe4)); }
// Note: i_movaps(xmm,xmm) above is emulated because theres no real intrinsic to move xmm-to-xmm

// MOVD
StInl void i_movd_zx	(__m128& dest, int src)				{ dest = ItoF(_mm_cvtsi32_si128(src)); }
StInl int  i_movd		(__m128 src)						{ return _mm_cvtsi128_si32(FtoI(src)); }

// MOVSHDUP / MOVSLDUP
StInl void i_movshdup	(__m128& dest, __m128 src)			{ dest = FtoF(_mm_movehdup_ps(FtoF(src))); }
StInl void i_movsldup	(__m128& dest, __m128 src)			{ dest = FtoF(_mm_moveldup_ps(FtoF(src))); }

// MOVUPS / MOVUPD
StInl void i_movups		(__m128& dest, memsrc_float*  src)	{ dest = FtoF(_mm_loadu_ps((float*) src)); }
StInl void i_movupd		(__m128& dest, memsrc_double* src)	{ dest = DtoF(_mm_loadu_pd((double*)src)); }
StInl void i_movups		(memdst_float*  dest, __m128 src)	{ _mm_storeu_ps((float*) dest, FtoF(src)); }
StInl void i_movupd		(memdst_double* dest, __m128 src)	{ _mm_storeu_pd((double*)dest, FtoD(src)); }

// MOVDDUP
StInl void i_movddup	(__m128& dest, __m128 src)			{ dest = DtoF(_mm_movedup_pd(FtoD(src))); }
StInl void i_movddup	(__m128& dest, memsrc_double* src)	{ dest = DtoF(_mm_loaddup_pd((double*)src)); }

// MOVDQU
StInl void i_movdqu		(__m128& dest, memsrc_m128i* src)	{ dest = ItoF(_mm_loadu_si128 ((__m128i*)src)); }
StInl void i_movdqu		(memdst_m128i*dest, __m128 src)		{ _mm_storeu_si128 ((__m128i*)dest, FtoI(src)); }

// MOVDQ2Q
StInl void i_movdq2q	(__m64&  dest, __m128 a)			{ dest =     (_mm_movepi64_pi64(FtoI(a))); }

// MOVHPS / MOVLPS / MOVHPD / MOVLPD
StInl void i_movhps		(__m128& dest, __m128 a, memsrc_u64* p)		{ dest = FtoF(_mm_loadh_pi(FtoF(a), (__m64*)p)); }
StInl void i_movlps		(__m128& dest, __m128 a, memsrc_u64* p)		{ dest = FtoF(_mm_loadl_pi(FtoF(a), (__m64*)p)); }
StInl void i_movhpd		(__m128& dest, __m128 a, memsrc_double* p)	{ dest = DtoF(_mm_loadh_pd(FtoD(a), (double*)p)); }
StInl void i_movlpd		(__m128& dest, __m128 a, memsrc_double* p)	{ dest = DtoF(_mm_loadl_pd(FtoD(a), (double*)p)); }
StInl void i_movhps		(memdst_u64*    dest, __m128 src)			{ _mm_storeh_pi((__m64*) dest, FtoF(src)); }
StInl void i_movlps		(memdst_u64*    dest, __m128 src)			{ _mm_storel_pi((__m64*) dest, FtoF(src)); }
StInl void i_movhpd		(memdst_double* dest, __m128 src)			{ _mm_storeh_pd((double*)dest, FtoD(src)); }
StInl void i_movlpd		(memdst_double* dest, __m128 src)			{ _mm_storel_pd((double*)dest, FtoD(src)); }

// MOVHLPS / MOVLHPS
StInl void i_movhlps	(__m128& dest, __m128 a, __m128 b)			{ dest = FtoF(_mm_movehl_ps(FtoF(a), FtoF(b))); }
StInl void i_movlhps	(__m128& dest, __m128 a, __m128 b)			{ dest = FtoF(_mm_movelh_ps(FtoF(a), FtoF(b))); }

// MOVMSKPS / MOVMSKPD / PMOVMSKB
StInl int  i_movmskps	(__m128 src)								{ return _mm_movemask_ps(FtoF(src)); }
StInl int  i_movmskpd	(__m128 src)								{ return _mm_movemask_pd(FtoD(src)); }
StInl int  i_pmovmskb	(__m128 src)								{ return _mm_movemask_epi8(FtoI(src)); }

// MOVNTPS / MOVNTPD / MOVNTI / MOVNTDQ / MOVNTDQA
StInl void i_movntps	(memsrc_float*	dest, __m128   src)			{ _mm_stream_ps		((float*)  dest, FtoF(src)); }
StInl void i_movntpd	(memsrc_double*	dest, __m128   src)			{ _mm_stream_pd		((double*) dest, FtoD(src)); }
StInl void i_movnti		(memsrc_u32*	dest, int      src)			{ _mm_stream_si32	((int*)    dest, src); }
StInl void i_movntdq	(memsrc_m128i*	dest, __m128   src)			{ _mm_stream_si128	((__m128i*)dest, FtoI(src)); }
StInl void i_movntdqa	(__m128&		dest, memsrc_m128i*	src)	{ dest = ItoF(_mm_stream_load_si128((__m128i*)src)); }

// MPSADBW
template<int mask> tmplInl void i_mpsadbw_(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_mpsadbw_epu8(FtoI(a), FtoI(b), mask)); }
#define i_mpsadbw(dest, a, b, mask) i_mpsadbw_<mask>(dest, a, b) // mask needs to be a constant expression / integer literal

// MUL
StInl void i_mulps		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_mul_ps(FtoF(a), FtoF(b))); }
StInl void i_mulss		(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_mul_ss(FtoF(a), FtoF(b))); }
StInl void i_mulpd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_mul_pd(FtoD(a), FtoD(b))); }
StInl void i_mulsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_mul_sd(FtoD(a), FtoD(b))); }

// PABS
StInl void i_pabsb		(__m128& dest, __m128 a)			{ dest = ItoF(_mm_abs_epi8 (FtoI(a))); }
StInl void i_pabsw		(__m128& dest, __m128 a)			{ dest = ItoF(_mm_abs_epi16(FtoI(a))); }
StInl void i_pabsd		(__m128& dest, __m128 a)			{ dest = ItoF(_mm_abs_epi32(FtoI(a))); }

// PAND / PANDN / POR / PXOR
StInl void i_pand		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_and_si128(FtoI(a), FtoI(b))); }
StInl void i_pandn		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_andnot_si128(FtoI(a), FtoI(b))); }
StInl void i_por		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_or_si128(FtoI(a), FtoI(b))); }
StInl void i_pxor		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_xor_si128(FtoI(a), FtoI(b))); }
StInl void i_pxor		(__m128& dest)						{ dest = ItoF(_mm_xor_si128(FtoI(dest), FtoI(dest))); }

// PTEST
StInl int i_ptestz		(__m128 a, __m128 b)				{ return (_mm_testz_si128  (FtoI(a), FtoI(b))); }
StInl int i_ptestc		(__m128 a, __m128 b)				{ return (_mm_testc_si128  (FtoI(a), FtoI(b))); }
StInl int i_ptestnzc	(__m128 a, __m128 b)				{ return (_mm_testnzc_si128(FtoI(a), FtoI(b))); }

// PADD / PSUB
StInl void i_paddb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_add_epi8 (FtoI(a), FtoI(b))); }
StInl void i_paddw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_add_epi16(FtoI(a), FtoI(b))); }
StInl void i_paddd		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_add_epi32(FtoI(a), FtoI(b))); }
StInl void i_paddq		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_add_epi64(FtoI(a), FtoI(b))); }
StInl void i_psubb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_sub_epi8 (FtoI(a), FtoI(b))); }
StInl void i_psubw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_sub_epi16(FtoI(a), FtoI(b))); }
StInl void i_psubd		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_sub_epi32(FtoI(a), FtoI(b))); }
StInl void i_psubq		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_sub_epi64(FtoI(a), FtoI(b))); }

// PADDS / PSUBS - Add/Sub with Signed Saturate
StInl void i_paddsb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_adds_epi8 (FtoI(a), FtoI(b))); }
StInl void i_paddsw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_adds_epi16(FtoI(a), FtoI(b))); }
StInl void i_psubsb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_subs_epi8 (FtoI(a), FtoI(b))); }
StInl void i_psubsw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_subs_epi16(FtoI(a), FtoI(b))); }

// PADDUS / PSUBUS - Add/Sub with Unsigned Saturate
StInl void i_paddusb	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_adds_epu8 (FtoI(a), FtoI(b))); }
StInl void i_paddusw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_adds_epu16(FtoI(a), FtoI(b))); }
StInl void i_psubusb	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_subs_epu8 (FtoI(a), FtoI(b))); }
StInl void i_psubusw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_subs_epu16(FtoI(a), FtoI(b))); }

// PACKSS / PACKUS
StInl void i_packsswb	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_packs_epi16 (FtoI(a), FtoI(b))); }
StInl void i_packssdw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_packs_epi32 (FtoI(a), FtoI(b))); }
StInl void i_packuswb	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_packus_epi16(FtoI(a), FtoI(b))); }
StInl void i_packusdw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_packus_epi32(FtoI(a), FtoI(b))); }

// PALIGNR
template<int imm8> tmplInl void i_alignr_(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_alignr_epi8(FtoI(a), FtoI(b), imm8)); }
#define i_align(dest, a, b, imm8) i_alignr_<imm8>(dest, a, b) // imm8 needs to be a constant expression / integer literal

// PAVG
StInl void i_pavgb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_avg_epu8 (FtoI(a), FtoI(b))); }
StInl void i_pavgw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_avg_epu16(FtoI(a), FtoI(b))); }

// PBLENDVB
StInl void i_pblendvb	(__m128& dest, __m128 a, __m128 b, __m128 mask)			{ dest = ItoF(_mm_blendv_epi8(FtoI(a), FtoI(b), FtoI(mask))); }

// PBLENDW	
template<int mask> tmplInl void i_pblendw_(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_blend_epi16(FtoI(a), FtoI(b), mask)); }
#define i_pblendw(dest, a, b, mask) i_pblendw_<mask>(dest, a, b)  // mask needs to be a constant expression / integer literal

#if defined(__PCLMUL__)
// PCLMULQDQ
template<int imm8> tmplInl void i_pclmuldq_(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_clmulepi64_si128(FtoI(a), FtoI(b), imm8)); }
#define i_pclmuldq(dest, a, b, imm8) i_pclmuldq_<imm8>(dest, a, b)  // imm8 needs to be a constant expression / integer literal
#endif

// PCMPEQ / PCMPGT
StInl void i_pcmpeqb(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpeq_epi8 (FtoI(a), FtoI(b))); }
StInl void i_pcmpeqw(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpeq_epi16(FtoI(a), FtoI(b))); }
StInl void i_pcmpeqd(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpeq_epi32(FtoI(a), FtoI(b))); }
StInl void i_pcmpeqq(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpeq_epi64(FtoI(a), FtoI(b))); }
StInl void i_pcmpgtb(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpgt_epi8 (FtoI(a), FtoI(b))); }
StInl void i_pcmpgtw(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpgt_epi16(FtoI(a), FtoI(b))); }
StInl void i_pcmpgtd(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpgt_epi32(FtoI(a), FtoI(b))); }
StInl void i_pcmpgtq(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpgt_epi64(FtoI(a), FtoI(b))); }

// PCMPESTRI / PCMPESTRM
template<int mode> tmplInl void	  i_pcmpestrm_(__m128& dest, __m128 a, int la, __m128 b, int lb) { dest = ItoF(_mm_cmpestrm(FtoI(a), la, FtoI(b), lb, mode)); }
template<int mode> tmplInl int    i_pcmpestri_(				 __m128 a, int la, __m128 b, int lb) { return     (_mm_cmpestri(FtoI(a), la, FtoI(b), lb, mode)); }
template<int mode> tmplInl int    i_pcmpestra_(				 __m128 a, int la, __m128 b, int lb) { return     (_mm_cmpestra(FtoI(a), la, FtoI(b), lb, mode)); }
template<int mode> tmplInl int    i_pcmpestrc_(				 __m128 a, int la, __m128 b, int lb) { return     (_mm_cmpestrc(FtoI(a), la, FtoI(b), lb, mode)); }
template<int mode> tmplInl int    i_pcmpestro_(				 __m128 a, int la, __m128 b, int lb) { return     (_mm_cmpestro(FtoI(a), la, FtoI(b), lb, mode)); }
template<int mode> tmplInl int    i_pcmpestrs_(				 __m128 a, int la, __m128 b, int lb) { return     (_mm_cmpestrs(FtoI(a), la, FtoI(b), lb, mode)); }
template<int mode> tmplInl int    i_pcmpestrz_(				 __m128 a, int la, __m128 b, int lb) { return     (_mm_cmpestrz(FtoI(a), la, FtoI(b), lb, mode)); }
#define i_pcmpestrm(dest,	a, la, b, lb, mode)	i_pcmpestrm_<mode>(dest, a, la, b, lb) // mode needs to be a constant expression / integer literal
#define i_pcmpestri(		a, la, b, lb, mode)	i_pcmpestri_<mode>(		 a, la, b, lb) // mode needs to be a constant expression / integer literal
#define i_pcmpestrc(		a, la, b, lb, mode)	i_pcmpestra_<mode>(		 a, la, b, lb) // mode needs to be a constant expression / integer literal
#define i_pcmpestro(		a, la, b, lb, mode)	i_pcmpestrc_<mode>(		 a, la, b, lb) // mode needs to be a constant expression / integer literal
#define i_pcmpestrs(		a, la, b, lb, mode)	i_pcmpestrs_<mode>(		 a, la, b, lb) // mode needs to be a constant expression / integer literal
#define i_pcmpestrz(		a, la, b, lb, mode)	i_pcmpestrz_<mode>(		 a, la, b, lb) // mode needs to be a constant expression / integer literal

// PCMPISTRI / PCMPISTRM
template<int mode> tmplInl void   i_pcmpistrm_(__m128& dest, __m128 a, __m128 b) { dest = ItoF(_mm_cmpistrm(FtoI(a), FtoI(b), mode)); }
template<int mode> tmplInl int    i_pcmpistri_(				 __m128 a, __m128 b) { return     (_mm_cmpistri(FtoI(a), FtoI(b), mode)); }
template<int mode> tmplInl int    i_pcmpistra_(				 __m128 a, __m128 b) { return     (_mm_cmpistra(FtoI(a), FtoI(b), mode)); }
template<int mode> tmplInl int    i_pcmpistrc_(				 __m128 a, __m128 b) { return     (_mm_cmpistrc(FtoI(a), FtoI(b), mode)); }
template<int mode> tmplInl int    i_pcmpistro_(				 __m128 a, __m128 b) { return     (_mm_cmpistro(FtoI(a), FtoI(b), mode)); }
template<int mode> tmplInl int    i_pcmpistrs_(				 __m128 a, __m128 b) { return     (_mm_cmpistrs(FtoI(a), FtoI(b), mode)); }
template<int mode> tmplInl int    i_pcmpistrz_(				 __m128 a, __m128 b) { return     (_mm_cmpistrz(FtoI(a), FtoI(b), mode)); }
#define i_pcmpistrm(dest,	a, b, mode) i_pcmpistrm_<mode>(dest, a, b) // mode needs to be a constant expression / integer literal
#define i_pcmpistri(		a, b, mode) i_pcmpistri_<mode>(		 a, b) // mode needs to be a constant expression / integer literal
#define i_pcmpistra(		a, b, mode) i_pcmpistra_<mode>(		 a, b) // mode needs to be a constant expression / integer literal
#define i_pcmpistrc(		a, b, mode) i_pcmpistrc_<mode>(		 a, b) // mode needs to be a constant expression / integer literal
#define i_pcmpistro(		a, b, mode) i_pcmpistro_<mode>(		 a, b) // mode needs to be a constant expression / integer literal
#define i_pcmpistrs(		a, b, mode) i_pcmpistrs_<mode>(		 a, b) // mode needs to be a constant expression / integer literal
#define i_pcmpistrz(		a, b, mode) i_pcmpistrz_<mode>(		 a, b) // mode needs to be a constant expression / integer literal

// PEXTR
template<int ndx> tmplInl int     i_pextrb_(__m128 src)		{ return (_mm_extract_epi8 (FtoI(src), ndx)); }
template<int ndx> tmplInl int     i_pextrw_(__m128 src)		{ return (_mm_extract_epi16(FtoI(src), ndx)); }
template<int ndx> tmplInl int     i_pextrd_(__m128 src)		{ return (_mm_extract_epi32(FtoI(src), ndx)); }
template<int ndx> tmplInl int64_t i_pextrq_(__m128 src)		{ return (_mm_extract_epi64(FtoI(src), ndx)); }
#define i_pextrb(m128_src, int_ndx) i_pextrb_<int_ndx>(m128_src) // ndx needs to be a constant expression / integer literal
#define i_pextrw(m128_src, int_ndx) i_pextrw_<int_ndx>(m128_src) // ndx needs to be a constant expression / integer literal
#define i_pextrd(m128_src, int_ndx) i_pextrd_<int_ndx>(m128_src) // ndx needs to be a constant expression / integer literal
#define i_pextrq(m128_src, int_ndx) i_pextrq_<int_ndx>(m128_src) // ndx needs to be a constant expression / integer literal

// PINSR
template<int ndx> tmplInl void i_pinsrb_(__m128& dest, __m128 a, int b)		{ dest = ItoF(_mm_insert_epi8 (FtoI(a), b, ndx)); }
template<int ndx> tmplInl void i_pinsrw_(__m128& dest, __m128 a, int b)		{ dest = ItoF(_mm_insert_epi16(FtoI(a), b, ndx)); }
template<int ndx> tmplInl void i_pinsrd_(__m128& dest, __m128 a, int b)		{ dest = ItoF(_mm_insert_epi32(FtoI(a), b, ndx)); }
template<int ndx> tmplInl void i_pinsrq_(__m128& dest, __m128 a, int64_t b)	{ dest = ItoF(_mm_insert_epi64(FtoI(a), b, ndx)); }
#define i_pinsrb(m128_dest, m128_a, int_b,   int_ndx) i_pinsrb_<int_ndx>(m128_dest, m128_a, int_b)   // ndx needs to be a constant expression / integer literal
#define i_pinsrw(m128_dest, m128_a, int_b,   int_ndx) i_pinsrw_<int_ndx>(m128_dest, m128_a, int_b)   // ndx needs to be a constant expression / integer literal
#define i_pinsrd(m128_dest, m128_a, int_b,   int_ndx) i_pinsrd_<int_ndx>(m128_dest, m128_a, int_b)   // ndx needs to be a constant expression / integer literal
#define i_pinsrq(m128_dest, m128_a, int64_b, int_ndx) i_pinsrq_<int_ndx>(m128_dest, m128_a, int64_b) // ndx needs to be a constant expression / integer literal

// PHADDW / PHADDD / PHADDSW / PHSUBW / PHSUBD / PHSUBSW
StInl void i_phaddw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_hadd_epi16 (FtoI(a), FtoI(b))); }
StInl void i_phaddd		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_hadd_epi32 (FtoI(a), FtoI(b))); }
StInl void i_phaddsw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_hadds_epi16(FtoI(a), FtoI(b))); }
StInl void i_phsubw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_hsub_epi16 (FtoI(a), FtoI(b))); }
StInl void i_phsubd		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_hsub_epi32 (FtoI(a), FtoI(b))); }
StInl void i_phsubsw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_hsubs_epi16(FtoI(a), FtoI(b))); }

// PHMINPOSUW
StInl void i_phminposuw	(__m128& dest, __m128 packed_words)	{ dest = ItoF(_mm_minpos_epu16(FtoI(packed_words))); }

// PMADDUBSW
StInl void i_pmaddubsw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_maddubs_epi16(FtoI(a), FtoI(b))); }

// PMADDWD
StInl void i_pmaddwd	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_madd_epi16(FtoI(a), FtoI(b))); }

// PMAXS / PMINS
StInl void i_pmaxsb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_max_epi8 (FtoI(a), FtoI(b))); }
StInl void i_pmaxsw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_max_epi16(FtoI(a), FtoI(b))); }
StInl void i_pmaxsd		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_max_epi32(FtoI(a), FtoI(b))); }
StInl void i_pminsb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_min_epi8 (FtoI(a), FtoI(b))); }
StInl void i_pminsw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_min_epi16(FtoI(a), FtoI(b))); }
StInl void i_pminsd		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_min_epi32(FtoI(a), FtoI(b))); }

// PMAXU / PMINU
StInl void i_pmaxub		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_max_epu8 (FtoI(a), FtoI(b))); }
StInl void i_pmaxuw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_max_epu16(FtoI(a), FtoI(b))); }
StInl void i_pmaxud		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_max_epu32(FtoI(a), FtoI(b))); }
StInl void i_pminub		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_min_epu8 (FtoI(a), FtoI(b))); }
StInl void i_pminuw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_min_epu16(FtoI(a), FtoI(b))); }
StInl void i_pminud		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_min_epu32(FtoI(a), FtoI(b))); }

// PMOVSX
StInl void i_pmovsxbw	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepi8_epi16 (FtoI(a))); }
StInl void i_pmovsxbd	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepi8_epi32 (FtoI(a))); }
StInl void i_pmovsxbq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepi8_epi64 (FtoI(a))); }
StInl void i_pmovsxwd	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepi16_epi32(FtoI(a))); }
StInl void i_pmovsxwq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepi16_epi64(FtoI(a))); }
StInl void i_pmovsxdq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepi32_epi64(FtoI(a))); }

// PMOVZX
StInl void i_pmovzxbw	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepu8_epi16 (FtoI(a))); }
StInl void i_pmovzxbd	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepu8_epi32 (FtoI(a))); }
StInl void i_pmovzxbq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepu8_epi64 (FtoI(a))); }
StInl void i_pmovzxwd	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepu16_epi32(FtoI(a))); }
StInl void i_pmovzxwq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepu16_epi64(FtoI(a))); }
StInl void i_pmovzxdq	(__m128& dest, __m128 a)			{ dest = ItoF(_mm_cvtepu32_epi64(FtoI(a))); }

// Intel doesn't provide from-memory overloads of PMOVZX, which I find rather inconvenient. --jstine
// (See Visual Studio Warning at top of file regarding the use of these from-memory intrinsics)

#if SSE_INTRIN_PMOV_HACK
StInl void i_pmovsxbw	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepi8_epi16 (_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovsxbd	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepi8_epi32 (_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovsxbq	(__m128& dest, memsrc_u16* src)		{ dest = ItoF(_mm_cvtepi8_epi64 (_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovsxwd	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepi16_epi32(_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovsxwq	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepi16_epi64(_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovsxdq	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepi32_epi64(_mm_loadu_si128((__m128i*)src))); }

StInl void i_pmovzxbw	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepu8_epi16 (_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovzxbd	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepu8_epi32 (_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovzxbq	(__m128& dest, memsrc_u16* src)		{ dest = ItoF(_mm_cvtepu8_epi64 (_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovzxwd	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepu16_epi32(_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovzxwq	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepu16_epi64(_mm_loadu_si128((__m128i*)src))); }
StInl void i_pmovzxdq	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepu32_epi64(_mm_loadu_si128((__m128i*)src))); }
#else
StInl void i_pmovsxbw	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepi8_epi16 (*(__m128i*)src)); }
StInl void i_pmovsxbd	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepi8_epi32 (*(__m128i*)src)); }
StInl void i_pmovsxbq	(__m128& dest, memsrc_u16* src)		{ dest = ItoF(_mm_cvtepi8_epi64 (*(__m128i*)src)); }
StInl void i_pmovsxwd	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepi16_epi32(*(__m128i*)src)); }
StInl void i_pmovsxwq	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepi16_epi64(*(__m128i*)src)); }
StInl void i_pmovsxdq	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepi32_epi64(*(__m128i*)src)); }

StInl void i_pmovzxbw	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepu8_epi16 (*(__m128i*)src)); }
StInl void i_pmovzxbd	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepu8_epi32 (*(__m128i*)src)); }
StInl void i_pmovzxbq	(__m128& dest, memsrc_u16* src)		{ dest = ItoF(_mm_cvtepu8_epi64 (*(__m128i*)src)); }
StInl void i_pmovzxwd	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepu16_epi32(*(__m128i*)src)); }
StInl void i_pmovzxwq	(__m128& dest, memsrc_u32* src)		{ dest = ItoF(_mm_cvtepu16_epi64(*(__m128i*)src)); }
StInl void i_pmovzxdq	(__m128& dest, memsrc_u64* src)		{ dest = ItoF(_mm_cvtepu32_epi64(*(__m128i*)src)); }
#endif


// PMULDQ / PMULHRSW / PMULHUW / PMULHW / PMULLW / PMULLD / PMULUDQ
StInl void i_pmuldq		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mul_epi32(FtoI(a), FtoI(b))); }
StInl void i_pmulhrsw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mulhrs_epi16(FtoI(a), FtoI(b))); }
StInl void i_pmulhuw	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mulhi_epu16(FtoI(a), FtoI(b))); }
StInl void i_pmulhw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mulhi_epi16(FtoI(a), FtoI(b))); }
StInl void i_pmullw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mullo_epi16(FtoI(a), FtoI(b))); }
StInl void i_pmulld		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mullo_epi32(FtoI(a), FtoI(b))); }
StInl void i_pmuludq	(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_mul_epu32(FtoI(a), FtoI(b))); }

// PSADBW - Compute Sum of Absolute Differences
StInl void i_psadbw		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_sad_epu8(FtoI(a), FtoI(b))); }

// PSHUFB 
StInl void i_pshufb		(__m128& dest, __m128 a, __m128 b)	{ dest = ItoF(_mm_shuffle_epi8(FtoI(a), FtoI(b))); }

// PSHUFD / PSHUFHW / PSHUFLW
template<int n> tmplInl void i_pshufd_ (__m128& dest,__m128 a){ dest = ItoF(_mm_shuffle_epi32(FtoI(a), n)); }
template<int n> tmplInl void i_pshufhw_(__m128& dest,__m128 a){ dest = ItoF(_mm_shufflehi_epi16(FtoI(a), n)); }
template<int n> tmplInl void i_pshuflw_(__m128& dest,__m128 a){ dest = ItoF(_mm_shufflelo_epi16(FtoI(a), n)); }
#define i_pshufd(dest, a, imm)  i_pshufd_ <imm>(dest, a) // imm needs to be a constant expression / integer literal
#define i_pshufhw(dest, a, imm) i_pshufhw_<imm>(dest, a) // imm needs to be a constant expression / integer literal
#define i_pshuflw(dest, a, imm) i_pshuflw_<imm>(dest, a) // imm needs to be a constant expression / integer literal

// PSIGN
StInl void i_psignb		(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_sign_epi8 (FtoI(a), FtoI(b))); }
StInl void i_psignw		(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_sign_epi16(FtoI(a), FtoI(b))); }
StInl void i_psignd		(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_sign_epi32(FtoI(a), FtoI(b))); }

// PSLL / PSRL
StInl void i_psllw		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_sll_epi16 (FtoI(a), FtoI(shift))); }
StInl void i_psllw		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_slli_epi16(FtoI(a), shift)); }
StInl void i_pslld		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_sll_epi32 (FtoI(a), FtoI(shift))); }
StInl void i_pslld		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_slli_epi32(FtoI(a), shift)); }
StInl void i_psllq		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_sll_epi64 (FtoI(a), FtoI(shift))); }
StInl void i_psllq		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_slli_epi64(FtoI(a), shift)); }
StInl void i_psrlw		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_srl_epi16 (FtoI(a), FtoI(shift))); }
StInl void i_psrlw		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_srli_epi16(FtoI(a), shift)); }
StInl void i_psrld		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_srl_epi32 (FtoI(a), FtoI(shift))); }
StInl void i_psrld		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_srli_epi32(FtoI(a), shift)); }
StInl void i_psrlq		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_srl_epi64 (FtoI(a), FtoI(shift))); }
StInl void i_psrlq		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_srli_epi64(FtoI(a), shift)); }
template<int shift> tmplInl void i_pslldq_(__m128& dest, __m128 a){ dest = ItoF(_mm_slli_si128(FtoI(a), shift)); }
template<int shift> tmplInl void i_psrldq_(__m128& dest, __m128 a){ dest = ItoF(_mm_srli_si128(FtoI(a), shift)); }
#define i_pslldq(dest, a, shift) i_pslldq_<shift>(dest, a) // shift needs to be a constant expression / integer literal
#define i_psrldq(dest, a, shift) i_psrldq_<shift>(dest, a) // shift needs to be a constant expression / integer literal

// PSRA
StInl void i_psraw		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_sra_epi16 (FtoI(a), FtoI(shift))); }
StInl void i_psraw		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_srai_epi16(FtoI(a), shift)); }
StInl void i_psrad		(__m128& dest, __m128 a, __m128 shift)	{ dest = ItoF(_mm_sra_epi32 (FtoI(a), FtoI(shift))); }
StInl void i_psrad		(__m128& dest, __m128 a, int    shift)	{ dest = ItoF(_mm_srai_epi32(FtoI(a), shift)); }

// PUNPCKH / PUNPCKL
StInl void i_punpckhbw	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpackhi_epi8 (FtoI(a), FtoI(b))); }
StInl void i_punpckhwd	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpackhi_epi16(FtoI(a), FtoI(b))); }
StInl void i_punpckhdq	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpackhi_epi32(FtoI(a), FtoI(b))); }
StInl void i_punpckhqdq	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpackhi_epi64(FtoI(a), FtoI(b))); }
StInl void i_punpcklbw	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpacklo_epi8 (FtoI(a), FtoI(b))); }
StInl void i_punpcklwd	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpacklo_epi16(FtoI(a), FtoI(b))); }
StInl void i_punpckldq	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpacklo_epi32(FtoI(a), FtoI(b))); }
StInl void i_punpcklqdq	(__m128& dest, __m128 a, __m128 b)		{ dest = ItoF(_mm_unpacklo_epi64(FtoI(a), FtoI(b))); }

// RCP
StInl void i_rcpps		(__m128& dest, __m128 a)				{ dest = FtoF(_mm_rcp_ps(FtoF(a))); }
StInl void i_rcpss		(__m128& dest, __m128 a)				{ dest = FtoF(_mm_rcp_ss(FtoF(a))); }

// ROUND
template<int rMode> tmplInl void i_roundps_(__m128& dest, __m128 a)			{ dest = FtoF(_mm_round_ps(FtoF(a),          rMode)); }
template<int rMode> tmplInl void i_roundpd_(__m128& dest, __m128 a)			{ dest = DtoF(_mm_round_pd(FtoD(a),          rMode)); }
template<int rMode> tmplInl void i_roundss_(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_round_ss(FtoF(a), FtoF(b), rMode)); }
template<int rMode> tmplInl void i_roundsd_(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_round_sd(FtoD(a), FtoD(b), rMode)); }
#define i_roundps(m128_dest, m128_a,         int_roundMode) i_roundps_<int_roundMode>(m128_dest, m128_a)         // RoundMode needs to be a constant expression / integer literal
#define i_roundpd(m128_dest, m128_a,         int_roundMode) i_roundpd_<int_roundMode>(m128_dest, m128_a)         // RoundMode needs to be a constant expression / integer literal
#define i_roundss(m128_dest, m128_a, m128_b, int_roundMode) i_roundss_<int_roundMode>(m128_dest, m128_a, m128_b) // RoundMode needs to be a constant expression / integer literal
#define i_roundsd(m128_dest, m128_a, m128_b, int_roundMode) i_roundsd_<int_roundMode>(m128_dest, m128_a, m128_b) // RoundMode needs to be a constant expression / integer literal

// FLOOR / CEIL (internally these use ROUND with a specified round-mode)
StInl void i_floorps	(__m128& dest, __m128 a)							{ dest = FtoF(_mm_floor_ps(FtoF(a))); }
StInl void i_floorss	(__m128& dest, __m128 a, __m128 b)					{ dest = FtoF(_mm_floor_ss(FtoF(a), FtoF(b))); }
StInl void i_floorpd	(__m128& dest, __m128 a)							{ dest = DtoF(_mm_floor_pd(FtoD(a))); }
StInl void i_floorsd	(__m128& dest, __m128 a, __m128 b)					{ dest = DtoF(_mm_floor_sd(FtoD(a), FtoD(b))); }
StInl void i_ceilps		(__m128& dest, __m128 a)							{ dest = FtoF(_mm_ceil_ps(FtoF(a))); }
StInl void i_ceilss		(__m128& dest, __m128 a, __m128 b)					{ dest = FtoF(_mm_ceil_ss(FtoF(a), FtoF(b))); }
StInl void i_ceilpd		(__m128& dest, __m128 a)							{ dest = DtoF(_mm_ceil_pd(FtoD(a))); }
StInl void i_ceilsd		(__m128& dest, __m128 a, __m128 b)					{ dest = DtoF(_mm_ceil_sd(FtoD(a), FtoD(b))); }

// SHUF
template<uint32_t imm8> tmplInl void i_shufps_(__m128&dest,__m128 a,__m128 b)	{ dest = FtoF(_mm_shuffle_ps(FtoF(a), FtoF(b), imm8)); }
template<uint32_t imm8> tmplInl void i_shufpd_(__m128&dest,__m128 a,__m128 b)	{ dest = DtoF(_mm_shuffle_pd(FtoD(a), FtoD(b), imm8)); }
#define i_shufps(dest, a, b, imm8) i_shufps_<imm8>(dest, a, b) // imm8 needs to be a constant expression / integer literal
#define i_shufpd(dest, a, b, imm8) i_shufpd_<imm8>(dest, a, b) // imm8 needs to be a constant expression / integer literal

// SQRT / RSQRT
StInl void i_sqrtps		(__m128& dest, __m128 a)			{ dest = FtoF(_mm_sqrt_ps(FtoF(a))); }
StInl void i_sqrtss		(__m128& dest, __m128 a)			{ dest = FtoF(_mm_sqrt_ss(FtoF(a))); }
StInl void i_sqrtpd		(__m128& dest, __m128 a)			{ dest = DtoF(_mm_sqrt_pd(FtoD(a))); }
StInl void i_sqrtsd		(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_sqrt_sd(FtoD(a), FtoD(b))); }
StInl void i_rsqrtps	(__m128& dest, __m128 a)			{ dest = FtoF(_mm_rsqrt_ps(FtoF(a))); }
StInl void i_rsqrtss	(__m128& dest, __m128 a)			{ dest = FtoF(_mm_rsqrt_ss(FtoF(a))); }

// UNPCKH / UNPCKL
StInl void i_unpckhps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_unpackhi_ps(FtoF(a), FtoF(b))); }
StInl void i_unpckhpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_unpackhi_pd(FtoD(a), FtoD(b))); }
StInl void i_unpcklps	(__m128& dest, __m128 a, __m128 b)	{ dest = FtoF(_mm_unpacklo_ps(FtoF(a), FtoF(b))); }
StInl void i_unpcklpd	(__m128& dest, __m128 a, __m128 b)	{ dest = DtoF(_mm_unpacklo_pd(FtoD(a), FtoD(b))); }

//------------------------------------------------------------------
// Misc Instructions
//------------------------------------------------------------------

// CLFLUSH
StInl void			i_clflush	(void const* p)				{ _mm_clflush(p); }

// CRC32
// (and yes, the 64-bit CRC32 generates an effective 32-bit result)

StInl uint32_t		i_crc32	(uint32_t crc, uint8_t  data)	{ return _mm_crc32_u8 (crc, data); }
StInl uint32_t		i_crc32	(uint32_t crc, uint16_t data)	{ return _mm_crc32_u16(crc, data); }
StInl uint32_t		i_crc32	(uint32_t crc, uint32_t data)	{ return _mm_crc32_u32(crc, data); }
StInl uint32_t		i_crc32	(uint32_t crc, uint64_t data)	{ return _mm_crc32_u64(crc, data); }

// EMMS
StInl void			i_emms		(void)						{ _mm_empty(); }

// LDMXCSR / STMXCSR
StInl void			i_ldmxcsr	(uint32_t i)				{ _mm_setcsr(i); }
StInl unsigned int	i_getcsr	(void)						{ return _mm_getcsr(); }

// LFENCE / MFENCE / SFENCE
StInl void			i_lfence	(void)						{ _mm_lfence(); }
StInl void			i_mfence	(void)						{ _mm_mfence(); }
StInl void			i_sfence	(void)						{ _mm_sfence(); }

// PAUSE
StInl void			i_pause		(void)						{ _mm_pause(); }

// POPCNT
StInl int			i_popcnt	(uint32_t a)				{ return _mm_popcnt_u32(a); }
StInl int64_t		i_popcnt	(uint64_t a)				{ return _mm_popcnt_u64(a); }

// MONITOR / MWAIT / PREFETCHh
#define i_monitor(ptr, extensions, hints)	_mm_monitor(ptr, extensions, hints)
#define i_mwait(extensions, hints)			_mm_mwait(extensions, hints)
#define i_prefetch(a, sel)					_mm_prefetch(a, sel)


// SPLATS
// These do not have direct asm connections.  The wrappers use industry standard terms and
// also return typeless __m128 (usable in all wrapper functions without cast).

StInl void	i_splat8	(__m128& dest, uint8_t val)				{ dest = ItoF(_mm_set1_epi8(val)); }
StInl void	i_splat16	(__m128& dest, uint16_t val)			{ dest = ItoF(_mm_set1_epi16(val)); }
StInl void	i_splat32	(__m128& dest, uint32_t val)			{ dest = ItoF(_mm_set1_epi32(val)); }
//StInl void	i_splat64	(__m128& dest, uint64_t val)			{ dest = ItoF(_mm_set1_epi64(val)); }

StInl void	i_splat_ps	(__m128& dest, float val)				{ dest = FtoF(_mm_set1_ps((float&)val)); }
StInl void	i_splat_pd	(__m128& dest, double val)				{ dest = DtoF(_mm_set1_pd((double&)val)); }


SSE_INTRIN_OPTIMIZE_END

#undef FtoD
#undef DtoF
#undef FtoI
#undef ItoF
#undef FtoF
#undef StInl
#undef tmplInl