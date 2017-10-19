/*
 * Copyright (c) 2009-2017, Farooq Mela
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _CORE_STACKTRACE_H_
#define _CORE_STACKTRACE_H_

#include <execinfo.h> // for backtrace
#include <dlfcn.h>    // for dladdr
#include <cxxabi.h>   // for __cxa_demangle

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>

namespace Core
{


// This function produces a stack backtrace with demangled function & method names.
std::string stack_trace(int skip = 1)
{
    void *callstack[128];
    const int nMaxFrames = sizeof(callstack) / sizeof(callstack[0]);
    char buf[1024];
    int nFrames = backtrace(callstack, nMaxFrames);
    char **symbols = backtrace_symbols(callstack, nFrames);

    std::ostringstream trace_buf;
    for (int i = skip; i < nFrames; i++) {

        Dl_info info;
        if (dladdr(callstack[i], &info) && info.dli_sname) {
            char *demangled = NULL;
            int status = -1;
            if (info.dli_sname[0] == '_') {
              demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
            }
            snprintf(buf, sizeof(buf), "%-3d %*p %s + %zd\n",
                     i, int(2 + sizeof(void*) * 2), callstack[i],
                     status == 0 ? demangled :
                     info.dli_sname == 0 ? symbols[i] : info.dli_sname,
                     (char *)callstack[i] - (char *)info.dli_saddr);
            free(demangled);

            trace_buf << buf;
        }
        else { /* only print resolved symbols */
          //          snprintf(buf, sizeof(buf), "%-3d %*p %s\n", i, int(2 + sizeof(void*) * 2), callstack[i], symbols[i]);
        }

    }
    free(symbols);
    if (nFrames == nMaxFrames)
        trace_buf << "[truncated]\n";
    return trace_buf.str();
}

// std::string stack_trace()
// {
//   std::stringstream trace_output;
//   void *array[10];
//   size_t size;
//   char **strings;
//   size_t i;

//   size = backtrace (array, 10);
//   strings = backtrace_symbols (array, size);

//   printf ("Obtained %zd stack frames.\n", size);

//   for (i = 0; i < size; i++)
//      printf ("%s\n", strings[i]);

//   ::free(strings);
//   return trace_output.str();
// }

// /** Print a demangled stack backtrace of the caller function to FILE* out. */
// static inline void print_stacktrace(FILE *out = stderr, unsigned int max_frames = 63)
// {
//     fprintf(out, "stack trace:\n");

//     // storage array for stack trace address data
//     void* addrlist[max_frames+1];

//     // retrieve current stack addresses
//     int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

//     if (addrlen == 0) {
// 	fprintf(out, "  <empty, possibly corrupt>\n");
// 	return;
//     }

//     // resolve addresses into strings containing "filename(function+address)",
//     // this array must be free()-ed
//     char** symbollist = backtrace_symbols(addrlist, addrlen);

//     // allocate string which will be filled with the demangled function name
//     size_t funcnamesize = 256;
//     char* funcname = (char*)malloc(funcnamesize);

//     // iterate over the returned symbol lines. skip the first, it is the
//     // address of this function.
//     for (int i = 1; i < addrlen; i++)
//     {
// 	char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

// 	// find parentheses and +address offset surrounding the mangled name:
// 	// ./module(function+0x15c) [0x8048a6d]
// 	for (char *p = symbollist[i]; *p; ++p)
// 	{
// 	    if (*p == '(')
// 		begin_name = p;
// 	    else if (*p == '+')
// 		begin_offset = p;
// 	    else if (*p == ')' && begin_offset) {
// 		end_offset = p;
// 		break;
// 	    }
// 	}

// 	if (begin_name && begin_offset && end_offset
// 	    && begin_name < begin_offset)
// 	{
// 	    *begin_name++ = '\0';
// 	    *begin_offset++ = '\0';
// 	    *end_offset = '\0';

// 	    // mangled name is now in [begin_name, begin_offset) and caller
// 	    // offset in [begin_offset, end_offset). now apply
// 	    // __cxa_demangle():

// 	    int status;
// 	    char* ret = abi::__cxa_demangle(begin_name,
// 					    funcname, &funcnamesize, &status);
// 	    if (status == 0) {
// 		funcname = ret; // use possibly realloc()-ed string
// 		fprintf(out, "  %s : %s+%s\n",
// 			symbollist[i], funcname, begin_offset);
// 	    }
// 	    else {
// 		// demangling failed. Output function name as a C function with
// 		// no arguments.
// 		fprintf(out, "  %s : %s()+%s\n",
// 			symbollist[i], begin_name, begin_offset);
// 	    }
// 	}
// 	else
// 	{
// 	    // couldn't parse the line? print the whole line.
// 	    fprintf(out, "??  %s\n", symbollist[i]);
// 	}
//     }

//     free(funcname);
//     free(symbollist);
// }

}  // namespace Core
#endif // _CORE_STACKTRACE_H_
