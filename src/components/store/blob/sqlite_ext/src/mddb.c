/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
** 2016-05-28
**
** The author disclaims copyright to this source code.  In place of
** a legal notice, here is a blessing:
**
**    May you do good and not evil.
**    May you find forgiveness for yourself and forgive others.
**    May you share freely, never taking more than you give.
**
******************************************************************************
**
** This file contains the implementation of an SQLite virtual table for
** reading CSV files.
**
** Usage:
**
**    .load ./csv
**    CREATE VIRTUAL TABLE temp.csv USING csv(filename=FILENAME);
**    SELECT * FROM csv;
**
** The columns are named "c1", "c2", "c3", ... by default.  But the
** application can define its own CREATE TABLE statement as an additional
** parameter.  For example:
**
**    CREATE VIRTUAL TABLE temp.csv2 USING csv(
**       filename = "../http.log",
**       schema = "CREATE TABLE x(date,ipaddr,url,referrer,userAgent)"
**    );
**
** Instead of specifying a file, the text of the CSV can be loaded using
** the data= parameter.
**
** If the columns=N parameter is supplied, then the CSV file is assumed to have
** N columns.  If the columns parameter is omitted, the CSV file is opened
** as soon as the virtual table is constructed and the first row of the CSV
** is read in order to count the tables.
**
** Some extra debugging features (used for testing virtual tables) are available
** if this module is compiled with -DSQLITE_TEST.
*/
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdio.h>
#include <common/logging.h>
#include "md.h"

#ifndef SQLITE_OMIT_VIRTUALTABLE

#define NOINLINE  __attribute__((noinline))


/* Max size of the error message in a CsvReader */
#define MDDB_MXERR 200

/* Size of the MddbReader input buffer */
#define MDDB_INBUFSZ 1024

/* A context object used when read a MDDB file. */
typedef struct MddbReader MddbReader;
struct MddbReader {
  FILE *in;              /* Read the MDDB text from this input stream */
  char *z;               /* Accumulated text for a field */
  int n;                 /* Number of bytes in z */
  int nAlloc;            /* Space allocated for z[] */
  int nLine;             /* Current line number */
  int bNotFirst;         /* True if prior text has been seen */
  int cTerm;             /* Character that terminated the most recent field */
  size_t iIn;            /* Next unread character in the input buffer */
  size_t nIn;            /* Number of characters in the input buffer */
  char *zIn;             /* The input buffer */
  char zErr[MDDB_MXERR];  /* Error message */
};

/* Initialize a MddbReader object */
static void mddb_reader_init(MddbReader *p){
  p->in = 0;
  p->z = 0;
  p->n = 0;
  p->nAlloc = 0;
  p->nLine = 0;
  p->bNotFirst = 0;
  p->nIn = 0;
  p->zIn = 0;
  p->zErr[0] = 0;
}

/* Close and reset a MddbReader object */
static void mddb_reader_reset(MddbReader *p){
  if( p->in ){
    fclose(p->in);
    sqlite3_free(p->zIn);
  }
  sqlite3_free(p->z);
  mddb_reader_init(p);
}

/* Report an error on a MddbReader */
static void mddb_errmsg(MddbReader *p, const char *zFormat, ...){
  va_list ap;
  va_start(ap, zFormat);
  sqlite3_vsnprintf(MDDB_MXERR, p->zErr, zFormat, ap);
  va_end(ap);
}

/* Open the file associated with a MddbReader
** Return the number of errors.
*/
static int mddb_reader_open(MddbReader *p,               /* The reader to open */
                            const char *zFilename,      /* Read from this filename */
                            const char *zData)           /*  ... or use this data */
{
  if( zFilename ){
    p->zIn = sqlite3_malloc( MDDB_INBUFSZ );
    if( p->zIn==0 ){
      mddb_errmsg(p, "out of memory");
      return 1;
    }
    p->in = fopen(zFilename, "rb");
    if( p->in==0 ){
      mddb_reader_reset(p);
      mddb_errmsg(p, "cannot open '%s' for reading", zFilename);
      return 1;
    }
  }else{
    assert( p->in==0 );
    p->zIn = (char*)zData;
    p->nIn = strlen(zData);
  }
  return 0;
}

/* The input buffer has overflowed.  Refill the input buffer, then
** return the next character
*/
static NOINLINE int mddb_getc_refill(MddbReader *p){
  size_t got;

  assert( p->iIn>=p->nIn );  /* Only called on an empty input buffer */
  assert( p->in!=0 );        /* Only called if reading froma file */

  got = fread(p->zIn, 1, MDDB_INBUFSZ, p->in);
  if( got==0 ) return EOF;
  p->nIn = got;
  p->iIn = 1;
  return p->zIn[0];
}

/* Return the next character of input.  Return EOF at end of input. */
static int mddb_getc(MddbReader *p){
  if( p->iIn >= p->nIn ){
    if( p->in!=0 ) return mddb_getc_refill(p);
    return EOF;
  }
  return ((unsigned char*)p->zIn)[p->iIn++];
}

/* Increase the size of p->z and append character c to the end. 
** Return 0 on success and non-zero if there is an OOM error */
static NOINLINE int mddb_resize_and_append(MddbReader *p, char c){
  char *zNew;
  int nNew = p->nAlloc*2 + 100;
  zNew = sqlite3_realloc64(p->z, nNew);
  if( zNew ){
    p->z = zNew;
    p->nAlloc = nNew;
    p->z[p->n++] = c;
    return 0;
  }else{
    mddb_errmsg(p, "out of memory");
    return 1;
  }
}

/* Append a single character to the MddbReader.z[] array.
** Return 0 on success and non-zero if there is an OOM error */
static int mddb_append(MddbReader *p, char c){
  if( p->n>=p->nAlloc-1 ) return mddb_resize_and_append(p, c);
  p->z[p->n++] = c;
  return 0;
}

/* Read a single field of MDDB text.  Compatible with rfc4180 and extended
** with the option of having a separator other than ",".
**
**   +  Input comes from p->in.
**   +  Store results in p->z of length p->n.  Space to hold p->z comes
**      from sqlite3_malloc64().
**   +  Keep track of the line number in p->nLine.
**   +  Store the character that terminates the field in p->cTerm.  Store
**      EOF on end-of-file.
**
** Return "" at EOF.  Return 0 on an OOM error.
*/
static char *mddb_read_one_field(MddbReader *p){
  int c;
  p->n = 0;
  c = mddb_getc(p);
  if( c==EOF ){
    p->cTerm = EOF;
    return "";
  }
  if( c=='"' ){
    int pc, ppc;
    int startLine = p->nLine;
    pc = ppc = 0;
    while( 1 ){
      c = mddb_getc(p);
      if( c<='"' || pc=='"' ){
        if( c=='\n' ) p->nLine++;
        if( c=='"' ){
          if( pc=='"' ){
            pc = 0;
            continue;
          }
        }
        if( (c==',' && pc=='"')
            || (c=='\n' && pc=='"')
            || (c=='\n' && pc=='\r' && ppc=='"')
            || (c==EOF && pc=='"')
            ){
          do{ p->n--; }while( p->z[p->n]!='"' );
          p->cTerm = (char)c;
          break;
        }
        if( pc=='"' && c!='\r' ){
          mddb_errmsg(p, "line %d: unescaped %c character", p->nLine, '"');
          break;
        }
        if( c==EOF ){
          mddb_errmsg(p, "line %d: unterminated %c-quoted field\n",
                     startLine, '"');
          p->cTerm = (char)c;
          break;
        }
      }
      if( mddb_append(p, (char)c) ) return 0;
      ppc = pc;
      pc = c;
    }
  }else{
    /* If this is the first field being parsed and it begins with the
    ** UTF-8 BOM  (0xEF BB BF) then skip the BOM */
    if( (c&0xff)==0xef && p->bNotFirst==0 ){
      mddb_append(p, (char)c);
      c = mddb_getc(p);
      if( (c&0xff)==0xbb ){
        mddb_append(p, (char)c);
        c = mddb_getc(p);
        if( (c&0xff)==0xbf ){
          p->bNotFirst = 1;
          p->n = 0;
          return mddb_read_one_field(p);
        }
      }
    }
    while( c>',' || (c!=EOF && c!=',' && c!='\n') ){
      if( mddb_append(p, (char)c) ) return 0;
      c = mddb_getc(p);
    }
    if( c=='\n' ){
      p->nLine++;
      if( p->n>0 && p->z[p->n-1]=='\r' ) p->n--;
    }
    p->cTerm = (char)c;
  }
  if( p->z ) p->z[p->n] = 0;
  p->bNotFirst = 1;
  return p->z;
}


/* Forward references to the various virtual table methods implemented
** in this file. */
static int tableCreate(sqlite3*, void*, int, const char*const*, 
                        sqlite3_vtab**,char**);
static int tableConnect(sqlite3*, void*, int, const char*const*, 
                         sqlite3_vtab**,char**);
static int tableBestIndex(sqlite3_vtab*,sqlite3_index_info*);
static int tableDisconnect(sqlite3_vtab*);
static int tableOpen(sqlite3_vtab*, sqlite3_vtab_cursor**);
static int tableClose(sqlite3_vtab_cursor*);
static int tableFilter(sqlite3_vtab_cursor*, int idxNum, const char *idxStr,
                        int argc, sqlite3_value **argv);
static int tableNext(sqlite3_vtab_cursor*);
static int tableEof(sqlite3_vtab_cursor*);
static int tableColumn(sqlite3_vtab_cursor*,sqlite3_context*,int);
static int tableRowid(sqlite3_vtab_cursor*,sqlite3_int64*);

/* An instance of the MDDB virtual table */
/* typedef struct Tablele { */
/*   sqlite3_vtab base;              /\* Base class.  Must be first *\/ */
/*   struct Metadata * mdobj; */
/*   int nCol;                       /\* Number of columns in the MDDB file *\/ */
/*   unsigned int tstFlags;          /\* Bit values used for testing *\/ */
/* } Tablele; */

/* Allowed values for tstFlags */
#define MDDBTEST_FIDX  0x0001      /* Pretend that constrained searchs cost less*/

/* A cursor for the MDDB virtual table */
typedef struct MddbCursor {
  sqlite3_vtab_cursor base;       /* Base class.  Must be first */
  MddbReader rdr;                 /* The MddbReader object */
  char **azVal;                   /* Value of the current row */
  int *aLen;                      /* Length of each entry */
  sqlite3_int64 iRowid;           /* The current rowid.  Negative for EOF */
} MddbCursor;

/* /\* Transfer error message text from a reader into a Tablele *\/ */
/* static void mddb_xfer_error(Tablele *pTab, MddbReader *pRdr){ */
/*   sqlite3_free(pTab->base.zErrMsg); */
/*   pTab->base.zErrMsg = sqlite3_mprintf("%s", pRdr->zErr); */
/* } */

/*
** This method is the destructor fo a Tablele object.
*/
static int tableDisconnect(sqlite3_vtab *pVtab){
  PLOG("Disconnecting %p", pVtab);
  mddb_free_instance(pVtab);
  
  return SQLITE_OK;
}

/* Skip leading whitespace.  Return a pointer to the first non-whitespace
** character, or to the zero terminator if the string has only whitespace */
static const char *mddb_skip_whitespace(const char *z){
  while( isspace((unsigned char)z[0]) ) z++;
  return z;
}

/* Remove trailing whitespace from the end of string z[] */
static void mddb_trim_whitespace(char *z){
  size_t n = strlen(z);
  while( n>0 && isspace((unsigned char)z[n]) ) n--;
  z[n] = 0;
}

/* Dequote the string */
static void mddb_dequote(char *z){
  int j;
  char cQuote = z[0];
  size_t i, n;

  if( cQuote!='\'' && cQuote!='"' ) return;
  n = strlen(z);
  if( n<2 || z[n-1]!=z[0] ) return;
  for(i=1, j=0; i<n-1; i++){
    if( z[i]==cQuote && z[i+1]==cQuote ) i++;
    z[j++] = z[i];
  }
  z[j] = 0;
}

/* Check to see if the string is of the form:  "TAG = VALUE" with optional
** whitespace before and around tokens.  If it is, return a pointer to the
** first character of VALUE.  If it is not, return NULL.
*/
static const char *mddb_parameter(const char *zTag, int nTag, const char *z){
  z = mddb_skip_whitespace(z);
  if( strncmp(zTag, z, nTag)!=0 ) return 0;
  z = mddb_skip_whitespace(z+nTag);
  if( z[0]!='=' ) return 0;
  return mddb_skip_whitespace(z+1);
}

/* Decode a parameter that requires a dequoted string.
**
** Return 1 if the parameter is seen, or 0 if not.  1 is returned
** even if there is an error.  If an error occurs, then an error message
** is left in p->zErr.  If there are no errors, p->zErr[0]==0.
*/
static int mddb_string_parameter(
                                MddbReader *p,            /* Leave the error message here, if there is one */
                                const char *zParam,      /* Parameter we are checking for */
                                const char *zArg,        /* Raw text of the virtual table argment */
                                char **pzVal             /* Write the dequoted string value here */
                                ){
  const char *zValue;
  zValue = mddb_parameter(zParam,(int)strlen(zParam),zArg);
  if( zValue==0 ) return 0;
  p->zErr[0] = 0;
  if( *pzVal ){
    mddb_errmsg(p, "more than one '%s' parameter", zParam);
    return 1;
  }
  *pzVal = sqlite3_mprintf("%s", zValue);
  if( *pzVal==0 ){
    mddb_errmsg(p, "out of memory");
    return 1;
  }
  mddb_trim_whitespace(*pzVal);
  mddb_dequote(*pzVal);
  return 1;
}


/* Return 0 if the argument is false and 1 if it is true.  Return -1 if
** we cannot really tell.
*/
static int mddb_boolean(const char *z){
  if( sqlite3_stricmp("yes",z)==0
      || sqlite3_stricmp("on",z)==0
      || sqlite3_stricmp("true",z)==0
      || (z[0]=='1' && z[1]==0)
      ){
    return 1;
  }
  if( sqlite3_stricmp("no",z)==0
      || sqlite3_stricmp("off",z)==0
      || sqlite3_stricmp("false",z)==0
      || (z[0]=='0' && z[1]==0)
      ){
    return 0;
  }
  return -1;
}


/*
** Parameters:
**    filename=FILENAME          Name of file containing MDDB content
**    data=TEXT                  Direct MDDB content.
**    schema=SCHEMA              Alternative MDDB schema.
**    header=YES|NO              First row of MDDB defines the names of
**                               columns if "yes".  Default "no".
**    columns=N                  Assume the MDDB file contains N columns.
**
** Only available if compiled with SQLITE_TEST:
**    
**    testflags=N                Bitmask of test flags.  Optional
**
** If schema= is omitted, then the columns are named "c0", "c1", "c2",
** and so forth.  If columns=N is omitted, then the file is opened and
** the number of columns in the first row is counted to determine the
** column count.  If header=YES, then the first row is skipped.
*/
static int tableConnect(sqlite3 *db,
                        void *pAux,
                        int argc, const char *const*argv,
                        sqlite3_vtab **ppVtab,
                        char **pzErr)
{
  TRACE();
  
  int rc = SQLITE_OK;        /* Result code from this routine */
  int nCol = -99;            /* Value of the columns= parameter */
  MddbReader sRdr;
  int i,j;
  static const char *azParam[] = {
    "pci", "partition", "schema", 
  };
  char *azPValue[3];         /* Parameter values */
# define MDDB_PCI       (azPValue[0])
# define MDDB_PARTITION (azPValue[1])
# define MDDB_SCHEMA    (azPValue[2])
  struct Metadata * mdptr = NULL;
  
  assert( sizeof(azPValue)==sizeof(azParam) );
  memset(&sRdr, 0, sizeof(sRdr));
  memset(azPValue, 0, sizeof(azPValue));
  for(i=3; i<argc; i++) {
    const char *z = argv[i];
    const char *zValue;
    for(j=0; j<sizeof(azParam)/sizeof(azParam[0]); j++){
      if(mddb_string_parameter(&sRdr, azParam[j], z, &azPValue[j])) break;
    }
    if(j<sizeof(azParam)/sizeof(azParam[0])) {
      if(sRdr.zErr[0])
        goto table_connect_error;
    }
    else {
      mddb_errmsg(&sRdr, "unrecognized parameter '%s'", z);
      goto table_connect_error;
    }
  }

  PLOG("creating new mddb instance (%s)", MDDB_PCI);
  mdptr = mddb_create_instance(MDDB_PCI,
                               strtol(MDDB_PARTITION,NULL,10),
                               "dwaddington"/* owner */,
                               1 /* io core */);
  assert(mdptr);
  
  *ppVtab = (sqlite3_vtab*) mdptr;
  PLOG("new mddb instance %p", *ppVtab);

  //  mddb_reader_reset(&sRdr);

  /* register schema */
  const char * schema = mddb_get_schema(mdptr);
  assert(schema);
  rc = sqlite3_declare_vtab(db, schema);
  if(rc) goto table_connect_error;


  /* all done, clean up */
  for(i=0; i<sizeof(azPValue)/sizeof(azPValue[0]); i++)
    sqlite3_free(azPValue[i]);

  mddb_check_canary(mdptr);

  return SQLITE_OK;

 table_connect_oom:
  rc = SQLITE_NOMEM;
  mddb_errmsg(&sRdr, "out of memory");

 table_connect_error:
  PERR("table connect error");
  if(mdptr)
    tableDisconnect((sqlite3_vtab *)mdptr);
  
  for(i=0; i<sizeof(azPValue)/sizeof(azPValue[0]); i++) {
    sqlite3_free(azPValue[i]);
  }
  
  if(sRdr.zErr[0]) {
    sqlite3_free(*pzErr);
    *pzErr = sqlite3_mprintf("%s", sRdr.zErr);
  }

  mddb_reader_reset(&sRdr);
  if(rc==SQLITE_OK) rc = SQLITE_ERROR;
  return rc;
}

/*
** Reset the current row content held by a MddbCursor.
*/
/* static void tableCursorRowReset(MddbCursor *pCur){ */
/*   Tablele *pTab = (Tablele*)pCur->base.pVtab; */
/*   int i; */
/*   for(i=0; i<pTab->nCol; i++){ */
/*     sqlite3_free(pCur->azVal[i]); */
/*     pCur->azVal[i] = 0; */
/*     pCur->aLen[i] = 0; */
/*   } */
/* } */

/*
** The xConnect and xCreate methods do the same thing, but they must be
** different so that the virtual table is not an eponymous virtual table.
*/
static int tableCreate(sqlite3 *db,
                       void *pAux,
                       int argc, const char *const*argv,
                       sqlite3_vtab **ppVtab,
                       char **pzErr) {
  TRACE();
  return tableConnect(db, pAux, argc, argv, ppVtab, pzErr);
}

/*
** Destructor for a MddbCursor.
*/
static int tableClose(sqlite3_vtab_cursor *cur){
  /* MddbCursor *pCur = (MddbCursor*)cur; */
  /* tableCursorRowReset(pCur); */
  /* mddb_reader_reset(&pCur->rdr); */
  /* sqlite3_free(cur); */
  return SQLITE_OK;
}

/*
** Constructor for a new Tablele cursor object.
*/
static int tableOpen(sqlite3_vtab *p, sqlite3_vtab_cursor **ppCursor)
{
  /* Tablele *pTab = (Tablele*)p; */
  /* MddbCursor *pCur; */
  /* size_t nByte; */

  PLOG("Opening...");
  
  /* nByte = sizeof(*pCur) + (sizeof(char*)+sizeof(int))*pTab->nCol; */
  /* pCur = sqlite3_malloc64( nByte ); */
  /* if( pCur==0 ) return SQLITE_NOMEM; */
  /* memset(pCur, 0, nByte); */
  /* pCur->azVal = (char**)&pCur[1]; */
  /* pCur->aLen = (int*)&pCur->azVal[pTab->nCol]; */
  /* *ppCursor = &pCur->base; */
  /* if( mddb_reader_open(&pCur->rdr, pTab->zFilename, pTab->zData) ){ */
  /*   mddb_xfer_error(pTab, &pCur->rdr); */
  /*   return SQLITE_ERROR; */
  /* } */
  return SQLITE_OK;
}


/*
** Advance a MddbCursor to its next row of input.
** Set the EOF marker if we reach the end of input.
*/
static int tableNext(sqlite3_vtab_cursor *cur)
{
  /* MddbCursor *pCur = (MddbCursor*)cur; */
  /* Tablele *pTab = (Tablele*)cur->pVtab; */
  /* int i = 0; */
  /* char *z; */
  /* do{ */
  /*   z = mddb_read_one_field(&pCur->rdr); */
  /*   if( z==0 ){ */
  /*     mddb_xfer_error(pTab, &pCur->rdr); */
  /*     break; */
  /*   } */
  /*   if( i<pTab->nCol ){ */
  /*     if( pCur->aLen[i] < pCur->rdr.n+1 ){ */
  /*       char *zNew = sqlite3_realloc64(pCur->azVal[i], pCur->rdr.n+1); */
  /*       if( zNew==0 ){ */
  /*         mddb_errmsg(&pCur->rdr, "out of memory"); */
  /*         mddb_xfer_error(pTab, &pCur->rdr); */
  /*         break; */
  /*       } */
  /*       pCur->azVal[i] = zNew; */
  /*       pCur->aLen[i] = pCur->rdr.n+1; */
  /*     } */
  /*     memcpy(pCur->azVal[i], z, pCur->rdr.n+1); */
  /*     i++; */
  /*   } */
  /* }while( pCur->rdr.cTerm==',' ); */
  /* if( z==0 || (pCur->rdr.cTerm==EOF && i<pTab->nCol) ){ */
  /*   pCur->iRowid = -1; */
  /* }else{ */
  /*   pCur->iRowid++; */
  /*   while( i<pTab->nCol ){ */
  /*     sqlite3_free(pCur->azVal[i]); */
  /*     pCur->azVal[i] = 0; */
  /*     pCur->aLen[i] = 0; */
  /*     i++; */
  /*   } */
  /* } */
  return SQLITE_OK;
}

/*
** Return values of columns for the row at which the MddbCursor
** is currently pointing.
*/
static int tableColumn(sqlite3_vtab_cursor *cur,   /* The cursor */
                       sqlite3_context *ctx,       /* First argument to sqlite3_result_...() */
                       int i) {                       /* Which column to return */
  /* MddbCursor *pCur = (MddbCursor*)cur; */
  /* Tablele *pTab = (Tablele*)cur->pVtab; */
  /* if( i>=0 && i<pTab->nCol && pCur->azVal[i]!=0 ){ */
  /*   sqlite3_result_text(ctx, pCur->azVal[i], -1, SQLITE_STATIC); */
  /* } */
  return SQLITE_OK;
}

/*
** Return the rowid for the current row.
*/
static int tableRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
  /* MddbCursor *pCur = (MddbCursor*)cur; */
  /* *pRowid = pCur->iRowid; */
  return SQLITE_OK;
}

/*
** Return TRUE if the cursor has been moved off of the last
** row of output.
*/
static int tableEof(sqlite3_vtab_cursor *cur){
  MddbCursor *pCur = (MddbCursor*)cur;
  return pCur->iRowid<0;
}

/*
** Only a full table scan is supported.  So xFilter simply rewinds to
** the beginning.
*/
static int tableFilter(
                        sqlite3_vtab_cursor *pVtabCursor, 
                        int idxNum, const char *idxStr,
                        int argc, sqlite3_value **argv
                        ){
  //  MddbCursor *pCur = (MddbCursor*)pVtabCursor;
  //  Tablele *pTab = (Tablele*)pVtabCursor->pVtab;
  /* pCur->iRowid = 0; */
  /* if( pCur->rdr.in==0 ){ */
  /*   assert( pCur->rdr.zIn==pTab->zData ); */
  /*   assert( pTab->iStart>=0 ); */
  /*   assert( (size_t)pTab->iStart<=pCur->rdr.nIn ); */
  /*   pCur->rdr.iIn = pTab->iStart; */
  /* }else{ */
  /*   fseek(pCur->rdr.in, pTab->iStart, SEEK_SET); */
  /*   pCur->rdr.iIn = 0; */
  /*   pCur->rdr.nIn = 0; */
  /* } */
  return tableNext(pVtabCursor);
}

/*
** Only a forward full table scan is supported.  xBestIndex is mostly
** a no-op.  If MDDBTEST_FIDX is set, then the presence of equality
** constraints lowers the estimated cost, which is fiction, but is useful
** for testing certain kinds of virtual table behavior.
*/
static int tableBestIndex(
                           sqlite3_vtab *tab,
                           sqlite3_index_info *pIdxInfo
                           ){
  pIdxInfo->estimatedCost = 1000000;
#ifdef SQLITE_TEST
  if( (((Tablele*)tab)->tstFlags & MDDBTEST_FIDX)!=0 ){
    /* The usual (and sensible) case is to always do a full table scan.
    ** The code in this branch only runs when testflags=1.  This code
    ** generates an artifical and unrealistic plan which is useful
    ** for testing virtual table logic but is not helpful to real applications.
    **
    ** Any ==, LIKE, or GLOB constraint is marked as usable by the virtual
    ** table (even though it is not) and the cost of running the virtual table
    ** is reduced from 1 million to just 10.  The constraints are *not* marked
    ** as omittable, however, so the query planner should still generate a
    ** plan that gives a correct answer, even if they plan is not optimal.
    */
    int i;
    int nConst = 0;
    for(i=0; i<pIdxInfo->nConstraint; i++){
      unsigned char op;
      if( pIdxInfo->aConstraint[i].usable==0 ) continue;
      op = pIdxInfo->aConstraint[i].op;
      if( op==SQLITE_INDEX_CONSTRAINT_EQ 
          || op==SQLITE_INDEX_CONSTRAINT_LIKE
          || op==SQLITE_INDEX_CONSTRAINT_GLOB
          ){
        pIdxInfo->estimatedCost = 10;
        pIdxInfo->aConstraintUsage[nConst].argvIndex = nConst+1;
        nConst++;
      }
    }
  }
#endif
  return SQLITE_OK;
}


static sqlite3_module MddbModule = {
  0,                       /* iVersion */
  tableCreate,            /* xCreate */
  tableConnect,           /* xConnect */
  tableBestIndex,         /* xBestIndex */
  tableDisconnect,        /* xDisconnect */
  tableDisconnect,        /* xDestroy */
  tableOpen,              /* xOpen - open a cursor */
  tableClose,             /* xClose - close a cursor */
  tableFilter,            /* xFilter - configure scan constraints */
  tableNext,              /* xNext - advance a cursor */
  tableEof,               /* xEof - check for end of scan */
  tableColumn,            /* xColumn - read data */
  tableRowid,             /* xRowid - read data */
  0,                       /* xUpdate */
  0,                       /* xBegin */
  0,                       /* xSync */
  0,                       /* xCommit */
  0,                       /* xRollback */
  0,                       /* xFindMethod */
  0,                       /* xRename */
};

#ifdef SQLITE_TEST
/*
** For virtual table testing, make a version of the MDDB virtual table
** available that has an xUpdate function.  But the xUpdate always returns
** SQLITE_READONLY since the MDDB file is not really writable.
*/
static int tableUpdate(sqlite3_vtab *p,int n,sqlite3_value**v,sqlite3_int64*x){
  return SQLITE_READONLY;
}


static sqlite3_module MddbModuleFauxWrite = {
  0,                       /* iVersion */
  tableCreate,            /* xCreate */
  tableConnect,           /* xConnect */
  tableBestIndex,         /* xBestIndex */
  tableDisconnect,        /* xDisconnect */
  tableDisconnect,        /* xDestroy */
  tableOpen,              /* xOpen - open a cursor */
  tableClose,             /* xClose - close a cursor */
  tableFilter,            /* xFilter - configure scan constraints */
  tableNext,              /* xNext - advance a cursor */
  tableEof,               /* xEof - check for end of scan */
  tableColumn,            /* xColumn - read data */
  tableRowid,             /* xRowid - read data */
  tableUpdate,            /* xUpdate */
  0,                       /* xBegin */
  0,                       /* xSync */
  0,                       /* xCommit */
  0,                       /* xRollback */
  0,                       /* xFindMethod */
  0,                       /* xRename */
};
#endif /* SQLITE_TEST */

#endif /* !defined(SQLITE_OMIT_VIRTUALTABLE) */



/* 
** This routine is called when the extension is loaded.  The new
** MDDB virtual table module is registered with the calling database
** connection.
*/
int sqlite3_mddb_init(sqlite3 *db, 
                     char **pzErrMsg, 
                     const sqlite3_api_routines *pApi)
{
#ifndef SQLITE_OMIT_VIRTUALTABLE	
  int rc;
  SQLITE_EXTENSION_INIT2(pApi);
  rc = sqlite3_create_module(db, "mddb", &MddbModule, 0);
#ifdef SQLITE_TEST
  if( rc==SQLITE_OK ){
    rc = sqlite3_create_module(db, "mddb_wr", &MddbModuleFauxWrite, 0);
  }
#endif
  return rc;
#else
  return SQLITE_OK;
#endif
}
