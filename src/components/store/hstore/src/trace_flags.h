#ifndef _DAWN_HSTORE_TRACE_FLAGS_H
#define _DAWN_HSTORE_TRACE_FLAGS_H

/* things to report */
#define TRACE_MANY 0
#define TRACE_PALLOC 0
#define TRACE_LOCK 0
#define TRACE_PERISHABLE_EXPIRY 0

#define TRACE_TABLE (TRACE_MANY || TRACE_PERISHABLE_EXPIRY)
#define TRACE_BUCKET TRACE_TABLE
#define TRACE_CONTENT TRACE_TABLE
#define TRACE_OWNER TRACE_TABLE

/* Data to track which is not normally needed but us required by some TRACE */
#define TRACK_OWNER (TRACE_OWNER || TRACE_CONTENT)
#define TRACK_POS TRACE_OWNER

#endif
