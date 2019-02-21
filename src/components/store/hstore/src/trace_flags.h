/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_TRACE_FLAGS_H
#define _COMANCHE_HSTORE_TRACE_FLAGS_H

/* things to report */
#define TRACE_MANY 0
#define TRACE_PALLOC 0
#define TRACE_PERSIST 0
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
