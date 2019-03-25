/*
   Copyright [2017-2019] [IBM Corporation]
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
