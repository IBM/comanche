/*
   Copyright [2018] [IBM Corporation]

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

#ifndef _FABRIC_CONNECTION_SERVER_H_
#define _FABRIC_CONNECTION_SERVER_H_

#include "fabric_connection.h"

class Fd_control;
struct fi_info;
struct fid_fabric;
struct fid_eq;

class Fabric_connection_server
  : public Fabric_connection
{
public:
  Fabric_connection_server(fid_fabric &fabric, fid_eq &, fi_info & info, Fd_control &&conn_fd);
};

#endif
