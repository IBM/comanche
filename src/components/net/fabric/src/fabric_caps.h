/* Primary capabilities */

X(FI_MSG)
/*
 * SPECIFIES that an endpoint should support sending and receiving messages or datagrams.1s
 * IMPLIES support for send and/or receive queues.
 * Endpoints supporting this capability support operations defined by struct fi_ops_msg.
*/

X(FI_RMA)
/* SPECIFIES that the endpoint should support RMA read and write operations.
 * Endpoints supporting this capability support operations defined by struct fi_ops_rma.1s
 * [In the absence of any relevant flags] IMPLIES the ability to initiate and be the target of remote memory reads and writes.
 * NOTE: Applications can use the FI_READ, FI_WRITE, FI_REMOTE_READ, and FI_REMOTE_WRITE flags to restrict the types of RMA operations supported by an endpoint.
*/

X(FI_TAGGED)
/* SPECIFIES that the endpoint should handle tagged message transfers.
 * NOTE: Tagged message transfers associate a user-specified key or tag with each message that is used for matching purposes at the remote side.
 * Endpoints supporting this capability support operations defined by struct fi_ops_tagged.
 * [In the absence of any relevant flags], IMPLIES the ability to send and receive tagged messages.
 * NOTE: Applications can use the FI_SEND and FI_RECV flags to optimize an endpoint as send-only or receive-only.
*/

X(FI_ATOMIC)
/* SPECIFIES that the endpoint supports some set of atomic operations.
 * Endpoints supporting this capability support operations defined by struct fi_ops_atomic.
*/

X(FI_MULTICAST)
/* INDICATES that the endpoint support multicast data transfers.
 * MUST be paired with at least one other data transfer capability, (e.g. FI_MSG, FI_SEND, FI_RECV, ...).
*/

X(FI_NAMED_RX_CTX)
/*  REQUESTS that endpoints which support multiple receive contexts allow an initiator to target (or name) a specific receive context as part of a data transfer operation.
*/

X(FI_DIRECTED_RECV)
/* REQUESTS that the communication endpoint use the source address of an incoming message when matching it with a receive buffer.
 * NOTE: If this capability is not set, then the src_addr parameter for msg and tagged receive operations is ignored.
*/

X(FI_READ)
/* INDICATES that the user requires an endpoint capable of initiating reads against remote memory regions.
 * MUST be paired with FI_RMA and/or FI_ATOMIC.
*/

X(FI_WRITE)
/* INDICATES that the user requires an endpoint capable of initiating writes against remote memory regions.
   MUST be paired with FI_RMA and/or FI_ATOMIC.
*/

X(FI_SEND)
/* INDICATES that the user requires an endpoint capable of sending message data transfers.
 * NOTE: Message transfers include base message operations as
    well as tagged message functionality.
*/

X(FI_RECV)
/*  INDICATES that the user requires an endpoint capable of receiving message data transfers.
 * NOTE: Message transfers include base message operations as
    well as tagged message functionality.
*/

X(FI_REMOTE_READ)
/* INDICATES that the user requires an endpoint capable of receiving read memory operations from remote endpoints.
 * MUST be paired with FI_RMA and/or FI_ATOMIC.
*/

X(FI_REMOTE_WRITE)
/*  INDICATES that the user requires an endpoint capable of receiving write memory operations from remote endpoints. This flag requires
    that FI_RMA and/or FI_ATOMIC be set.
*/

/* (not defined in pur copy of the include files) */
#if 0
X(FI_VARIABLE_MSG)
/* 
 * REQUESTS that the provider must notify a receiver when a variable length message is ready to be received prior to attempting to place the data.
 * NOTE: Such notification will include the size of the message and any associated message tag (for FI_TAGGED).
 * NOTE: Variable length messages are any messages larger than an endpoint configurable size.1s
 * MUST be paired with FI_MSG and/or FI_TAGGED.
*/
#endif

/*
    Secondary capabilities: FI_MULTI_RECV, FI_SOURCE, FI_RMA_EVENT, FI_SHARED_AV, FI_TRIGGER, FI_FENCE, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_SOURCE_ERR,
    FI_RMA_PMEM.
*/

X(FI_MULTI_RECV)
/* SPECIFIES that the endpoint must support the FI_MULTI_RECV flag when posting receive buffers.
*/

X(FI_SOURCE)
/*  REQUESTS that the endpoint return source addressing data as part of its completion data.
*/

X(FI_RMA_EVENT)
/* REQUESTS that an endpoint support the generation of completion events when it is the target of an RMA and/or atomic operation.
 * REQUIRES that FI_REMOTE_READ and/or FI_REMOTE_WRITE be enabled on the endpoint.
*/

X(FI_SHARED_AV)
/* REQUESTS or INDICATES support for address vectors which may be shared among multiple processes.
*/

X(FI_TRIGGER)
/* INDICATES that the endpoint should support triggered operations.
*/

X(FI_FENCE)
/* INDICATES that the endpoint support the FI_FENCE flag on data transfer operations.
*/

X(FI_LOCAL_COMM)
/* INDICATES that the endpoint support host local communication.
 * MAY be used in conjunction with FI_REMOTE_COMM to indicate that local and remote communication are required.
 * NOTE: If neither FI_LOCAL_COMM or FI_REMOTE_COMM are specified, then the provider will indicate support for the configuration that minimally affects performance.
 * NOTE: Providers that set FI_LOCAL_COMM but not FI_REMOTE_COMM, for example a shared memory provider, may only be used to communication between processes on the same system.
*/

X(FI_REMOTE_COMM)
/*  INDICATES that the endpoint support communication with endpoints located at remote nodes (across the fabric).
    Providers that set FI_REMOTE_COMM but not FI_LOCAL_COMM, for example NICs that lack loopback support, cannot be used to communicate with processes on the same system.
*/

X(FI_SOURCE_ERR)
/*  REQUESTS that raw source addressing data be returned as part of completion data for any address that has not been inserted into the local address vector.
 *  MUST be paired with FI_SOURCE.
*/

X(FI_RMA_PMEM)
/* INDICATES that the provider is 'persistent memory aware' and supports RMA operations to and from persistent memory.
   REQUIRES that FI_RMA be set.
*/
