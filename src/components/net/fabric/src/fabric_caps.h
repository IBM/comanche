/* Primary capabilities */

X(FI_MSG) /* sending and receiving messages or datagrams */
X(FI_RMA) /* RMA read and write operations */
X(FI_TAGGED) /* tagged message transfers */
X(FI_ATOMIC) /* some set of atomic operations */
X(FI_MULTICAST) /* multicast data transfers */
X(FI_NAMED_RX_CTX) /* allow an initiator to target a specific receive context */
X(FI_DIRECTED_RECV) /* use the source address of an incoming message when matching it with a receive buffer */
X(FI_READ) /*  capable of initiating reads against remote memory regions */
X(FI_WRITE) /* capable of initiating writes against remote memory regions */
X(FI_SEND) /* capable of sending message data transfers */
X(FI_RECV) /* capable of receiving message data transfers */
X(FI_REMOTE_READ) /* capable of receiving read memory operations from remote endpoints */
X(FI_REMOTE_WRITE) /* capable of receiving write memory operations from remote endpoints */
#if 0
X(FI_VARIABLE_MSG) /* in doc but not header */
#endif

/* Secondary capabilities */

X(FI_MULTI_RECV) /* must support the FI_MULTI_RECV flag */
X(FI_SOURCE) /* return source addressing data as part of its completion data */
X(FI_RMA_EVENT) /* support the generation of completion events when it is the target of an RMA or atomic operation */
X(FI_SHARED_AV) /* support for shared address vectors */
X(FI_TRIGGER) /* support triggered operations */
X(FI_FENCE) /* support the FI_FENCE flag */
X(FI_LOCAL_COMM) /* support host local communication */
X(FI_REMOTE_COMM) /* support remote communication */
X(FI_SOURCE_ERR) /* raw source addressing data be returned as part of completion data for any addressing error */
X(FI_RMA_PMEM) /* provider is 'persistent memory aware' */
