#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <rte_mbuf.h>
#include <rte_compressdev.h>

#define COMPRESS_BUF_SIZE 4096

int compress_mbuf(struct rte_mbuf *mbuf, uint16_t compress_dev_id, uint16_t compress_qp_id)
{
    struct rte_comp_op *op;
    struct rte_comp_compress_xform comp_xform;
    struct rte_comp_op **ops;
    int ret, i;

    // Create compression operation
    ops = malloc(sizeof(struct rte_comp_op *));
    if (ops == NULL) {
        printf("Failed to allocate memory for compression operation.\n");
        return -1;
    }

    op = rte_comp_op_alloc(NULL);
    if (op == NULL) {
        printf("Failed to allocate compression operation.\n");
        free(ops);
        return -1;
    }

    // Set compression operation parameters
    op->op_type = RTE_COMP_OP_STATELESS;
    op->compress.level = RTE_COMP_LEVEL_DEFAULT;
    op->m_src = mbuf;
    op->m_dst = rte_pktmbuf_alloc(rte_pktmbuf_pool(mbuf));

    // Set compression transformation
    comp_xform.type = RTE_COMP_COMPRESS;
    comp_xform.algo = RTE_COMP_ALGO_DEFLATE;
    comp_xform.compress.level = RTE_COMP_LEVEL_DEFAULT;
    comp_xform.window_size = 15;

    op->xform = &comp_xform;
    ops[0] = op;

    // Submit compression operation to the compress device
    ret = rte_compressdev_enqueue_burst(compress_dev_id, compress_qp_id, ops, 1);
    if (ret < 0) {
        printf("Failed to enqueue compression operation.\n");
        rte_pktmbuf_free(op->m_dst);
        rte_comp_op_free(op);
        free(ops);
        return -1;
    }

    // Wait for completion of compression operation
    ret = rte_compressdev_dequeue_burst(compress_dev_id, compress_qp_id, ops, 1);
    if (ret < 0) {
        printf("Failed to dequeue compression operation.\n");
        rte_pktmbuf_free(op->m_dst);
        rte_comp_op_free(op);
        free(ops);
        return -1;
    }

    // Process the compressed data (op->m_dst)

    // Cleanup
    rte_pktmbuf_free(op->m_dst);
    rte_comp_op_free(op);
    free(ops);

    return 0;
}

int main()
{
    // Initialize DPDK and compress device
    // ...

    struct rte_mbuf *mbuf;
    uint16_t compress_dev_id = 0;  // Compress device ID
    uint16_t compress_qp_id = 0;   // Compress queue pair ID

    // Create and populate the mbuf with data
    // ...

    // Compress the mbuf using rte_compress
    if (compress_mbuf(mbuf, compress_dev_id, compress_qp_id) < 0) {
        printf("Failed to compress mbuf.\n");
        // Handle compression failure
        return 1;
    }

    // Further processing or transmission of the compressed mbuf

    return 0;
}
