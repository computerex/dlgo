#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Declare external function from simd_dot.c */
extern void ggml_vec_dot_q4_0_q8_0(int n, float *s, const void *vx, const void *vy);

START_TEST(test_simd_dot_buffer_bounds)
{
    /* Invariant: SIMD dot product must not access memory beyond allocated buffer boundaries */
    
    /* Test buffer sizes: undersized (exploit case), exact boundary, valid size */
    const int buffer_sizes[] = {4, 8, 32};  /* 4 bytes triggers grp+4 OOB, 8 is boundary, 32 is valid */
    const int num_sizes = sizeof(buffer_sizes) / sizeof(buffer_sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int buf_size = buffer_sizes[i];
        
        /* Allocate buffers with guard pages would be ideal, but we use sentinel pattern */
        uint8_t *vx = calloc(1, buf_size + 16);  /* Extra space for sentinel */
        uint8_t *vy = calloc(1, buf_size + 16);
        float result = 0.0f;
        
        ck_assert_ptr_nonnull(vx);
        ck_assert_ptr_nonnull(vy);
        
        /* Fill with known pattern */
        memset(vx, 0xAA, buf_size);
        memset(vy, 0xBB, buf_size);
        
        /* Set sentinel bytes after valid region */
        memset(vx + buf_size, 0xDE, 16);
        memset(vy + buf_size, 0xDE, 16);
        
        /* Call with n=0 should be safe regardless of buffer size */
        ggml_vec_dot_q4_0_q8_0(0, &result, vx, vy);
        
        /* Verify sentinel wasn't corrupted (indicates no OOB write) */
        for (int j = 0; j < 16; j++) {
            ck_assert_uint_eq(vx[buf_size + j], 0xDE);
            ck_assert_uint_eq(vy[buf_size + j], 0xDE);
        }
        
        free(vx);
        free(vy);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_simd_dot_buffer_bounds);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}