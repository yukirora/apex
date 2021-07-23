python scripts/amd/apex_gbn_add_relu_test.py

# gdb -ex "set breakpoint pending on" \
#     -ex 'break NhwcBatchNorm::fwd' \
#     -ex 'run' \
#     --args python scripts/amd/apex_gbn_add_relu_test.py

# python -m pdb \
#     -c "b BatchNorm2d_NHWC.forward" \
#     -c c scripts/amd/apex_gbn_add_relu_test.py

# python -m pdb \
#     -c c scripts/amd/apex_gbn_add_relu_test.py
