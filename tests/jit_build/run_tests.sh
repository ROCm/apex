#parse the arguments
JIT_CONDITION="$2"
echo "JIT_CONDITION $JIT_CONDITION"

#run the apex unit tests
LOG_FILE=results_jit_unit_test${JIT_CONDITION}.log
LOG_FILE2=results_jit_unit_test${JIT_CONDITION}c.log

cd tests/L0 
PYTHONUNBUFFERED=1 sh run_rocm.sh 2>&1 | tee ../../$LOG_FILE
cd ../../

cd apex/contrib/test
PYTHONUNBUFFERED=1 python run_rocm_extensions.py 2>&1 | tee ../../../$LOG_FILE2
cd ../../../

torchrun --nproc_per_node 8 apex/contrib/peer_memory/peer_halo_exchange_module_tests.py 2>&1 | tee -a $LOG_FILE2

cd tests/distributed/synced_batchnorm
sh unit_test.sh 2>&1 | tee -a ../../../$LOG_FILE2
cd ../../../

#explicitly load the builder and build the remaining extensions
python tests/jit_build/load_extra_extensions.py 2>&1 | tee $LOG_FILE

FAILED_TESTS=$(python tests/jit_build/count_failed_unit_tests.py $LOG_FILE)
FAILED_TESTS2=$(python tests/jit_build/count_failed_unit_tests.py $LOG_FILE2)
BUILT_SO_COUNT=$(python tests/jit_build/count_built_so.py)
TORCH_EXTENSIONS_COUNT=$(python tests/jit_build/count_torch_extensions.py)

echo "Failed L0 tests = $FAILED_TESTS"
echo "Failed contrib tests = $FAILED_TESTS2"
echo ".so count = $BUILT_SO_COUNT"
echo "JIT torch extensions count = $TORCH_EXTENSIONS_COUNT"

echo "$FAILED_TESTS $FAILED_TESTS2 $BUILT_SO_COUNT $TORCH_EXTENSIONS_COUNT"