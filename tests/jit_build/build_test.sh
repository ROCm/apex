#parse the arguments
JIT_CONDITION="$2"

sh tests/jit_build/build.sh "condition" $JIT_CONDITION
sh tests/jit_build/run_tests.sh "condition" $JIT_CONDITION