#parse the arguments
JIT_CONDITION="$2"

echo $(pwd)

WORKSPACE_DIR=/myworkspace
mkdir -p $WORKSPACE_DIR 

cd $WORKSPACE_DIR  
git clone https://github.com/rocm/apex.git --recursive 
cd apex 
git checkout Refactor_build
git submodule update --init --recursive 

sh tests/jit_build/build.sh "condition" $JIT_CONDITION

# Capture the output from run_tests.sh
TEST_RESULTS=$(sh tests/jit_build/run_tests.sh "condition" $JIT_CONDITION | tail -1)

# Parse the returned values
read FAILED_TESTS FAILED_TESTS2 BUILT_SO_COUNT TORCH_EXTENSIONS_COUNT <<< "$TEST_RESULTS"

MULTIPLE_RESULTS_FILE="../results_jit_unit_test.csv"
#echo "condition,failed unit tests" > "$MULTIPLE_RESULTS_FILE"
echo "$JIT_CONDITION,$FAILED_TESTS,$FAILED_TESTS2,$BUILT_SO_COUNT,$TORCH_EXTENSIONS_COUNT" >> "$MULTIPLE_RESULTS_FILE"