import sys 

test_file = sys.argv[1]

#read lines from test file
with open(test_file, "r") as f:
    lines = f.readlines()

failed_tests = []
for line in lines:
    if "ERROR: " in line:
        failed_tests.append(line[7:].strip())
    if " FAILED" in line:
        failed_tests.append(line[: -6].strip())
print(len(failed_tests))
#print(str(len(failed_tests)) + "," + ";".join(failed_tests)) 