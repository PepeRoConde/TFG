set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track parameters
declare -a MAIN_FLAGS=()
BMIN=32
BMAX=512
RUN_TRAINING=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -bmin)
            BMIN="$2"
            shift 2
            ;;
        -bmax)
            BMAX="$2"
            shift 2
            ;;
        -run)
            RUN_TRAINING=true
            shift
            ;;
        -verbose)
            VERBOSE=true
            shift
            ;;
        *)
            # Pass everything else to main.py
            MAIN_FLAGS+=("$1")
            shift
            ;;
    esac
done

echo -e "${YELLOW}=== Batch Size Binary Search ===${NC}"
echo "Search range: $BMIN - $BMAX"
echo "Main flags: ${MAIN_FLAGS[@]}"
echo ""

# Function to test a batch size
test_batch_size() {
    local bsize=$1
    local test_output=$(mktemp)
    trap "rm -f $test_output" RETURN

    if [[ "$VERBOSE" == true ]]; then
        echo -ne "Testing batch size: $bsize ... "
    fi

    # Run with epochs=1 to quickly test memory allocation
    if python -m src.main "${MAIN_FLAGS[@]}" \
        --batch_size "$bsize" \
        --epochs 1 \
        --paciencia 1 \
        > "$test_output" 2>&1; then

        if [[ "$VERBOSE" == true ]]; then
            echo -e "${GREEN}✓ OK${NC}"
        else
            echo -ne "."
        fi
        return 0
    else
        # Check if it's an OOM error
        if grep -q "CUDA out of memory" "$test_output" || \
           grep -q "RuntimeError" "$test_output" || \
           grep -q "OutOfMemory" "$test_output"; then

            if [[ "$VERBOSE" == true ]]; then
                echo -e "${RED}✗ OOM${NC}"
            else
                echo -ne "X"
            fi
        else
            if [[ "$VERBOSE" == true ]]; then
                echo -e "${RED}✗ FAILED${NC}"
                head -20 "$test_output" | sed 's/^/  /'
            else
                echo -ne "F"
            fi
        fi
        return 1
    fi
}

# Binary search for maximum batch size
low=$BMIN
high=$BMAX
max_batch_size=$BMIN
step_count=0

echo -ne "${YELLOW}Searching${NC}: "

while [ $low -le $high ]; do
    mid=$(( (low + high) / 2 ))
    step_count=$((step_count + 1))

    if test_batch_size $mid; then
        # Success: this size works, try higher
        max_batch_size=$mid
        low=$((mid + 1))
    else
        # Failure: this size causes OOM, try lower
        high=$((mid - 1))
    fi
done

echo ""
echo ""
echo -e "${GREEN}=== Result ===${NC}"
echo "Search steps: $step_count"
echo -e "${GREEN}Maximum batch size found: ${max_batch_size}${NC}"
echo ""

# If -run flag provided, run full training with optimal batch size
if [[ "$RUN_TRAINING" == true ]]; then
    echo -e "${YELLOW}=== Starting full training with batch size: $max_batch_size ===${NC}"
    echo ""
    python -m src.main "${MAIN_FLAGS[@]}" \
        -b "$max_batch_size"

    echo ""
    echo -e "${GREEN}=== Training complete ===${NC}"
else
    echo -e "${YELLOW}To run full training with this batch size, use:${NC}"
    echo ""
    echo "python -m src.main ${MAIN_FLAGS[@]} --batch-size $max_batch_size"
    echo ""
    echo -e "${YELLOW}Or run this script with -run flag:${NC}"
    echo "$(basename $0) ${@//-run/} -run"
fi
