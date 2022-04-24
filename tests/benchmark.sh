benchmark() {
    algo_name=$1
    
    echo "Test File Name, Nodes Expanded, Nodes In The Frontier" >> "docs/results/${algo_name}_nodes.csv"
    for input_file in tests/input*.txt; do
        display_line=`python3 src/numbrix.py $input_file --display --$algo_name | head -n 1`
        expanded=`echo "$display_line" | awk '{print $1}'`
        in_frontier=`echo "$display_line" | awk '{print $7}'`
        bname=`basename $input_file`
        echo $bname,$expanded,$in_frontier >> "docs/results/${algo_name}_nodes.csv"
    done
    
    echo "Test File Name, Execution Time (s)" >> "docs/results/${algo_name}_times.csv"
    for input_file in tests/input*.txt; do
        time=`(time (python3 src/numbrix.py $input_file  --$algo_name > /dev/null)) 2>&1 | grep real | awk '{print $2}'`
        bname=`basename $input_file`
        echo $bname,$time >> "docs/results/${algo_name}_times.csv"
    done

}

# Overall Comparison
benchmark "dfs"
benchmark "bfs"
benchmark "astar"
benchmark "greedy"

# TODO: Compare Better Algos in Bigger Inputs 
# TODO: Compare Heuristics (fitting parameter)