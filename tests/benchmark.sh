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

benchmark_big() {
    algo_name=$1

    echo "Test File Name, Nodes Expanded, Nodes In The Frontier" >> "docs/results/${algo_name}_big_nodes.csv"
    for input_file in tests/big*.txt; do
        display_line=`python3 src/numbrix.py $input_file --display --$algo_name | head -n 1`
        expanded=`echo "$display_line" | awk '{print $1}'`
        in_frontier=`echo "$display_line" | awk '{print $7}'`
        bname=`basename $input_file`
        echo $bname,$expanded,$in_frontier >> "docs/results/${algo_name}_big_nodes.csv"
    done
    
    echo "Test File Name, Execution Time (s)" >> "docs/results/${algo_name}_big_times.csv"
    for input_file in tests/big*.txt; do
        time=`(time (python3 src/numbrix.py $input_file  --$algo_name > /dev/null)) 2>&1 | grep real | awk '{print $2}'`
        bname=`basename $input_file`
        echo $bname,$time >> "docs/results/${algo_name}_big_times.csv"
    done
}

benchmark_heuristic() {
    algo_name=$1
    alpha=$2

    echo "Test File Name, Alpha, Nodes Expanded, Nodes In The Frontier" > "docs/results/alpha_nodes.csv"
    for alpha in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}; do
        display_line=`python3 src/numbrix.py tests/big1.txt --display --$algo_name --alpha $alpha | head -n 1`
        expanded=`echo "$display_line" | awk '{print $1}'`
        in_frontier=`echo "$display_line" | awk '{print $7}'`
        bname=`basename tests/big1.txt`
        echo $bname,$alpha,$expanded,$in_frontier >> "docs/results/alpha_nodes.csv"
    done
    
    echo "Test File Name, Alpha, Execution Time (s)" > "docs/results/alpha_times.csv"
    for alpha in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}; do
        time=`(time (python3 src/numbrix.py tests/big1.txt  --$algo_name --alpha $alpha > /dev/null)) 2>&1 | grep real | awk '{print $2}'`
        bname=`basename tests/big1.txt`
        echo $bname,$alpha,$time >> "docs/results/alpha_times.csv"
    done
}

# Overall Comparison
# benchmark "dfs"
# benchmark "bfs"
# benchmark "astar"
# benchmark "greedy"

# Compare Better Algos in Bigger Inputs 
# benchmark_big "dfs"
# benchmark_big "greedy"

# Compare Heuristics
benchmark_heuristic "greedy"