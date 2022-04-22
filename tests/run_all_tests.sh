for input_file in tests/input*.txt; do
    time=`(time (python3 src/numbrix.py $input_file > /dev/null)) 2>&1 | grep real`
    echo `basename $input_file` $time
done

for input_file in tests/input*.txt; do
    bname=`basename $input_file`
    nodes=`python3 src/numbrix.py $input_file | grep _pickle.loads | awk '{print $1}'`
    printf "%s: %d nodes\n" $bname $nodes 
done