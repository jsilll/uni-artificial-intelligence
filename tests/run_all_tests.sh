echo "PROGRAMA FANTASTICO DO JONECA"
for input_file in tests/input*.txt; do
    time=`(time (python3 src/numbrix.py $input_file > /dev/null)) 2>&1 | grep real`
    echo `basename $input_file` $time
done

echo "PROGRAMAZORD VASKOZORD"
for input_file in tests/input*.txt; do
    time=`(time (python3 src/numbrix_vasko.py $input_file > /dev/null)) 2>&1 | grep real`
    echo `basename $input_file` $time
done