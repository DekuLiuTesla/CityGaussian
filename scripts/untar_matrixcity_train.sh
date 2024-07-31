for num in {1..10}
do  
mkdir block_$num/input
tar -xvf block_$num.tar
mv block_$num/*.png block_$num/input
done 