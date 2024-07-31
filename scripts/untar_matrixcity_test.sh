for num in {1..10}
do  
mkdir block_${num}_test/input
tar -xvf block_${num}_test.tar
mv block_${num}_test/*.png block_${num}_test/input
done 