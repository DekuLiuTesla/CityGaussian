# Aerial
for num in {1..10}
do  
mkdir block_${num}_test/input
tar -xvf block_${num}_test.tar
mv block_${num}_test/*.png block_${num}_test/input
done

# Street
mkdir small_city_road_down_test/input
tar -xvf small_city_road_down_test.tar
mv small_city_road_down_test/*.png small_city_road_down_test/input

mkdir small_city_road_horizon_test/input
tar -xvf small_city_road_horizon_test.tar
mv small_city_road_horizon_test/*.png small_city_road_horizon_test/input

mkdir small_city_road_outside_test/input
tar -xvf small_city_road_outside_test.tar
mv small_city_road_outside_test/*.png small_city_road_outside_test/input

mkdir small_city_road_vertical_test/input
tar -xvf small_city_road_vertical_test.tar
mv small_city_road_vertical_test/*.png small_city_road_vertical_test/input