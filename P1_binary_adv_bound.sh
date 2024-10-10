#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

ab_main="$HOME/ROSAR-framework/alpha-beta-CROWN/complete_verifier"
ab_crown_vnnlib="$ab_main/models/yolox/vnnlib"
ab_crown_instances="$ab_main/models/yolox/instances.csv"

prop_main="$HOME/ROSAR-framework"
prop_stat="$prop_main/statistics_`date +%s`.csv"

echo "AB-Crown main:       $ab_main"
echo "AB-Crown vnnlib:     $ab_crown_vnnlib"
echo "AB-Crown instances:  $ab_crown_instances"

echo "Properties main:     $prop_main"
echo "Properties stats:    $prop_stat"

initial_low=0.00
initial_high=0.16
time_limit="2m"
max_iterations=15

touch $prop_stat

for input_img in ./data/*.png; do
    echo $input_img

    get_num_bbox="python get_num_bbox.py $input_img | tail -n 1"
    num_bboxes=$(eval $get_num_bbox)
    num_bboxes=$((num_bboxes))

    echo "Total number of bboxes on image: $num_bboxes"
    for ((i = 0; i < num_bboxes; i++)); do
        echo "Attacking bounding box index: $i"

        low=$initial_low
        high=$initial_high
        step_factor=3.0
        found_adversarial=0
        # echo $low $high $step_factor $found_adversarial

        for iteration in $(seq 1 $max_iterations); do
            if [ $iteration -le $((max_iterations / 2)) ]; then
                step=$(echo "scale=10; ($high - $low) / $step_factor" | bc)
                mid=$(echo "scale=10; $low + $step" | bc)
            else
                mid=$(echo "scale=10; ($low + $high) / 2" | bc)
            fi

            delta=$(printf "%.5f" $mid)

            echo "generate_P1.py $input_img $i $delta" 

            python generate_P1.py $input_img $i $delta > /dev/null 2>&1

            for vnn_file in ./vnnlib/*.vnnlib; do
                rm -f "$ab_crown_vnnlib"/*
                cp $vnn_file $ab_crown_vnnlib

                rm $ab_crown_instances
                echo "onnx/KD_yolox_nano_L_ViT.onnx,$vnn_file,125" > $ab_crown_instances 

                ls $ab_crown_vnnlib
                cat $ab_crown_instances

                cd $ab_main

                if timeout "$time_limit" python abcrown.py --config exp_configs/yolox/yolox.yaml --device cpu --save_adv_example > /dev/null 2>&1; then
                    echo "Adversarial example found for image $input_img delta $delta"
                    echo "FOUND,$input_img,$delta,$vnn_file" >> $prop_stat
                    high=$mid
                    if [ $iteration -le $(($max_iterations / 2)) ]; then
                        step_factor=$(echo "scale=10; ($step_factor + 2.0) / 2" | bc)  # Reduce step factor for finer adjustments
                    fi
                    found_adversarial=1
                else
                    echo "Could not find AE"
                    echo "NOT_FOUND,$input_img,$delta,$vnn_file" >> $prop_stat
                    low=$mid
                    if [ $(echo "$high == $initial_high" | bc) -eq 1 ]; then
                        high=$(echo "$high * 2" | bc)  # Double the high value to expand the search range
                    fi
                fi

                cd $prop_main
            done
            
            # Break if adversarial example is found in the last iteration
            if [ $found_adversarial -eq 1 ] && [ $iteration -eq $max_iterations ]; then
                break
            fi

        done

        final_bound=$(printf "%.2f" $high)
        echo "Final estimated minimal adversarial bound for image $input_img: $final_bound%"

    done

done
