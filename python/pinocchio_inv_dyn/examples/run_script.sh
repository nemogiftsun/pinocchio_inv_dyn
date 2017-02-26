#!/bin/bash
cd python
for i in `seq 100`;
   do
   echo "
         /////////////////////////////////////////////
         //  Executing test_reaching, test number $i
         /////////////////////////////////////////////
"   
   python example_hrp2_ng.py reach1taskwrl >> _reaching_task_1_log_$i.txt
done
cd ..
echo "
        Well, destroy those germs!"
