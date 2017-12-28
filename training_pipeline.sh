#!/bin/bash
# if  [ $1 == 'upload-full' ]
# then
#     zip -qr records.zip records
#     aws s3 cp records.zip s3://yang-carnd/records.zip
#     aws s3 cp model.py s3://yang-carnd/model.py
# fi

# if  [ $1 == 'upload' ]
# then
#     aws s3 cp model.py s3://yang-carnd/model.py
# fi



# if  [ $1 == 'train-full' ]
# then
#     ssh carnd@$CLOUD_URL "
#     rm -rf records
#     rm records.zip
#     aws s3 cp s3://yang-carnd/records.zip records.zip
#     unzip records.zip
#     aws s3 cp s3://yang-carnd/model.py model.py
#     source ~/anaconda3/bin/activate carnd-term1
#     python model.py
#     aws s3 cp model.h5 s3://yang-carnd/model.h5
#     # shutdown
#     "
#     aws s3 cp s3://yang-carnd/model.h5 model.h5
# fi

# if [ $1 == 'train' ]
# then
#     ssh carnd@$CLOUD_URL "
#     aws s3 cp s3://yang-carnd/model.py model.py
#     source ~/anaconda3/bin/activate carnd-term1
#     python model.py
#     aws s3 cp model.h5 s3://yang-carnd/model.h5
#     # shutdown
#     "
#     aws s3 cp s3://yang-carnd/model.h5 model.h5
# fi

if [ $1 == 'sync' ]
then
    ls -l records/IMG | wc -l
    scp records/driving_log.csv carnd@$CLOUD_URL:~/records/driving_log.csv
    scp model.py carnd@$CLOUD_URL:~/model.py
    rsync --ignore-existing --recursive records/IMG carnd@$CLOUD_URL:~/records  
fi

# ssh carnd@$CLOUD_URL ; say attention. secured shell connection lost
# ./train ; exit

if [ $1 == 'drive' ]
then
    aws s3 cp s3://yang-carnd/model.h5 model.h5
    python drive.py model.h5
fi

afplay /System/Library/Sounds/Hero.aiff
say attention. $1 done