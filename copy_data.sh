#zip -r pascal_context.zip pascal_context/
#zip -r ADE20K.zip ADE20K/
#zip -r mapillary2cityscapes.zip mapillary2cityscapes/
#zip -r coco_stuff_16k.zip coco_stuff_16k/
#zip -r coco_stuff_10k.zip coco_stuff_10k/


#sshpass -p "openseg@msra2019" scp -P 12019 cityscapes.zip  yuhui@23.98.133.162:/mnt/openseg/
#sshpass -p "openseg@msra2019" scp -P 12019 ADE20K.zip  yuhui@23.98.133.162:/mnt/openseg/
sshpass -p "openseg@msra2019" scp -P 12019 coco_stuff_10k.zip  yuhui@23.98.133.162:/mnt/openseg/
sshpass -p "openseg@msra2019" scp -P 12019 pascal_context.zip  yuhui@23.98.133.162:/mnt/openseg/
sshpass -p "openseg@msra2019" scp -P 12019 lip.zip  yuhui@23.98.133.162:/mnt/openseg/
sshpass -p "openseg@msra2019" scp -P 12019 mapillary-vistas-dataset_public_v1.1.zip  yuhui@23.98.133.162:/mnt/openseg/
sshpass -p "openseg@msra2019" scp -P 12019 coco_stuff_16k.zip  yuhui@23.98.133.162:/mnt/openseg/


zip -r pascal_voc.zip pascal_voc/
sshpass -p "openseg@msra2019" scp -P 12019 pascal_voc.zip  yuhui@23.98.133.162:/mnt/openseg/


ps aux | grep frnet | awk '{print $2}' | xargs sudo kill -9

ps aux | grep hrnet | awk '{print $2}' | xargs sudo kill -9

ps aux | grep res101 | awk '{print $2}' | xargs sudo kill -9