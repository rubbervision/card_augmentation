# card_augmentation
This repo keeps a script that helps you to automatically run data augmentation for TCG's like Pokémon.

Why? 
I created that as I .. 
1. .. didn't want to collect all the data manually.
2. .. didn't want to add the labels myself.
3. .. and wanted to create software that can identify cards for live streams or videos to collect and store the information of cards that have been within the booster.

Sooo basically just save a lot of time spending on manual annoying tasks. Let's not waste more time and jump into functionality.

-------------------------------------------------------------------

Upfront:
1. python script "auto_card_augmentation.py"
2. .env file filled with your variables, paths etc.
3. .json file with the information of your cards like id, name etc.
4. Folder with all cards for which you want to create augmentaions for. I personally got a set from an open db wich has the images of the cards in best quality. IMPORTANT: They somehow need to be identified. Best is that the information of the name is fully or partially within the .json file.

How to run the script:
1. Prepre all images. Maybe there is an open library somewhere in the www that you can use to get an image of each card and also a .json with all the references. 
2. When step one is done make yourself familiar with the .env.example file and fill out everything as needed like folder path of the images etc. After that rename to the file to ".env".
3. Check within the python script if the class_id is references correct. In my .json example it was the "model_class_id" but it 