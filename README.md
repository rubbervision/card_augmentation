# card_augmentation
This repo keeps a script that helps you to automatically run data augmentation for TCG's like Pokémon. It automatically sets cards into images with some shadow, rotations etc. and also directly creates the lables that are necessary to train e.g. a YOLO model. 

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


With this augmentation I was able to create a very stable card recognition tool together with the yolo library. Each augmentation has a bit of randomness in it and places the cards within the images with a bit of a rotation. YOu can simply create more of those based on the functions you find in the code. I used 2 for training and 1 for validation. I ran each augmentaion 5 times on each card to get a set of 10 training images and 5 validation images. This was quite low with ~3750 images (~250 cards * 15 augmentaions) for a YOLO model but already enough to first let my pc sh** itself for 12 hours of building the model and than successfully having a very strong model. (video follows soon)

I hope that this helps you creating your own models for TCG's like Pokémon, Magic or whatevery you like to play and find a way why you want to scan the cards. (Friend of mine wanted to include it to his streaming overlay.)
