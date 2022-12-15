import shutil
import os
 
file_source = 'dataset/dataverse_files/'
file_destination = 'dataset/audio/neutral/'
 
get_files = os.listdir(file_source)
 
for g in get_files:
    if 'neutral' in g:
        shutil.move(file_source + g, file_destination)