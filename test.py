import os

# x = os.walk("./models/")


subfolders = sorted([ f.name for f in os.scandir('./model/Unet') if f.is_dir() ])

next_num = 1
if len(subfolders) > 0:
    next_num += subfolders[-1][subfolders[-1].rfind('_')+1:]
print(subfolders[-1][subfolders[-1].rfind('_')+1:])


