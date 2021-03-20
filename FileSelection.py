import shutil, random, os
dirpath = 'C:/Users/visha/Downloads/forecasting_val_v1.1.tar/forecasting_val_v1.1/val/data'
destDirectory = 'C:/Users/visha/MasterThesis/ArgoverseSamples/test1'

filenames = random.sample(os.listdir(dirpath), 1680)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)