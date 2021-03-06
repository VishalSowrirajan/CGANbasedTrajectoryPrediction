import shutil, random, os
dirpath = 'C:/Users/visha/Downloads/forecasting_test_v1.1 (1).tar/forecasting_test_v1.1 (1)/test_obs/data'
destDirectory = 'C:/Users/visha/MasterThesis/ArgoverseSamples/test'

filenames = random.sample(os.listdir(dirpath), 1680)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)