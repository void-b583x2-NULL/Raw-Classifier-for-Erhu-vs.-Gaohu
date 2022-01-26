# Description
This is the first attempt of mine into the field of computational musicology, a raw classifier for classifying Erhu and Gaohu, two similar but totally different musical instruments. Initiatives used in the project are slicing features by *librosa* and a casually drawn neural network, maybe something too naive to be mentioned. Datasets come from my own recordings, with single notes for training and complex melodies for testing. A relatively good result of training is given in the files; whether to use the result already trained is determined by the parameter *train_enable*, *False* for using that result while *True* for another training. Many parameters involved in the project can be adjusted for a better performance hopefully; while features are chosen in a somewhat random way, the result of classifying may change as well.

# Files
'eh-\*' means Erhu, while 'gh-\*' means Gaohu.

# Execution
Execute *Echo.py* and everything should be done; the rest is to modify those parameters.
