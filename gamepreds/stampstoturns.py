# Idea: given a list of timestamps, deduce whose turn it is.
# Surprisingly difficult to reason through an algorithm for this.

from readercleaner import get_image_data

timestamps = [2,24,54,78]

# opencv video reader
# call classify
# use get_image_data
# save in a list of turns
# add in columns: whoseturn, targetball, targetpock, success, (obstacles)
