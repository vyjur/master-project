from collections import Counter

# TODO: do we use this?
def majority_element(nums):
    counts = Counter(nums)
    return max(counts, key=counts.get)