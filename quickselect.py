from typing import List

def ge(a, b):
    return a >= b

def le(a, b):
    return a <= b

def select_k(nums: List[int], k: int, ge=ge, le=le, reverse=False) -> int:
    '''
    Use Quickselect algorithm to select the K-th largest/smallest element in the given list (nums).
    Custom comparison functions can be defined (functions ge() and le()). Default is largest sorting (largest to the left of list).
    The 'reverse' argument is used to reverse the defined ge and le functions (change from largest to smallest).

    Returns the element in the K-th position and the partially sorted list.
    '''

    if reverse:
        ge, le = le, ge, le
       
    left, right = 0, len(nums)-1
    
    def partition(l,r):
        '''
        the partition function from Quick Sort
        https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme
        '''
        # Set the pivot to the left index
        pivot = l

        while l < r:
            # keep increasing l until l==r or l-th element not less than the pivot element
            while l <= r and ge(nums[l], nums[pivot]):
                l += 1
            # keep increasing r until l==r or r-th element not more than the pivot element
            while l <= r and le(nums[r], nums[pivot]):
                r -= 1
            # if l still less than r (meaning the l-th element is not less than pivot and r-th element not more than pivot)
            if l < r:
                # switch the l-th and r-th elements
                nums[l],nums[r] = nums[r],nums[l]
        # switch the r-th element with the pivot element
        nums[r],nums[pivot] = nums[pivot],nums[r]
        # return the right index
        return r

    while True:
        # we only focus on one half
        # where the Kth element is
        p = partition(left,right)
        if p == k-1:
            return nums[p], p, nums
        if p > k-1:
            right = p-1
        else:
            left = p+1