def getBit(num: int, i: int):
    """
    Get the ith bit of num
    :param num:
    :param i:
    :return:
    """
    return (num & (1 << i)) != 0

def setBit(num: int, i: int):
    """
    Set the ith bit in num to 1
    :param num:
    :param i:
    :return:
    """
    return num | (1 << i)

def clearBit(num: int, i: int):
    """
    Set the ith bit to 0
    :param num:
    :param i:
    :return:
    """
    return num & (~(1 << i))

def clearMSBthroughi(num: int, i: int):
    """
    Clear bits from most signi bit through i
    :param num:
    :param i:
    :return:
    """
    return num & ((1 << i) - 1)

def clearithroughLSB(num: int, i: int):
    """
    Clear bits from i to least signi bit
    :param num:
    :param i:
    :return:
    """
    return num & ((-1) << (i + 1))

def printBinary(num: str):
    """
    Prints num as a binary no
    :param num:
    :return:
    """
    return int(num, 2)

# 5.1

def insertion(n: int, m: int, i: int, j: int):
    """
    Insertion: You are given two 32-bit numbers, N and M, and two bit positions, i and
    j. Write a method to insert M into N such that M starts at bit j and ends at bit i. You
    can assume that the bits j through i have enough space to fit all of M. That is, if
    M = 10011, you can assume that there are at least 5 bits between j and i. You would not, for
    example, have j = 3 and i = 2, because M could not fully fit between bit 3 and bit 2.
    :param n:
    :param m:
    :param i:
    :param j:
    :return:
    """
    # Creating mask here
    left = (-1) << (j + 1)
    right = ((1 << i) - 1)
    mask = left | right # All 1s except where u wanna insert
    n_masked = n & mask # 0s in area u wanna insert

    return n_masked | (m << i)

def binaryToString(n: float):
    """
    Converts n to binary form
    :param n:
    :return:
    """
    if n < 0 or n > 1:
        return "ERROR"

    binNo = "."
    while n > 0:
        newNum = n * 2
        dig = int(newNum / 1)

        if dig == 0:
            binNo += "0"
        else:
            binNo += "1"

        n = newNum % 1
        print(dig)
        print(n)

    return binNo

# 5.3
def flipBitToWin(n: int):
    """
    Flip Bit to Win: You have an integer and you can flip exactly one bit from a 0 to a 1. Write code to
    find the length of the longest sequence of ls you could create.

    EXAMPLE

    Input: 1775 (or: 11011101111)
    Output: 8

    :param n:
    :return:

    So what im thinking is go through the bits in one pass and get the size of each island
    separating them by 0 if there is multiple 0s between the islands.
    Then in the new list, go through it and sum up adjacent neighbours and add 1 to it.
    Return the largest sum. So for the given no it looks like
    2, 3, 4

    If no was 11001111000100 den it wud be 2, 0, 4, 0, 1, 0
    This is O(b) time and space where b is legth of binary no
    """
    zeroCtr = 0
    oneCtr = 0
    res = []

    temp = n
    if temp == 0:
        return 1

    while temp > 0: # Going from rt to left
        dig = temp % 2
        if dig == 1:
            zeroCtr = 0
            oneCtr += 1

        elif dig == 0:
            zeroCtr += 1
            if zeroCtr == 1:
                if oneCtr != 0:
                    res.append(oneCtr)
                    oneCtr = 0
            elif zeroCtr == 2: # We only wanna append one 0 if many 0s together.
                res.append(0)

        temp = temp // 2

    if oneCtr > 0:
        res.append(oneCtr)

    if len(res) == 1:
        return 1 if res[0] == 0 else res[0]

    adjSum = 0
    largestSum = 0
    for i in range(len(res) - 1):
        adjSum = res[i] + res[i + 1] + 1
        if adjSum > largestSum:
            largestSum = adjSum

    return largestSum

def flipBitToWin2(n: int):
    """
    I try to reduce the space usage here by keeping track of concurrent ones.
    :param n:
    :return:
    """

    # There is a bug when the entry is all 1s. I can hard code a special case for it.
    # Or ill just assume that there is a 0 in the beginning of every no.



    prevOne = 0
    currOne = 0
    longest = 1
    zeroCtr = 0

    while n != 0:
        if (n & 1) > 0: # If bit is a 1
            currOne += 1
            zeroCtr = 0
        else:
            zeroCtr += 1
            if zeroCtr == 1:
                prevOne = currOne
                currOne = 0
            else:
                prevOne = 0
                currOne = 0

        longest = max(longest, prevOne + currOne + 1)
        n = n >> 1

    return longest

# 4.4

def nextNumber(n: int):
    """
    Next Number: Given a positive integer, print the next smallest and the next largest number that
    have the same number of 1 bits in their binary representation.
    :param n:
    :return:

    Well brute force is just keep adding 1 and checking next number till u get a new no
    with same no of 1s.

    Well the next biggest no that is possible, u wanna flip the rightmost non trailing
    zero to a one.
    """
    c = n
    c0 = 0
    c1 = 0

    while (c & 1) == 0 and c != 0:
        c0 += 1
        c = c >> 1

    while (c & 1) == 1:
        c1 += 1
        c = c >> 1



    # So c0 is the no of trailing zeroes and c1 is the no of 1s coming before
    # the trailing zeroes.

    p = c0 + c1 # p is the bit to flip to a 1. It is a 0

    n = n | (1 << p) # Setting p bit to 1

    # Now, we wanna shift all the ones to the rightmost area except for one 1.
    # i.e. u want the bits from c1 - 2 to 0 to all be set to a 1.
    # To do this, set all bits right of p to a 0, then set bit c1 - 1 to a 1 and
    # subtract 1

    mask = ~((1 << p) - 1)
    n = n & mask # All 0 after p

    n = n | (1 << (c1 - 1))
    n = n - 1

    return n

def prevNumber(n: int):
    """
    Same prblem as above but to get the largest smaller number.


    :param n:
    :return:
    """
    c = n
    c0 = 0
    c1 = 0
    # To get the previous largest no, we wanna flip the rightmost non trailing 1 to a 0.
    # And then move the remaining ones immediately to the right of that bit.
    while (c & 1) == 1:
        c1 += 1
        c = c >> 1

    while (c & 1) == 0 and c != 0:
        c0 += 1
        c = c >> 1

    p = c0 + c1

    n = n & ~(1 << p)

    # Setting everything after p to 0
    n = n & ~((1 << p) - 1)

    temp = 1 << (c1 + 1)
    temp = temp - 1
    temp = temp << (c0 - 1)
    n = n | temp






# 5.5
# It checks whether u got exactly a signle 1 in ur binary no. If u got more than
# a single one then it returns false. This is cuz n - 1 changes the first occurence of 1 to 0
# and all the 0s to ones.
# Ultimately, it checks whether a no is a power of 2.

# 5.6
def conversion(a: int, b: int):
    """
    Conversion: Write a function to determine the number of bits you would need to flip to convert
    integer A to integer B.
    EXAMPLE
    Input: 29 (or: 11101), 15 (or: 01111)
    Output: 2
    :param a:
    :param b:
    :return:

    Well the first approach I think of is checking whether every bit is diff
    or not. If so increase a counter. This is O(b) where b is bit size. Another similar
    way is to and A and B and then count the number of zeroes in that. Or use xor and
    count the 1s.

    Yoooo, on reading the soln, I just realized c = c & (c - 1) clears the rightmost 0
    every time. Doing this is O(number of different bits in a and b) time. Whihch is waay better.
    """
    temp = a ^ b
    ctr = 0

    # Original implem below
    # while temp != 0:
    #     if temp & 1 == 1:
    #         ctr += 1
    #     temp = temp >> 1

    while(temp != 0):
        ctr += 1
        temp = temp & (temp - 1)

    return ctr

# 4.6
def pairWiseSwap(n: int):
    """
    Pairwise Swap: Write a program to swap odd and even bits in an integer with as few instructions as
    possible (e.g., bit 0 and bit 1 are swapped, bit 2 and bit 3 are swapped, and so on).
    :param n:
    :return:

    Well the soln to this is straight up stupid. Idk how anyone wud think like dat.
    U shift the even dig left by one and the odd dig right logically by one.
    And u or them. So u need a way to extract just the even dig and odd dig and put
    the remaining dig to 0 in each case. For even bit, 0101 works and this is 5 in Hex.
    1010 for odd is A in hex
    """
    m1 = 0x55555555 & n # Contains only the even dig in A
    m2 = 0xAAAAAAAA & n # Cont only odd dig in A

    # No logical right in python
    return (m1 << 1) | (m2 >> 1)
