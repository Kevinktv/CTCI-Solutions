"""Solutions to the arrays and strings part of CTCI"""


class IsUnique:
    """Determines whether a string has all unique characters or not

    A brute force approach wud be go through the string and for every character, see whether
    that chara occurs later again. This is O(n^2).

    An algo based on sorting wud be to first sort the string which takes nlgn. And then go
    through the list once and see whether charas repeat. O(nlgn)

    A hash based approach is hashing all charas in string to a table and checking whether
    multiple charas hash to the same slot. This is an O(n) time and O(n) space complexity.
    U cud also use a bit array/bit vector but im too lazy to do dat


    """

    def isUnique1(self, string):
        """
        Hash table implementation. Goes through list once and hashes charas and sees whether
        multi charas hash to same slot.
        """
        table = set()
        for i in string:
            if i not in table:
                table.add(i)
            else:
                return False

        return True


class CheckPermutation:
    """Given two strings, write a method to decide if one is a permutation of the
    other.

    On seeing the problem for the first time, I would assume the best worst case run time
    (The best conceivable runtime/bcr) to be O(n) cuz u wud needa go through the string atleast
    once.

    Firstly both strings shud be the same length.

    A brute force approach would be to go through the first string and check for every chara
    in first, it is present in the second by going through that as well. And removing that
    chara u see each time. O(n^2)

    Sorting the list allows O(nlgn) cuz u can just check whether both strings are the same or
    not.

    Hashing can help. U hash one string with duplicates or a counter into a table. And u check
    whether the second string charas hash into the same slots reducing the counter by 1 each
    time u encounter the same letter. This is O(n) time and O(charactersetsize) space.
    Basically, check whether they have same no of characters of each kind.

    """

    def checkPermutation1(self, s1, s2):
        """Hash implem"""
        table = {}
        if len(s1) != len(s2):
            return False
        # Adding s1 letters to table with a counter
        for i in s1:
            if i not in table:
                table[i] = 1
            else:
                table[i] += 1

        for i in s2:
            if i not in table:
                return False
            elif table[i] == 0:  # Count is 0 so ran outta letters
                return False
            else:
                table[i] -= 1

        return True


class URLify:
    """Write a method to replace all spaces in a string with '%20'. You may assume that the string
    has sufficient space at the end to hold the additional characters, and that you are given the "true"
    length of the string. (Note: If implementing in Java, please use a character array so that you can
    perform this operation in place.)
    EXAMPLE
    Input: "Mr John Smith     ", 13
    Output: "Mr%20John%20Smith

    BCR/BWCRT = O(n) cuz go through string array once.

    NOTE: I am solving this using arrays cuz otherwise no inplace.


    If space is not a issue, then u can simply create a new list and insert into it as
    required. O(n) time complexity.

    If inplace is needed, the key fact is that they got space in the end for the extra
    characters. The size of new string, newsize = size of old + (2*no of space).
    So if the char array is called arr, start from arr[newsize - 1] and copy items from the
    string one at a time inserting %20 each time u see a space.
    """

    def url1(self, string, length):
        noofspaces = 0
        for i in string:
            if i == ' ':
                noofspaces += 1

        newsize = length + (2 * noofspaces)
        arr = list(string)  # Converting string to an array for in place
        arr += [None] * (2 * noofspaces)  # making enough space at end cuz python no fixed size

        size = newsize - 1  # Copy
        for i in range(length - 1, -1, -1):
            if arr[i] == ' ':
                arr[size] = '0'
                size -= 1
                arr[size] = '2'
                size -= 1
                arr[size] = '%'
                size -= 1

            else:
                arr[size] = arr[i]
                size -= 1

        return "".join(arr)


class PalindromePermutation:
    """
    Given a string, write a function to check if it is a permutation of a palindrome.
    A palindrome is a word or phrase that is the same forwards and backwards. A permutation
    is a rearrangement of letters. The palindrome does not need to be limited to just dictionary words.
    1.5
    1.6
    EXAMPLE
    Input: Tact Coa
    Output: True (permutations: "taco cat", "atco eta", etc.)

    This thingy depends on whether the string is odd or even length. If odd, every character
    that occurs must have a pair with it except one. If even length, then every chara must
    have a pair with it. Also, I am assuming spaces are irrelevant so
     remove those.
     One way to solve this wud be using a hash table/character bit array. U hash each
     character in the string into the table except spaces in the first pass. Then, go through
     every key in the hash table and check whether its count is even when the length of string
     is even or check whether exactly one character has odd count and rest are even when
     length of string is odd. U can use the built in counter class in python which
     I forgor about. O(n)

     EDIT: The first even condition is redundant. U can basically check whether odd no of
     chara appears atmost once.


    """

    def palindromepermutation(self, string):
        """Above imple"""
        table = {}
        for i in string:
            if i not in table:
                table[i] = 1
            else:
                table[i] += 1

        # Even length
        if len(string) % 2 == 0:
            for i in table:
                if table[i] % 2 != 0:
                    return False
            return True

        else:
            flag = 0
            for i in table:
                if table[i] % 2 != 0:
                    if flag == 0:
                        flag = 1
                    else:
                        return False

            return True


class OneWay:
    """
    There are three types of edits that can be performed on strings: insert a character,
    remove a character, or replace a character. Given two strings,
    write a function to check if they are
    one edit (or zero edits) away.
    EXAMPLE
    pale, ple -> true
    pales, pale -> true
    pale, bale -> true
    pale, bake -> false
    """

    def oneway(self, s1: str, s2: str):
        """
        apple
        appd

        :param s1:
        :param s2:
        :return:
        """
        if abs(len(s1) - len(s2)) >= 2:
            return False

        if len(s1) == len(s2):  # Only replace is possible. Only one max possible diff
            changes = 0
            for i in range(len(s1)):
                if s2[i] != s1[i]:
                    changes += 1

                if changes >= 2:
                    return False
            return True

        else:  # Then there is a difference of 1 between both strings. Dat means remove/inrt
            (shorter, longer) = (s1, s2) if len(s1) < len(s2) else (s2, s1)

            for i in range(len(shorter)):  # No need to check the last letter if different.
                if longer[i] != shorter[i]:  # Then dat means one chara was shifted in shorter.
                    # We needa check if rest of string is equal.
                    return longer[i + 1:] == shorter[i:]
            return True

    def onewaywithperm(self, s1: str, s2: str):
        """Okay, so this implementation is wrong cuz I was thinking about the prev
        question as well while implementing it. This function checks whether s1 can be changed
        to s2 with only one change provided they can also be permutations of each other.
        So pale and elap gives true
        pale and elab gives true etc.

        """
        (shorter, longer) = (s1, s2) if len(s1) < len(s2) else (s2, s1)
        lenshorter = len(shorter)
        lenlonger = len(longer)

        # If more than 2 difference in length
        if abs(lenlonger - lenshorter) > 1:
            return False

        table = {}

        # So only remove/insert possible
        if lenshorter != lenlonger:
            for i in longer:
                if i not in table:
                    table[i] = 1
                else:
                    table[i] += 1

            for i in shorter:
                if i not in table:
                    return False

                elif table[i] == 0:
                    return False

                else:
                    table[i] -= 1
            return True

        else:  # This means both strings have same size so only replace is possible.
            for i in longer:
                if i not in table:
                    table[i] = 1
                else:
                    table[i] += 1

            count = 0

            for i in shorter:
                if i not in table:  # A new letter
                    count += 1
                elif table[i] == 0:
                    count += 1  # An extra letter in shorter not in longer
                else:
                    table[i] -= 1

                if count > 1:
                    return False

            return True


class StringCompression:
    """
    Implement a method to perform basic string compression using the counts
    of repeated characters. For example, the string aabcccccaaa would become a2blc5a3. If the
    "compressed" string would not become smaller than the original string, your method should return
    the original string. You can assume the string has only uppercase and lowercase letters (a - z).

    I am using a stack keeping track of the latest string and how often
    """

    def stringCompression(self, string):

        stack = []

        for i in string:
            if stack == []:
                stack.append([i, 1])
            else:
                if stack[-1][0] == i:
                    stack[-1][1] += 1
                else:
                    stack.append([i, 1])

        result = ""
        for i in stack:
            result += i[0] + str(i[1])

        return result if len(result) < len(string) else string


class RotateMatrix:
    """
    Given an image represented by an NxN matrix, where each pixel in the image is 4
    bytes, write a method to rotate the image by 90 degrees. Can you do this in place?
    """

    def rotateMatrixNotInPlace(self, arr: list[list], n: int):
        """
        I am going layer by layer and putting it in correct place. So for a 90 degree
        clockwise rotation, row 0 becomes col n - 1
        row 1 becomes col n - 2
        row 2 becomes col n - 3
        etc etc etc

        row i col j -> (i, j) gets translated to (j, n - 1 - i)

        This is an inplace rotation using a result matrix
        """
        result = [[None for x in range(n)] for y in range(n)]
        for i in range(n):
            for j in range(n):
                result[j][n - i - 1] = arr[i][j]

        for i in range(n):
            print(result[i])

        return result

    def rotateMatrixInPlace(self, arr: list[list], n: int):
        """


        :param arr: 2d array
        :param n: Size of array
        :return: Rotated 90 degrees
        """
        for i in range(0, n // 2):
            for j in range(i, n - i - 1): # Till n - i - 1 (rightmost excluded) cuz u dont wanna
                # rotate edges again.
                top = arr[i][j]
                print(top)
                arr[i][j] = arr[j][n - i - 1] # Top to right
                arr[j][n - i - 1] = arr[n - i - 1][n - j - 1]  # RIGHT TO BOTTOM
                arr[n - i - 1][n - j - 1] = arr[n - j - 1][i] # bottom to left
                arr[n - j - 1][i] = top # Left to top


class ZeroMatrix:
    """
    Write an algorithm such that if an element in an MxN matrix is 0, its entire row and
    column are set to 0.
    """
    def zeroMatrixOmegaofmplusnspace(self, arr: list[list], m: int, n: int):
        """
        This solution uses an array to hold which all rows have a zero in it. In one
        pass, I check which rows have zero and which col have zero then next pass zero em.
        """
        rows = set()
        cols = set()
        for i in range(m):
            for j in range(n):
                if arr[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        for i in rows:
            self.setRowZero(arr, m, n, i)

        for j in cols:
            self.setColZero(arr, m, n, j)


    def setRowZero(self, arr: list[list], m, n, row):
        """
        Sets the given row in arr into 0s
        :param arr:
        :param row:
        :return:
        """
        for i in range(n):
            arr[row][i] = 0

    def setColZero(self, arr: list[list], m, n, col):
        """
        Sets the given col in arr into 0s
        :param arr:
        :param m:
        :param n:
        :param row:
        :return:
        """
        for i in range(m):
            arr[i][col] = 0


class StringRotation:
    """
    Assume you have a method isSubstringwhich checks if one word is a substring
    of another. Given two strings, sl and s2, write code to check if s2 is a rotation of sl using only one
    call to isSubstring (e.g., "waterbottle" is a rotation of"erbottlewat").
    """
    def stringRotation(self, s1: str, s2: str):
        """
        Checks whether s1 is a rotation of s2.
        s1 is a rotation of s2 if s1 can be written as xy and s2 can be written as yx.
        This means that s1s1 = xyxy so the middle part has a yx in it. U can use substring
        s2 in s1s1
        """
        new = s1 + s1
        return True if new.find(s2) != -1 else False

