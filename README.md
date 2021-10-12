# techseries-dev-coderpro

## Time-Space Complexity

![Screen Shot 2021-10-12 at 1 23 33](https://user-images.githubusercontent.com/51218415/136902728-38cd98cc-2594-4054-8715-bfcf2f2ca99b.png)

- **Linear. O(n)** - Most optimal algorithms run in linear time.  An easy way to identify this is to determine if you're visiting every node or item once and only once.  If you are, it is linear... it doesn't matter how many operations you're doing whether it's 1, 2, 3, or 4 lines of code you're executing per node.  Generally, you are still doing a constant amount of work per input.
- **Constant.  O(k)** - Constant time algorithms have a running time independent of the input size.  Mathematical formulas for instance have fixed running times and are considered constant time.
- **Logarithmic. O(log(n))** - Logarithmic algorithms are often seen in trees.  It's best to think of "logarithmic" as the "height of the tree."  So, a binary search, for instance, often includes traversing down the height of a tree and can be considered logarithmic in time.  (Although, it may still be more accurate to say that for an unbalanced tree, the runtime is in the worst case linear.)  
- **Superlinear. O(n*log(n))**.  Most sorts operate in "n log n" time.  This includes popular sorting algorithms like quicksort, mergesort, or heapsort.  (Actually, quicksort is O(n2) time in the worst-case scenario generally).
- **Quadratic or Cubic / Polynomial. O(n^2) or O(n^3)**.  Brute force algorithms often run in O(n2) or O(n3) time where you may be looping within a loop.  It's easy to identify if you see a for-loop inside a for-loop, where for each element i you iterate through another element j, for instance.  A common scenario is, given two arrays, find the common elements in each array where you would simply go through each element and check whether it exists in the other array.  This would execute in O(n*m) time, where n and m are the sizes of each array.  It's still great to name these brute force algorithms if you can identify them.
- **Exponential. O(2^n)**.  Exponential algorithms are quite terrible in running time.  A classic example is determining every permutation of a set of n bits (it would take 2n combinations).  Another example is computing the fibonacci sequence fib(n) = fib(n-1) + fib(n-2), where for each item, it requires the computation of two more subproblems.  
- **Factorial. O(n!)**.  These algorithms are the slowest and don't show up that often.  You might see this in combinatorial problems, or like a "traveling salesman" problem where given n nodes, you need to find the optimal path from start to finish.  In your first iteration, you have a selection of n cities to visit, then n-1 cities, then n-2 cities, n-3 cities, etc., until you reach the last city.   That runtime is n * (n -1 ) * (n - 2) * (n -3 ) ... 1 = O(n!).

A lot of candidates get stuck here by either getting too deep in nitty gritty details and overcomplicating this like saying "This is O(3 * k *  n2), where k is the number of comparisons..." Most software engineers don't care about this level of detail, and you can often get away with simply saying "This is quadratic time because we have two for-loops, each one iterating from 1 to n."

One more tip - do not say "This is O(m + v + e)," when you haven't defined what m, v, or e are.  You generally want to say "... where m is the height of the matrix, v is the number of vertices, e is the number of edges, etc.,"  Once you start reciting formulas without defining the constants you're using, your analysis will appear amateurish.

Most interviewers will focus on time-complexity, but it is great to also consider space-complexity too.  Algorithms are commonly tradeoffs between time and space.  For instance, you may be able to take a polynomial algorithm and convert it to an O(n) algorithm, but it requires creation of a hashmap of size O(n).  That's a good trade-off to be able to talk about because additional space is needed.

## Setup

Code Runner for VSCode

 ```
  "code-runner.clearPreviousOutput": true,
  "code-runner.executorMap": {
    "python":"python3",
  },
 ```
 
 Or Online Python interpreter

https://replit.com/languages/python3

## Valid Binary Search Tree

```python
class Node(object):
  def __init__(self, val, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class Solution(object):
  def _isValidBSTHelper(self, n, low, high):
    if not n:
      return True
    val = n.val
    if ((val > low and val < high) and
        self._isValidBSTHelper(n.left, low, n.val) and
        self._isValidBSTHelper(n.right, n.val, high)):
        return True
    return False

  def isValidBST(self, n):
    return self._isValidBSTHelper(n, float('-inf'), float('inf'))


#   5
#  / \
# 4   7
node = Node(5)
node.left = Node(4)
node.right = Node(7)
print(Solution().isValidBST(node))

#   5
#  / \
# 4   7
#    /
#   2
node = Node(5)
node.left = Node(4)
node.right = Node(7)
node.right.left = Node(2)
print(Solution().isValidBST(node))
# False
```

## Ransom Note

```python
from collections import defaultdict

class Solution(object):
  def canSpell(self, magazine, note):
    letters = defaultdict(int)
    for c in magazine:
      letters[c] += 1

    for c in note:
      if letters[c] <= 0:
        return False
      letters[c] -= 1

    return True

print(Solution().canSpell(['a', 'b', 'c', 'd', 'e', 'f'], 'bed'))
# True

print(Solution().canSpell(['a', 'b', 'c', 'd', 'e', 'f'], 'cat'))
# False
```

## Add two numbers as a linked list

```python
class Node(object):
  def __init__(self, x):
    self.val = x
    self.next = None


class Solution:
  def addTwoNumbers(self, l1, l2):
    return self.addTwoNumbersRecursive(l1, l2, 0)
    # return self.addTwoNumbersIterative(l1, l2)

  def addTwoNumbersRecursive(self, l1, l2, c):
    val = l1.val + l2.val + c
    c = val // 10
    ret = Node(val % 10)

    if l1.next != None or l2.next != None:
      if not l1.next:
        l1.next = Node(0)
      if not l2.next:
        l2.next = Node(0)
      ret.next = self.addTwoNumbersRecursive(l1.next, l2.next, c)
    elif c:
      ret.next = Node(c)
    return ret

  def addTwoNumbersIterative(self, l1, l2):
    a = l1
    b = l2
    c = 0
    ret = current = None

    while a or b:
      val = a.val + b.val + c
      c = val // 10
      if not current:
        ret = current = Node(val % 10)
      else:
        current.next = Node(val % 10)
        current = current.next

      if a.next or b.next:
        if not a.next:
          a.next = Node(0)
        if not b.next:
          b.next = Node(0)
      elif c:
        current.next = Node(c)
      a = a.next
      b = b.next
    return ret

l1 = Node(2)
l1.next = Node(4)
l1.next.next = Node(3)

l2 = Node(5)
l2.next = Node(6)
l2.next.next = Node(4)

answer = Solution().addTwoNumbers(l1, l2)
while answer:
  print(answer.val, end=' ')
  answer = answer.next
# 7 0 8
```

## Two Sum

```python
class Solution(object):
  def twoSum(self, nums, target):
    for i1, a in enumerate(nums):
      for i2, b in enumerate(nums):
        if a == b:
          continue
        if a + b == target:
          return [i1, i2]
    return []

  def twoSumB(self, nums, target):
    values = {}
    for i, num in enumerate(nums):
      diff = target - num
      if diff in values:
        return [i, values[diff]]
      values[num] = i
    return []


print(Solution().twoSumB([2, 7, 11, 15], 18))
```

## First and Last Indices of an Element in a Sorted Array

```python
class Solution:
  def getRange(self, arr, target):
    first = self.binarySearchIterative(arr, 0, len(arr) - 1, target, True)
    last = self.binarySearchIterative(arr, 0, len(arr) - 1, target, False)
    return [first, last]

  def binarySearch(self, arr, low, high, target, findFirst):
    if high < low:
      return -1
    mid = low + (high - low) // 2
    if findFirst:
      if (mid == 0 or target > arr[mid - 1]) and arr[mid] == target:
        return mid
      if target > arr[mid]:
        return self.binarySearch(arr, mid + 1, high, target, findFirst)
      else:
        return self.binarySearch(arr, low, mid - 1, target, findFirst)
    else:
      if (mid == len(arr)-1 or target < arr[mid + 1]) and arr[mid] == target:
        return mid
      elif target < arr[mid]:
        return self.binarySearch(arr, low, mid - 1, target, findFirst)
      else:
        return self.binarySearch(arr, mid + 1, high, target, findFirst)

  def binarySearchIterative(self, arr, low, high, target, findFirst):
    while True:
      if high < low:
        return -1
      mid = low + (high - low) // 2
      if findFirst:
        if (mid == 0 or target > arr[mid - 1]) and arr[mid] == target:
          return mid
        if target > arr[mid]:
          low = mid + 1
        else:
          high = mid - 1
      else:
        if (mid == len(arr)-1 or target < arr[mid + 1]) and arr[mid] == target:
          return mid
        elif target < arr[mid]:
          high = mid - 1
        else:
          low = mid + 1

arr = [1, 3, 3, 5, 7, 8, 9, 9, 9, 15]
x = 9
print(Solution().getRange(arr, 9))
# [6, 8]
```

## Permutations

```python
class Solution(object):
  def _permuteHelper(self, nums, start=0):
    if start == len(nums) - 1:
      return [nums[:]]

    result = []
    for i in range(start, len(nums)):
      nums[start], nums[i] = nums[i], nums[start]
      result += self._permuteHelper(nums, start + 1)
      nums[start], nums[i] = nums[i], nums[start]
    return result

  def permute(self, nums):
    return self._permuteHelper(nums)

  def permute2(self, nums, values=[]):
    if not nums:
      return [values]
    result = []
    for i in range(len(nums)):
      result += self.permute2(nums[:i] + nums[i+1:], values + [nums[i]])
    return result

  def permute2Iterative(self, nums):
    results = []
    stack = [(nums, [])]
    while len(stack):
      nums, values = stack.pop()
      if not nums:
        results += [values]
      for i in range(len(nums)):
        stack.append((nums[:i]+nums[i+1:], values+[nums[i]]))
    return results


print(Solution().permute([1, 2, 3]))
# [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]

print(Solution().permute2([1, 2, 3]))
# [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]

print(Solution().permute2Iterative([1, 2, 3]))
# [[3, 2, 1], [3, 1, 2], [2, 3, 1], [2, 1, 3], [1, 3, 2], [1, 2, 3]]
```

## Sorting a list with 3 unique numbers

```python
ef sortNums(nums):
  counts = {}
  for n in nums:
    counts[n] = counts.get(n, 0) + 1
  return ([1] * counts.get(1, 0) +
          [2] * counts.get(2, 0) +
          [3] * counts.get(3, 0))


def sortNums2(nums):
  one_index = 0
  three_index = len(nums) - 1
  index = 0
  while index <= three_index:
    if nums[index] == 1:
      nums[index], nums[one_index] = nums[one_index], nums[index]
      one_index += 1
      index += 1
    elif nums[index] == 2:
      index += 1
    elif nums[index] == 3:
      nums[index], nums[three_index] = nums[three_index], nums[index]
      three_index -= 1
  return nums


print(sortNums2([3, 3, 2, 1, 3, 2, 1]))
# [1, 1, 2, 2, 3, 3, 3]
```

## Queue Reconstruction By Height

```python
class Solution:
  def reconstructQueue(self, input):
    input.sort(key=lambda x:
               (-x[0], x[1])
               )
    res = []
    for person in input:
      res.insert(person[1], person)
    return res


input = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
print(Solution().reconstructQueue(input))
# [[5,0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]
```

## Find the non-duplicate number

```python
class Solution(object):
  def singleNumber(self, nums):
    occurrence = {}

    for n in nums:
      occurrence[n] = occurrence.get(n, 0) + 1

    for key, value in occurrence.items():
      if value == 1:
        return key

  def singleNumber2(self, nums):
    unique = 0
    for n in nums:
      unique ^= n # XOR
    return unique
    

print(Solution().singleNumber2([4, 3, 2, 4, 1, 3, 2]))
# 1
```

## Reverse A Linkedlist

```python
class Node(object):
  def __init__(self, val, next=None):
    self.val = val
    self.next = next

  def __repr__(self):
    res = str(self.val)
    if self.next:
      res += str(self.next)
    return res

class Solution(object):
  def reverse(self, node):
    curr = node
    prev = None

    while curr != None:
      tmp = curr.next
      curr.next = prev
      prev = curr
      curr = tmp

    return prev

node = Node(1, Node(2, Node(3, Node(4, Node(5)))))

print(Solution().reverse(node))
# 54321
```

## Maximum In A Stack

```python
class MaxStack(object):
  def __init__(self):
    self.stack = []
    self.maxes = []

  def push(self, val):
    self.stack.append(val)
    if self.maxes and self.maxes[-1] > val:
      self.maxes.append(self.maxes[-1])
    else:
      self.maxes.append(val)

  def pop(self):
    if self.maxes:
      self.maxes.pop()
    return self.stack.pop()

  def max(self):
    return self.maxes[-1]

s = MaxStack()
s.push(1)
s.push(2)
s.push(3)
s.push(2)
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
print('max', s.max())
print(s.pop())
```

## Course Schedule

```python
class Solution:

  def _hasCycle(self, graph, course, seen, cache):
    if course in cache:
      return cache[course]

    if course in seen:
      return True
    if course not in graph:
      return False

    seen.add(course)
    ret = False
    for neighbors in graph[course]:
      if self._hasCycle(graph, neighbors, seen, cache):
        ret = True
        break
    seen.remove(course)

    cache[course] = ret
    return ret

  def canFinish(self, numCourses, prerequisites):
    graph = {}
    for prereq in prerequisites:
      if prereq[0] in graph:
        graph[prereq[0]].append(prereq[1])
      else:
        graph[prereq[0]] = [prereq[1]]

    for course in range(numCourses):
      if self._hasCycle(graph, course, set(), {}):
        return False

    return True


print(Solution().canFinish(2, [[1, 0]]))
# True

print(Solution().canFinish(2, [[1, 0], [0, 1]]))
# False
```

## Find Pythagorean Triplets

```python
def findPythagoreanTriplets(nums):
  for a in nums:
    for b in nums:
      for c in nums:
        if a*a + b*b == c*c:
          return True
  return False

def findPythagoreanTriplets2(nums):
  squares = set([n*n for n in nums])

  for a in nums:
    for b in nums:
      if a * a + b * b in squares:
        return True
  return False

print(findPythagoreanTriplets2([3, 5, 12, 5, 13]))
# True
```

