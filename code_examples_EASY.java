//IS HAPPY?
//Kids With the Greatest Number of Candies
//Sort Integers by The Number of 1 Bits
//The K Weakest Rows in a Matrix - SOLVE WITH MAP && REDUCE
// FLOOD FILL - Solve with JAVA
// Design HashSet - ALso designing other type of data structures
// Merge two binary trees - Solve with JAVA
//Arranging coins - JAVA


//BALANCED BINARY TREE
function isBalanced(root: TreeNode | null): boolean {
    if (!root) return true;

  let height = function (node: TreeNode | null) {
    if (!node) return 0;
    return 1 + Math.max(height(node.right), height(node.left));
  };

  return (
    Math.abs(height(root.left) - height(root.right)) < 2 &&
    isBalanced(root.right) &&
    isBalanced(root.left)
  );
};

// WORD PATTERN
function wordPattern(pattern: string, s: string): boolean {
  const patternMap = new Map<string, string>();
  const wordsMap = new Map<string, string>();
  const patternLen = pattern.length;
  const words = s.split(" ");
  if (patternLen != words.length) {
    return false;
  }
  for (let i = 0; i < patternLen; i++) {
    const ch = pattern[i];
    const word = words[i];
    if (patternMap.has(ch)) {
      if (patternMap.get(ch) != word) {
        return false;
      }
    } else {
      patternMap.set(ch, word);
    }
    if (wordsMap.has(word)) {
      if (wordsMap.get(word) != ch) {
        return false;
      }
    } else {
      wordsMap.set(word, ch);
    }
  }
  return true;
};

//INTERSECTION OF TWO ARRAYS
function intersection(nums1: number[], nums2: number[]): number[] {
  const set = new Set();
  const res = new Set<number>();
  nums1.forEach((num) => set.add(num));
  nums2.forEach((num) => {
    if (set.has(num)) res.add(num);
  });
  return [...res];
}

//STRING REPEAT
function repeatedStringMatch(a: string, b: string): number {
    const aLength = a.length;
    const bLength = b.length;

    let repeatedA = "";
    let count = 0;
    while (repeatedA.length < b.length) {
        repeatedA = repeatedA + a;
        count++;
    }
    if(repeatedA.includes(b)) {
        return count;
    }
    if ((repeatedA + a).includes(b)) {
        return ++count;
    }
    return -1;
};

//BST MIN DEPTH
//DFS
function minDepth(root: TreeNode | null): number {
    if (!root) return 0;
    
    return minDepthHelper(root, 0);
    // T.C: O(N)
    // S.C: O(H)
};

function minDepthHelper(node: TreeNode | null, prevDepth: number): number {
    if (!node) return Infinity;
    if (!node.left && !node.right) return prevDepth + 1;
    
    return Math.min(minDepthHelper(node.left, prevDepth + 1), minDepthHelper(node.right, prevDepth + 1));
}
//BFS
function minDepth(root: TreeNode | null): number {
    if (root === null) return null;
    if (root.left === null) {
        return minDepth(root.right) + 1;
    }
    if (root.right === null) {
        return minDepth(root.left) + 1;
    }
    let left = minDepth(root.left);
    let right = minDepth(root.right);
    return Math.min(left, right) + 1;
};

// LONGEST PALINDROME
function longestPalindrome(s: string): number {
    if (s.length === 0) return 0;
    if (s.length === 1) return 1;
    let map = new Map();
    s.split('').map(char => {
        if (!map[char]) map[char] = 1;
        else map[char]++;
    });
    let result = 0;
    Object.values(map).forEach((value, index) => {
        if ((value & 1) == 0) {
            //console.log(value);
            result = result + value;
        } else if ((value & 1) == 1) {
            //console.log(value);
            //console.log(Math.floor(value / 2) * 2);
            result += Math.floor(value / 2) * 2;
            if (result%2 === 0) {
                result++
            }
        }
    })
    //console.log(map);
    return result;
};

// ISOMORPHIC STRINGS
function isIsomorphic(s: string, t: string): boolean {
    if (s.length != t.length) return false;
    
    let sCharMappings: Map<string, string> = new Map<string, string>();
    let tCharMappings: Map<string, string> = new Map<string, string>();
    
    for (let i = 0; i < s.length; i++) {
        const currentSChar: string = s[i];
        const currentTChar: string = t[i];
        
        // check if s char and t char aren't mapped already
        const sCharMapped: boolean = sCharMappings.has(currentSChar);
        const tCharMapped: boolean = tCharMappings.has(currentTChar);
        
        if (sCharMapped) {
            // if sChar is mapped already, is it mapped to the same t value ?
            if (sCharMappings.get(currentSChar) != currentTChar)
                return false;
        }
        else if (tCharMapped) {
            // if tChar is mapped already, is it mapped to the same s value ?
            if (tCharMappings.get(currentTChar) != currentSChar)
                return false;
        }
        else {
            sCharMappings.set(currentSChar, currentTChar);
            tCharMappings.set(currentTChar, currentSChar);
        }
    }
    return true;
};

//ATBASH SIPHER
const alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"];
const reverseAlphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"].reverse();
export const encode = (strEncode) => {
  let encodedStr = "";
  strEncode.replace(/\s/g, "").toLowerCase().split('').map(char => {
    let index = alphabet.findIndex(letter => letter === char);
    
    encodedStr += reverseAlphabet[index];
  });
  return encodedStr;
};
export const decode = (strDecode) => {
  let decodedStr = "";
  strDecode.replace(/\s/g, "").toLowerCase().split('').map(char => {
    let index = alphabet.findIndex(letter => letter === char);
    decodedStr += reverseAlphabet[index];
  });
  return decodedStr;

// N-th PRIME ON GIVEN PRIME NUMBER
export const prime = (primeNum) => {
  if (primeNum === 0) {
    throw new Error('there is no zeroth prime');
  }
  let count = 0;
  for (let i = 2; i < 104744; i++) {
    if(isPrime(i) === true) {
      count++;
    }
    if (count === primeNum) {
      return i;
      break;
    }
  }
};
const isPrime = (num) => {
  let output = true;
  for (let i = 2; i < num; i++) {
    if(num % i === 0) {
      output = false;
      break;
    }
  }
  return output
}

//FLATTEN DEEP NESTED ARRAYS WITH RECURSION
export const flatten = (array) => {
  let flattenedArr = [];

  for (let i = 0; i < array.length; i++) {
    if (array[i] != null) {
      if(Array.isArray(array[i])) {
      flattenedArr = flattenedArr.concat(flatten(array[i]))
      } else {
        flattenedArr.push(array[i])
      }
    }
  }
  return flattenedArr;
};

//HOUSE ROBBER
function rob(nums: number[]): number {
    if (!nums || nums.length === 0) return 0;
    
    const dp: number[] = new Array(nums.length).fill(0);
    dp[0] = nums[0];
    dp[1] = Math.max(dp[0], nums[1]);
    for (let i = 2; i < nums.length; i++) {
        dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
    }
    return dp[nums.length - 1];
};


//COUNTING BITS 2 WAY
function countBits(n: number): number[] {
    let resArr = [];
    for (let i = 0; i <= n; i++) {
        let binaryString = (i >>> 0).toString(2);
        //console.log((n >>> 0).toString(2));
        let count = 0;
        for (let j = 0; j < binaryString.length; j++) {
            if (binaryString[j] === '1') {
                count++
            };
        }
        resArr.push(count);
    }
    return resArr;
};
function countBits(num: number): number[] {
    const res = new Uint8Array(num + 1);
    for (let i = 0; i < res.length; i++) res[i] = res[i >> 1] + (i & 1);
    return [...res];
}


//DYNAMIC CLIMBING STAIRS WITH MEMO ARRAY
function climbStairs(n: number): number {
    let memo = [];
    memo[0] = 0;
    memo[1] = 1;
    memo[2] = 2;

    if (n < 3) return memo[n];

    for (let i = 3; i <= n; i++) {
        memo[i] = memo[i-1] + memo[i-2];
    }
    return memo[memo.length-1];
};

//BST - PATH SUM
function pathSum(root: TreeNode | null, targetSum: number): number[][] {
    const paths: number[][] = []; 
    function dfs(node: TreeNode | null, sum: number, path: number[]): void {
        if(!node) return; 
        if(!node.left && !node.right) {
            if(sum + node.val === targetSum) {
                paths.push([...path, node.val]); 
            }
            return; 
        }
        dfs(node.left, sum + node.val, [...path, node.val]); 
        dfs(node.right, sum + node.val, [...path, node.val])
    }
    dfs(root, 0, []); 
    return paths; 
};

//BST MAX DEPTH
function maxDepth(root: TreeNode | null): number {
    if(!root) return 0;
    
    let left = maxDepth(root.left);
    console.log("LEFT: ", left);

    let right = maxDepth(root.right);
    console.log("RIGHT: ", right);

    return 1 + Math.max(left, right);
};



//BST - ROOT TO LEAF PATH
function binaryTreePaths(root: TreeNode | null): string[] {
    let resultPaths = [];
    const dfs = (root: TreeNode | null, currentPath: string) => {
        if (!root) return;
        currentPath += !currentPath ? root.val : `->${root.val}`;
        if (isLeaf(root)) resultPaths.push(currentPath)
        dfs(root.left, currentPath);
        dfs(root.right, currentPath);
    };
    dfs(root, '');
    return resultPaths;
};

const isLeaf = (root: TreeNode | null): boolean => {
    return (root && !root.left && !root.right)
}


//PATH SUM BST
function hasPathSum(root: TreeNode | null, targetSum: number): boolean {
    if(!root) return false;
    const reducedSum = targetSum - root.val;
    console.log(`reducedSum: ${reducedSum}`)
    return ((reducedSum === 0 && isLeaf(root)) ||
            hasPathSum(root?.left, reducedSum) ||
            hasPathSum(root?.right, reducedSum)
    )
};
const isLeaf = (root: TreeNode | null): boolean => {
    return (root && !root.left && !root.right);
}


//BST TRAVERSAL INORDER-PREORDER-POSTORDER
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */
 function postorderTraversal(root: TreeNode | null): number[] {
        if (!root) return [];
        return [
            ...postorderTraversal(root.left),
            ...postorderTraversal(root.right),
            root.val
        ]
};
function preorderTraversal(root: TreeNode | null): number[] {
    if (!root) return [];

    const result = [
        root.val,
        ...preorderTraversal(root.left),
        ...preorderTraversal(root.right)
    ];
    console.log(result);
    return result;
};
function inorderTraversal(root: TreeNode | null): number[] {
    if (root == null) return [];
    //console.log(root);
    //inorderTraversal(root.right);
    const resultArr = [
        ...inorderTraversal(root.left),
        root.val,
        ...inorderTraversal(root.right)
    ]
    console.log(resultArr);
    return resultArr;
};

//TREE MAX SUM
class Solution {
    static int maxSum(TreeNode root) {
        // TODO: implementation
        if (root == null) return 0;
        return root.value + Math.max(maxSum(root.left),maxSum(root.right));
    }
}

//AMOUNT OF UNIQUE SUBSETS - EFFICIENT
import java.util.*;
public class Subsets {  
  public static <T> long count(T[] elems) {
    
    LinkedHashSet<T> set
            = new LinkedHashSet<T>();
 
    for (int i = 0; i < elems.length; i++)
        set.add(elems[i]);
    
    //System.out.println(set);
    Long result = (1L << set.size()) - 1L;
    //System.out.println((1 << set.size()) - 1);
    return result;
  }  
}


//ESTIMATING AMOUNTS OF SUBSETS
import java.util.*;
public class Subsets {  
  public static <T> long count(T[] elems) {
    
    LinkedHashSet<T> set
            = new LinkedHashSet<T>();
 
    for (int i = 0; i < elems.length; i++)
        set.add(elems[i]);
    
    
    System.out.println((1 << set.size()) - 1);
    return (1 << set.size()) - 1;
  }  
}


//COUNT THE SMILEY FACES
/**
Given an array (arr) as an argument complete the function countSmileys that should return the total number of smiling faces.

Rules for a smiling face:

Each smiley face must contain a valid pair of eyes. Eyes can be marked as : or ;
A smiley face can have a nose but it does not have to. Valid characters for a nose are - or ~
Every smiling face must have a smiling mouth that should be marked with either ) or D
No additional characters are allowed except for those mentioned.

Valid smiley face examples: :) :D ;-D :~)
Invalid smiley faces:  ;( :> :} :] */
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.List;
import java.util.Arrays;
public class SmileFaces {
  
  public static int countSmileys(List<String> arr) {
      // Just Smile :)
      String[] allSmileys = {":)",";)",":D",";D",":-D",":~D",":-)",":~)",";~D",";~)",";-D",";-)"};
      int count = 0;
      for (String str : arr) {
          if (Arrays.asList(allSmileys).contains(str)) count++;   
      }
      return count;
  }
}

//CONSECUTIVE STRINGS
class LongestConsec {
    
    public static String longestConsec(String[] strarr, int k) {
        // your code
      
      String mergedStr = "";
      for (int i = 0; i < strarr.length - k + 1; i++) {
        String strItr = "";
        for (int j = i; j < i + k; j++) {
                strItr += strarr[j];
        }
        if (strItr.length() > mergedStr.length()) {
          mergedStr = strItr;
        }
      }
      //System.out.println(mergedStr);
      return mergedStr;
    }
}

//DETECT PANGRAM
import java.util.*;
public class PangramChecker {
  public boolean check(String sentence){
    //code
    String replaced = sentence.replaceAll("[^A-Za-z]", "");
    
    System.out.println(replaced);
    
    Set<String> letters = new HashSet<>(Arrays.asList(replaced.split("")));
    
    System.out.println(letters);
    
    return letters.size() >= 26;
  }
}


//BUILD A PILE OF CUBES
export function findNb(m: number): number {
  // your code
  let total = 0
  let n = 0

  while(total < m) {
    n += 1
    total += n**3
  }

  return total === m ? n : -1
}

//FIBONACCI WITH CACHE
class Solution {
    public int fib(int n) {
        
        if (n <= 1) return n;
        
        int[] cache = new int[n + 1];
        cache[1] = 1;
        for (int i = 2; i <= n; i++) {
            cache[i] = cache[i-1] + cache[i-2];
        }
        return cache[n];
        
    }
}


//REVERSE WORD IN A STRING
class Solution {
    public String reverseWords(String s) {
        
        String[] wordsArr = s.split(" ");
        StringBuilder result = new StringBuilder();
        
        for (String word: wordsArr) {
            result.append(new StringBuffer(word).reverse().toString() + " ");
        }
        return result.toString().trim();
    }
}

//RESHAPE MATRIX USING QUEUE
class Solution {
    public int[][] matrixReshape(int[][] nums, int r, int c) {
        int[][] res = new int[r][c];
        if (nums.length == 0 || r * c != nums.length * nums[0].length)
            return nums;
        Queue<Integer> queue = new LinkedList();
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums[0].length; j++) {
                queue.add(nums[i][j]);
            }
        }
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                res[i][j] = queue.remove();
            }
        }
        return res;
    }
}

//TWO SUM IN BST
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */

function findTarget(root: TreeNode | null, k: number): boolean {
    const set = new Set<number>();

  const queue: (TreeNode | null)[] = [root];

  while (queue.length) {
    const curr = queue.shift()!;
    if (curr) {
      if (set.has(k - curr.val)) {
        return true;
      }
      set.add(curr.val);
      queue.push(curr.left, curr.right);
    }
  }

  return false;
};

//SEARCH IN BINARY SEARCH TREE
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */

function searchBST(root: TreeNode | null, val: number): TreeNode | null {
    let cur = root;
    while(cur) {
        if(val < cur.val) {
            cur = cur.left;
        } else if(val > cur.val) {
            cur = cur.right;
        } else {
            return cur
        }
    }
    return null;

};

//BINARY SEARCH
class Solution {
    public int search(int[] nums, int target) {
        
        int start = 0;
        int end = nums.length - 1;
        int mid;
        
        int i = 0;
        while (start <= end) {
            mid = start + (end - start) / 2;
            if (nums[mid] == target) return mid;
            if (target < nums[mid]) end = mid - 1;
            else start = mid + 1;
        }
        return -1;
    }
}


//SMALLEST LETTER GREATER THAN TARGET
function nextGreatestLetter(letters: string[], target: string): string {
    
    let start = 0;
    let end = letters.length;
    
    while (start < end) {
        
        let mid = start + Math.floor((end - start)/2);
        
        if (letters[mid] <= target) start = mid + 1;
        else end = mid;
    }
    return letters[start % letters.length];

};

//Min Cost Climbing Stairs - DP CACHE
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int[] cache = new int[cost.length+1];
        for (int i = 2; i < cache.length; i++) {
            cache[i] = Math.min((cache[i-1] + cost[i-1]), (cache[i-2] + cost[i-2]));
        }
        return cache[cache.length-1];
    }
}


//NTH TRIBONACCI
class Solution {
    public int tribonacci(int n) {
        
     
        
        int[] cache = new int[38];
        cache[1] = 1;
        cache[2] = 1;
        for (int i = 3; i <= n; i++) {
            cache[i] = cache[i-1] + cache[i-2] + cache[i-3];
        }
        return cache[n];
    }
}


//The K Weakest Rows in a Matrix - JAVA Priority Queue
class Solution {
    public int[] kWeakestRows(int[][] mat, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] != b[0] ? b[0] - a[0] : b[1] - a[1]);
        int[] ans = new int[k];
        
        for (int i = 0; i < mat.length; i++) {
            pq.offer(new int[] {numOnes(mat[i]), i});
            if (pq.size() > k)
                pq.poll();
        }
        
        while (k > 0)
            ans[--k] = pq.poll()[1];
        
        return ans;
    }
    
    private int numOnes(int[] row) {
        int lo = 0;
        int hi = row.length;
        
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            
            if (row[mid] == 1)
                lo = mid + 1;
            else
                hi = mid;
        }
        
        return lo;
    }
}

//NUMBER OF STEPS TO REDUCE A NUMBER TO ZERO
class Solution {

    // Notes:
    // & is AND Operation (1 AND 1 is 1, 1 AND 0 is 0, 0 AND 0 is 0)
    // num & 1 == 1 meaning odd, == 0 meaning even.
    // Example:
    // n = 15 or 1111. n & 0001 = 0001
    // n = 8 or 1000. n & 0001 = 0000.
    //
    // ^ is XOR Operation (1 OR 1 is 0, 1 OR 0 is 1, 0 OR 0 is 0)
    // num ^ 1 is num - 1 if num is odd, or num + 1 if num is even.
    // We only use num ^ 1 when num is odd.
    // Example:
    // n = 15 or 1111. n ^ 0001 = 1110 (14)
    // n = 8 or 1000. n ^ 0001 = 1001 (9)
    //
    // >> is SHIFT RIGHT Operation, the number is the number of bits moved (moving the whole binary one bit right).
    // num >> 1 is num / 2 if num is even. If num is odd, then is (num - 1) / 2.
    // Example:
    // n = 15 or 1111. n >> 1 = 0111 (7)
    // n = 8 or 1000. n >> 1 = 0100 (4)

    public int numberOfSteps(int num) {
        int count = 0;

        while (num > 0) {
            num = (num & 1) == 1 ? num ^ 1 : num >> 1;
            count++;
        }
        return count;
    }
}


// Sort Integers by The Number of 1 Bits
// JAVA PRIORITY QUEUE
class Solution {
    public int[] sortByBits(int[] arr) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if(a[1] == b[1]){
                return a[0] - b[0];
            }else{
                return a[1] - b[1];
            }
        });
        for(int i = 0; i < arr.length; i++){
            int[] newArr = new int[2];
            newArr[0] = arr[i];
            newArr[1] = Integer.bitCount(arr[i]);
            pq.offer(newArr);
        }
        int[] ans = new int[arr.length];
        int i = 0;
        while(!pq.isEmpty()){
            int[] curr = pq.poll();
            ans[i++] = curr[0];
        }
        return ans;
    }
}


//REMOVE DUBLICATES
class Solution {
    public int removeDuplicates(int[] nums) {
        int insertIndex = 1;
        for(int i = 1; i < nums.length; i++){
            // We skip to next index if we see a duplicate element
            if(nums[i - 1] != nums[i]) {
                /* Storing the unique element at insertIndex index and incrementing
                   the insertIndex by 1 */
                nums[insertIndex] = nums[i];     
                insertIndex++;
            }
        }
        return insertIndex;
        
    }
}

// TWO SUM
class Solution {
    public int[] twoSum(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[j] == target - nums[i]) {
                    return new int[] { i, j };
                }
            }
        }
        // In case there is no solution, we'll just return null
        return null;
        
    }
}

//ROMAN TO INTEGER
class Solution {
    public int romanToInt(String s) {
        
        HashMap<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("V", 5);
        map.put("X", 10);
        map.put("L", 50);
        map.put("C", 100);
        map.put("D", 500);
        map.put("M", 1000);
        
        int sum = 0;
        int i = 0;
        while(i < s.length()) {
            String currentSymbol = s.substring(i , i + 1);
            int currentValue = map.get(currentSymbol);
            int nextValue = 0;
            if (i + 1 < s.length()) {
                String nextSymbol = s.substring(i + 1, i + 2);
                nextValue = map.get(nextSymbol);
            }
            
            if (currentValue < nextValue) {
                sum += nextValue - currentValue;
                i += 2;
            }
            else {
                sum += currentValue;
                i += 1;
            }
        }
        return sum;
    }
}

//LONGEST COMMON PREFIX
class Solution {
    public String longestCommonPrefix(String[] strs) {
        
        for(int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            
            for(int j = 1; j < strs.length; j++) {
                
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
            
        }
        return strs[0];
    }
}

//VALID PARANTHESIS
class Solution {
    public boolean isValid(String s) {
        //Time: O(n)
        //Space: O(n)
        
        HashMap<Character, Character> map = new HashMap<>();
        
        map.put('(',')');
        map.put('{','}');
        map.put('[',']');
        
        Stack<Character> stack = new Stack<>();
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            
            
            //decide if it is open
            if (map.containsKey(c)) {
                stack.push(c);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                char open = stack.pop();
                if (map.get(open) != c) {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }
}

//MERGE TWO SORTED LISTS
function mergeTwoLists(list1: ListNode | null, list2: ListNode | null): ListNode | null {
    if (list1 === null) {
        return list2;
    }
    if (list2 === null) {
        return list1;
    }
    if (list1.val < list2.val) {
        return new ListNode(list1.val, mergeTwoLists(list1.next, list2));
    }
    return new ListNode(list2.val, mergeTwoLists(list1, list2.next));
};

//REMOVE DUBLICATE FROM SORTED ARRAY
class Solution {
    public int removeDuplicates(int[] nums) {
        int insertIndex = 1;
        for(int i = 1; i < nums.length; i++){
            // We skip to next index if we see a duplicate element
            if(nums[i - 1] != nums[i]) {
                /* Storing the unique element at insertIndex index and incrementing
                   the insertIndex by 1 */
                nums[insertIndex] = nums[i];     
                insertIndex++;
            }
        }
        return insertIndex;
        
    }
}

//SEARCH INSERT POSTIOON
function searchInsert(nums: number[], target: number): number {
    
    let pivot = 0;
    let start = 0;
    let end = nums.length - 1;
    
    while (start <= end) {
        let pivot = start + Math.floor((end - start)/2);
        
        if (nums[pivot] === target) return pivot;
        if (target < nums[pivot]) end = pivot - 1;
        else start = pivot + 1;
    }
    return start;

};

//SQRT(X)
var mySqrt = function(x) {
    let sqrt = 1;
    if(x===0){
        return 0;
    }
    for(let i=1; i*i<=x; i++){
      sqrt = i;
    }
    return sqrt;
};

//CLIMBING STAIRS
class Solution {
    public int climbStairs(int n) {
        if (n == 1) return 1;
        
        int[] cache = new int[n+1];
        
        cache[1] = 1;
        cache[2] = 2;
        
        for (int i = 3; i <= n; i++) {
            cache[i] = cache[i-1] + cache[i-2];
        }
        return cache[n];
    }
}

//REMOVE DUBLICATES FROM SORTED LIST
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     val: number
 *     next: ListNode | null
 *     constructor(val?: number, next?: ListNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.next = (next===undefined ? null : next)
 *     }
 * }
 */

function deleteDuplicates(head: ListNode | null): ListNode | null {
    // RECURSE THROUGH THE LIST
    const readAndChange = (list: ListNode, head: ListNode): ListNode => {
	
		// IF WE REACH THE END
        if (!list) return head;

		// IF THERE IS A DUPLICATE
        else if (list.next && list.val === list.next.val) {
			// REASSIGN NEXT
            list.next = list.next.next;
            return readAndChange(list, head);
        } 
		// IF NO DUPLICATE MOVE TO THE NEXT
		else return readAndChange(list.next, head);
    }
    return readAndChange(head, head);
};

//MERGE SORTED ARRAY
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        for (int i = 0; i < n; i++) {
            nums1[i + m] = nums2[i];
        }
        Arrays.sort(nums1);       
    }
}

//Binary Tree Inorder Traversal
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */
function inorderTraversal(root: TreeNode | null): number[] {
  if (!root) return [];

  return [
    ...inorderTraversal(root.left),
    root.val,
    ...inorderTraversal(root.right),
  ];
}

//Symmetric Tree
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */
function isSymmetric(root: TreeNode | null): boolean {
    let queue = [];
    queue.push(root)
    while(queue.length) {
        let cur = [];
        let n = queue.length;
        for(let i = 0; i < n; i++) {
            let node = queue.shift();
            if(node.left) queue.push(node.left);
            if(node.right) queue.push(node.right);
            cur.push(node.left,node.right);
        }
        if(cur.length%2 !== 0) return false;
        for(let i = 0; i < cur.length/2; i++) {
            let j = cur.length-1-i;
            if(cur[i] == null && cur[j] == null) continue;
            if((cur[i] == null && cur[j] !== null) || (cur[j] == null && cur[i] !== null)) return false;
            if(cur[i].val !== cur[j].val) return false;
        }
    }
    return true;

};

//MAX DEPTH OF BINARY TREE
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */

function maxDepth(root: TreeNode | null): number {
    
    let treeStack = [{ node: root, depth: 1 }];
    let current = treeStack.pop();
    let max = 0;
    while (current && current.node) {
        let node = current.node;
        if (node.left) {
            treeStack.push({ node: node.left, depth: current.depth + 1 });
        }
        if (node.right) {
            treeStack.push({ node: node.right, depth: current.depth + 1 });
        }
        if (current.depth > max) {
            max = current.depth;
        }
        current = treeStack.pop();
    }
    return max;
};

//PATH SUM
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */

function hasPathSum(root: TreeNode | null, sum: number): boolean {
  if (!root) return false;
  const nextSum = sum - root.val;
  return (
    (nextSum === 0 && isLeaf(root)) ||
    hasPathSum(root.left, nextSum) ||
    hasPathSum(root.right, nextSum)
  );
}
const isLeaf = (node: TreeNode | null): boolean => {
  return Boolean(node && !node.left && !node.right);
};


//MIDDLE OF LINKED LIST
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode middleNode(ListNode head) {
        
        ListNode[] newNode = new ListNode[100];
        int i = 0;
        while(head != null) {
            newNode[i++] = head;
            head = head.next;
        }
        return newNode[i/2];
        
        
    }
}


//INTERSECTION OF TWO LINKED LISTS
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        
        Set<ListNode> nodesInA = new HashSet<ListNode>();

        while (headA != null) {
            nodesInA.add(headA);
            headA = headA.next;
        }
        
        System.out.println(nodesInA);
        
        while (headB != null) {
            System.out.println(headB);
            if (nodesInA.contains(headB)) {
                return headB;
            }
            headB = headB.next;
        }
        return null;
    }
}


//REVERSE BITS
public class Solution {
    
    public int reverseBits(int num) {
        
        num = ((num & 0xffff0000) >>> 16) | ((num & 0x0000ffff) << 16);
        num = ((num & 0xff00ff00) >>> 8) | ((num & 0x00ff00ff) << 8);
        num = ((num & 0xf0f0f0f0) >>> 4) | ((num & 0x0f0f0f0f) << 4);
        num = ((num & 0xcccccccc) >>> 2) | ((num & 0x33333333) << 2);
        num = ((num & 0xaaaaaaaa) >>> 1) | ((num & 0x55555555) << 1);
        
        return num;
        
    }
}

public int reverseBits(int n) {
    if (n == 0) return 0;
    
    int result = 0;
    for (int i = 0; i < 32; i++) {
        result <<= 1;
        if ((n & 1) == 1) result++;
        n >>= 1;
    }
    return result;
}


//Number of 1 bits
public class Solution {
	public int hammingWeight(int n) {
		int count=0;
		String str=Integer.toBinaryString(n);
		for(int i=0;i<str.length();i++){
			if(str.charAt(i)=='1')
				count++;
		}
		return count;
	}
}


//REMOVE LINKED LIST ELEMENTS
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     val: number
 *     next: ListNode | null
 *     constructor(val?: number, next?: ListNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.next = (next===undefined ? null : next)
 *     }
 * }
 */

function removeElements(head: ListNode | null, val: number): ListNode | null {
    if (!head) return null;
    let current = head;
    while (current.next) {
        if (current.next.val == val) {
            current.next = current.next.next;
        } else {
            current = current.next;
        }
    }
    return head.val == val ? head.next : head;

};

// REVERSE LINKED LISTS
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     val: number
 *     next: ListNode | null
 *     constructor(val?: number, next?: ListNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.next = (next===undefined ? null : next)
 *     }
 * }
 */

function reverseList(head: ListNode | null): ListNode | null {
    let [curr, prev, next] = [head, null, null];

    while (curr) { 
        next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next
    }
    return prev
};

//PALINDROME LINKED LIST
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public boolean isPalindrome(ListNode head) {
        ArrayList<Integer> arrayList = new ArrayList<>();
        //Convert into arrayList
        ListNode currentNode = head;
        while(currentNode != null) {
            arrayList.add(currentNode.val);
            currentNode = currentNode.next;
        }
        
        int start = 0;
        int end = arrayList.size() - 1;
        
        while (start < end) {
            if (arrayList.get(start) != arrayList.get(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true; 
    }
}


//CONTAIN DUBLICATES
class Solution {
    public boolean containsDuplicate(int[] nums) {
        
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i + 1]) return true;
        }
        
        return false;
    }
}

//INVERT TREE
/**
 * Definition for a binary tree node.
 * class TreeNode {
 *     val: number
 *     left: TreeNode | null
 *     right: TreeNode | null
 *     constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
 *         this.val = (val===undefined ? 0 : val)
 *         this.left = (left===undefined ? null : left)
 *         this.right = (right===undefined ? null : right)
 *     }
 * }
 */

function invertTree(root: TreeNode | null): TreeNode | null {
    if (root === null) return null;
	[root.left, root.right] = [root.right, root.left];
	invertTree(root.left);
	invertTree(root.right);
	return root;

};

//POWER OF THW SOLVED BY BIT
/* the binary form of everypow(2, n) - 1 likes0b11..1
1 - 1 = 0 = 0b0        =>  1 & 0 = 0b1    & 0b0    = 0
2 - 1 = 1 = 0b1        =>  2 & 1 = 0b10   & 0b1    = 0
4 - 1 = 3 = 0b11       =>  4 & 3 = 0b100  & 0b11   = 0
8 - 1 = 7 = 0b111      =>  8 & 7 = 0b1000 & 0b111  = 0
...
so we can find pow(2, n) & (pow(2, n) - 1) == 0
for example, num = 4 = 0b100
4 - 1 = 3 = 0b11
4 & 3 = 0b100 & 0b11 = 0 */
class Solution {
    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n & n - 1) == 0;
    }
}


//VALID ANAGRAM
class Solution
{
public boolean isAnagram(String s, String t)
{
char arr[] = (s.toLowerCase()).toCharArray();
char brr[] = (t.toLowerCase()).toCharArray();

    Arrays.sort(arr);
    Arrays.sort(brr);

    if(Arrays.equals(arr,brr))
    {
        return true;
    }
    else 
    {
        return false;
    }
}
}

//MISSING NUMBER
class Solution {
    public int missingNumber(int[] nums) {
        
        int n = nums.length;
        Arrays.sort(nums);
        
   
        if (nums[nums.length-1] != nums.length) {
            return nums.length;
        }

        else if (nums[0] != 0) {
            return 0;
        }
        
        for (int i = 1; i < n; i++) {
            
            if (nums[i] != nums[i-1] + 1)
                return nums[i-1] + 1;
        }
        return -1;  
    }
}

//MOVE ZEROS
class Solution {
    public void moveZeroes(int[] nums) {
        
        int i=0;
        int n=nums.length;
        for(int j=0; j<n; j++){
            if(nums[j] != 0){
                nums[i++] = nums[j];
            }
        }
        for(int k=i;k<n;k++){
            nums[k] = 0;
        }
        
    }
}

//POWER OF THREE
class Solution {
    public boolean isPowerOfThree(int n) {
        if (n < 1) {
            return false;
        }
        while ( n % 3 == 0) {
            n = n/3;
        }
        return n == 1;
    }
}

//REVERSE STRING
class Solution {
    public void reverseString(char[] s) {
        int N = s.length - 1;
        int start = 0;
        int end = N;
        while(start < end) {
            char temp = s[start];
            s[start] = s[end];
            s[end] = temp;
            start++;
            end--;
        }  
    }
}

// INTERSECTION OF TWO ARRAYS
// Runtime: 1 ms, faster than 99.13% of Java online submissions for Intersection of Two Arrays II.
// Memory Usage: 42.5 MB, less than 92.71% of Java online submissions for Intersection of Two Arrays II.
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        // Sort both the arrays first...
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        // Create an array list...
        ArrayList<Integer> arr = new ArrayList<Integer>();
        // Use two pointers i and j for the two arrays and initialize both with zero.
        int i = 0, j = 0;
        while(i < nums1.length && j < nums2.length){
            // If nums1[i] is less than nums2[j]...
            // Leave the smaller element and go to next(greater) element in nums1...
            if(nums1[i] < nums2[j]) {
                i++;
            }
            // If nums1[i] is greater than nums2[j]...
            // Go to next(greater) element in nums2 array...
            else if(nums1[i] > nums2[j]){
                j++;
            }
            // If both the elements intersected...
            // Add this element to arr & increment both i and j.
            else{
                arr.add(nums1[i]);
                i++;
                j++;
            }
        }
        // Create a output list to store the output...
        int[] output = new int[arr.size()];
        int k = 0;
        while(k < arr.size()){
            output[k] = arr.get(k);
            k++;
        }
        return output;
    }
}


//PERFECT SQUARE
public boolean isPerfectSquare(int num) {
     int i = 1;
     while (num > 0) {
         num -= i;
         i += 2;
     }
     return num == 0;
 }
// log(n)
 public boolean isPerfectSquare(int num) {
        int low = 1, high = num;
        while (low <= high) {
            long mid = (low + high) >>> 1;
            if (mid * mid == num) {
                return true;
            } else if (mid * mid < num) {
                low = (int) mid + 1;
            } else {
                high = (int) mid - 1;
            }
        }
        return false;
    }

//newton method
public boolean isPerfectSquare(int num) {
    long x = num;
    while (x * x > num) {
        x = (x + num / x) >> 1;
    }
    return x * x == num;
}

//GUESS THE NUMBER HIGHER IR LOWER
public int guessNumber(int n) {
        int low = 1;
        int high = n;
        
        while(low<=high){
            int mid = low + (high - low) / 2;
            
            if(guess(mid) == 1) low = mid + 1;
            else if(guess(mid) == -1) high = mid - 1;
            else return mid;
        }
        return 1;
    }

//RANSOM NOTE: Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.
//Using Array
public boolean canConstruct(String ransomNote, String magazine) {
        int a[] = new int[26]; // find occurence of each character in string magazine
        for (char i : magazine.toCharArray()) {
            a[i - 'a']++;
        }
        for (char i : ransomNote.toCharArray()) {
            if (a[i - 'a'] == 0) { // character is not found in magazine or a particular character doesn't have same or greater count than count in magazine
                return false;
            } else {
                a[i - 'a']--; // decrement if character exists
            }
        }
        return true;
    }

//Using HashMap
public boolean canConstruct(String ransomNote, String magazine) {
        HashMap<Character, Integer> hm = new HashMap<>();
        for (char i : magazine.toCharArray()) {
            if (hm.containsKey(i)) {
                hm.put(i, hm.get(i) + 1);
            } else {
                hm.put(i, 1);
            }
        }
        for (char i : ransomNote.toCharArray()) {
            if (!hm.containsKey(i) || hm.get(i) == 0) {
                return false;
            } else {
                hm.put(i, hm.get(i) - 1);
            }
        }
        return true;
    }

//FIRST UNIQUE CHARACTER IN A STRING
class Solution {
    public int firstUniqChar(String s) {
        HashMap<Character,Integer> map = new HashMap<>();//Creating a hashmap which will take avery character and note its occurrence.
        for(int i = 0 ; i < s.length() ; i++){
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i),0)+1);
        }
        for(int i = 0 ; i < s.length() ; i++){
            if(map.get(s.charAt(i))==1) return i;//If the occurrence of any char is once, then it is the required ans
        }
        return -1;//else returning -1
    }
}

// LCS WITH 2D ARRAY TABLE
/* Returns length of LCS for X[0..m-1], Y[0..n-1] */
function isSubsequence(s: string, t: string): boolean {
    let m = s.length;
    let n = t.length;
    return lcs(s, t, m-1, n-1);;
};

function lcs(X, Y, m, n): boolean {
    if(m < 0) return true;
    if(n < 0) return false;
    if (X.charAt(m) === Y.charAt(n)) {
        return lcs(X, Y, m - 1, n - 1);
    }
    return lcs(X,Y,m,n-1);
};

function lcs(X, Y, m, n, dp) // might have problems
{
    if (m == 0 || n == 0)
        return 0;
    if (X[m - 1] == Y[n - 1])
        return dp[m][n] = 1 + lcs(X, Y, m - 1, n - 1, dp);
  
    if (dp[m][n] != -1) {
        return dp[m][n];
    }
    return dp[m][n] = Math.max(lcs(X, Y, m, n - 1, dp),
                          lcs(X, Y, m - 1, n, dp));
}

// IS SUBSEQUENCE
class Solution {
    public boolean isSubsequence(String s, String t) {
        int i = 0, j = 0;
        while(i < s.length() && j < t.length()){
            if(s.charAt(i) == t.charAt(j)) i++;
            j++;
        }
        return i == s.length();
    }
}

// SUM OF THE LEFT LEAVES
class Solution {
    public int sumOfLeftLeavesHelper(TreeNode root,boolean isLeft)
    {
        // If the tree is empty, sum is 0.
        if(root==null)
            return 0;
        // If current node is leaf node and if it is left node, then return its value
        if(root.left==null && root.right==null && isLeft==true)
         return root.val;
        // Perform the same on left and right subtrees, and return their sum. Whenever calling left subtree, pass true as isLeft, because its on left.
        return sumOfLeftLeavesHelper(root.left,true)+sumOfLeftLeavesHelper(root.right,false);
    }
    public int sumOfLeftLeaves(TreeNode root) {
        // If the tree is empty, sum is 0.
        if(root==null)
            return 0;
        // Perform the same on left and right subtrees, and return their sum. Whenever calling left subtree, pass true as isLeft, because its on left.
        return sumOfLeftLeavesHelper(root.left,true)+sumOfLeftLeavesHelper(root.right,false); 
     
    }
}

// IF ONE SWAP IS ENOUGH FOR EQUAL STRING
class Solution {
    public boolean areAlmostEqual(String s1, String s2) {
        List<Integer> l = new ArrayList<>();
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) != s2.charAt(i)) l.add(i);
			if (l.size() > 2) return false; // added this line to short circuit the loop
        }
        return l.size() == 0 || (l.size() == 2
                                 && s1.charAt(l.get(0)) == s2.charAt(l.get(1))
                                 && s1.charAt(l.get(1)) == s2.charAt(l.get(0)));
    }
}

//SUM OF ODD LENGTHS OF AN ARRAY
public int sumOddLengthSubarrays(int[] A) {
        int res = 0, n = A.length;
        for (int i = 0; i < n; ++i) {
            res += ((i + 1) * (n - i) + 1) / 2 * A[i];
        }
        return res;
    }

// KTH MISSING NUMBER IN AN ARRAY
public int findKthPositive(int[] A, int k) {
        int l = 0, r = A.length, m;
        while (l < r) {
            m = (l + r) / 2;
            if (A[m] - 1 - m < k)
                l = m + 1;
            else
                r = m;
        }
        return l + k;
    }


// SHUFFELED STRING
class Solution {
    public String restoreString(String s, int[] indices) {
        
        StringBuilder chArr = new StringBuilder(s);

        for(int i = 0; i < s.length(); i++){
            chArr.setCharAt(indices[i], s.charAt(i));
        }
        return chArr.toString();
    }
}































