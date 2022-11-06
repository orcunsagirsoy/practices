//Rotting Oranges
//BEST TIME TO SELL AND BUY PROBLEMS
//Matrix Block Sum
//Maximum Distance Between a Pair of Values
//Count Sub Islands
//Coin change II
//Pacific Atlantic Water Flow
//Wiggle Subsequence
//Number of Islands
//House Robber
//Rotate Array
//Find Minimum in Rotated Sorted Array
//Search a 2D Matrix
//Minimum Path Sum
//Unique Paths II
//Jump Game
//Search in Rotated Sorted Array
//Longest Palindromic Substring
//Longest Substring Without Repeating Characters

//2D MATRIX SET ZEROS
function setZeroes(matrix: number[][]): void {
    let cols = [];
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            if (matrix[i][j] === 0) {
                cols.push(j);
                continue;
            }
            if (matrix[i].includes(0)) {
                matrix[i][j] = 0;
            }
        }
    }
    for (let m = 0; m < matrix.length; m++) {
        for (let n = 0; n < cols.length; n++) {
            matrix[m][cols[n]] = 0;
        }
    }
    
//GROUP ANAGRAM HASH TABLES
function groupAnagrams(strs: string[]): string[][] {
    let strMap = {};
    strs.map(str => {
        console.log(str.split("").sort().join(''));
        let sortedStr = str.split("").sort().join('');
        if (strMap[sortedStr]) {
            strMap[sortedStr].push(str);
        } else {
            strMap[sortedStr] = [str];
        }
    })
    console.log(strMap);
    return Object.values(strMap);
};

//LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS
//Sliding window
public int lengthOfLongestSubstring(String s) {
	int n = s.length(), longest = 0;
	int[] nextIndex = new int[128]; 

	for (int r=0, l=0; r<n; r++) {
		l = Math.max(nextIndex[s.charAt(r)], l); 
		longest = Math.max(longest, r - l + 1);
		nextIndex[s.charAt(r)] = r + 1;
	}

	return longest;
}

//LONGEST PALINDROMIC SUBSTRING
public String longestPalindrome(String s) {
	int n = s.length(), start = 0, end = 0;
	boolean[][] dp = new boolean[n][n];

	for (int len=0; len<n; len++) {
		for (int i=0; i+len<n; i++) {
			dp[i][i+len] = s.charAt(i) == s.charAt(i+len) && (len < 2 || dp[i+1][i+len-1]);
			if (dp[i][i+len] && len > end - start) {
				start = i;
				end = i + len;
			}
		}
	}

	return s.substring(start, end + 1);
}


//Remove Nth Node From End of List
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
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode newNode = new ListNode(0);
        newNode.next = head;
        ListNode first = newNode;
        ListNode second = newNode;
        
        for (int i = 1; i <= n + 1; i++) {
            first = first.next;
        }
        
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return newNode.next;
    }
}


// VALID SUDOKU
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int N = 9;

        // Use hash set to record the status
        HashSet<Character>[] rows = new HashSet[N];
        HashSet<Character>[] cols = new HashSet[N];
        HashSet<Character>[] boxes = new HashSet[N];
        for (int r = 0; r < N; r++) {
            rows[r] = new HashSet<Character>();
            cols[r] = new HashSet<Character>();
            boxes[r] = new HashSet<Character>();
        }
        
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                char val = board[r][c];

                // Check if the position is filled with number
                if (val == '.') {
                    continue;
                }

                // Check the row
                if (rows[r].contains(val)) {
                    return false;
                }
                rows[r].add(val);

                // Check the column
                if (cols[c].contains(val)) {
                    return false;
                }
                cols[c].add(val);

                // Check the box
                int idx = (r / 3) * 3 + c / 3;
                if (boxes[idx].contains(val)) {
                    return false;
                }
                boxes[idx].add(val);
            }
        }
        return true;
    }
}



//PERMUTATIONS
//Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.
public List<List<Integer>> permute(int[] nums) {
   List<List<Integer>> list = new ArrayList<>();
   // Arrays.sort(nums); // not necessary
   backtrack(list, new ArrayList<>(), nums);
   return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums){
   if(tempList.size() == nums.length){
      list.add(new ArrayList<>(tempList));
   } else{
      for(int i = 0; i < nums.length; i++){ 
         if(tempList.contains(nums[i])) continue; // element already exists, skip
         tempList.add(nums[i]);
         backtrack(list, tempList, nums);
         tempList.remove(tempList.size() - 1);
      }
   }
}



//MAXIMUM SUBARRAY
class Solution {
    public int maxSubArray(int[] nums) {
        
        int currentSubarray = nums[0];
        int maxSubarray = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            int num = nums[i];
            currentSubarray = Math.max(num, currentSubarray + num);
            maxSubarray = Math.max(maxSubarray, currentSubarray);
        }
        
        
        
        return maxSubarray;
    }
}

//**** MINIMUM PATH SUM ****
//Recursıon
public static int minPathSum(int[][] grid) {

            int height = grid.length;
            int width = grid[0].length;
            return min(grid, height - 1, width - 1);
			
        }
		
public static int min(int[][]grid, int row, int col){

            if(row == 0 && col == 0) return grid[row][col]; // this is the exit of the recursion
            if(row == 0) return grid[row][col] + min(grid, row, col - 1); /** when we reach the first row, we could only move horizontally.*/
            if(col == 0) return grid[row][col] + min(grid, row - 1, col); /** when we reach the first column, we could only move vertically.*/
            return grid[row][col] + Math.min(min(grid, row - 1, col), min(grid, row, col - 1)); /** we want the min sum path so we pick the cell with the less value */
			
}
//DP
public static int minPathSum(int[][] grid) {

            int height = grid.length;
            int width = grid[0].length;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    if(row == 0 && col == 0) grid[row][col] = grid[row][col];
                    else if(row == 0 && col != 0) grid[row][col] = grid[row][col] + grid[row][col - 1];
                    else if(col == 0 && row != 0) grid[row][col] = grid[row][col] + grid[row - 1][col];
                    else grid[row][col] = grid[row][col] + Math.min(grid[row - 1][col], grid[row][col - 1]);
                }
            }
            return grid[height - 1][width - 1];
        }





//Search a 2D Matrix
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        if (m == 0)
            return false;
        int n = matrix[0].length;

        // binary search
        int left = 0, right = m * n - 1;
        int pivotIdx, pivotElement;
        while (left <= right) {
            pivotIdx = (left + right) / 2;
            pivotElement = matrix[pivotIdx / n][pivotIdx % n]; //finding the middle el
            if (target == pivotElement)
                return true;
            else {
                if (target < pivotElement)
                    right = pivotIdx - 1;
                else
                    left = pivotIdx + 1;
            }
        }
        return false;
        
    }
}

//Combinations
//Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n]. - BACKTRACKING
public static List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> combs = new ArrayList<List<Integer>>();
		combine(combs, new ArrayList<Integer>(), 1, n, k);
		return combs;
	}
	public static void combine(List<List<Integer>> combs, List<Integer> comb, int start, int n, int k) {
		if(k==0) {
			combs.add(new ArrayList<Integer>(comb));
			return;
		}
		for(int i=start;i<=n;i++) {
			comb.add(i);
			combine(combs, comb, i+1, n, k-1);
			comb.remove(comb.size()-1);
		}
	}

//VALIDATE BINARY SEARCH TREE
class Solution {
    public boolean isValidBST(TreeNode root) {
      return checker(root, null, null);  
    }
    public boolean checker(TreeNode root, Integer min, Integer max) {
        if(root == null) return true;

        if((min != null && root.val <= min) || (max != null && root.val >= max))
        return false;

        return checker(root.left, min, root.val) && checker(root.right, root.val, max);
    }
}

//Find Peak Element
class Solution {
    public int findPeakElement(int[] arr) {
        int start = 0;
        int end =  arr.length - 1;
        while(start < end){
            int mid = start + (end - start)/2;
            if(arr[mid] > arr[mid+1] ){
                //we are in dec part of the array
                end = mid;
            }
            else{//we are in inc part of array
                start = mid + 1;
            }
        }
        return start;
    }
}

//HOUSE ROBBER
class Solution {
    
    public int rob(int[] nums) {
        
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        int max1 = rob_simple(nums, 0, nums.length - 2);
        int max2 = rob_simple(nums, 1, nums.length - 1);
        
        return Math.max(max1, max2);
    }
    
    public int rob_simple(int[] nums, int start, int end) {
            int t1 = 0;
            int t2 = 0;
            
            for (int i = start; i <= end; i++) {
                int temp = t1;
                int current = nums[i];
                t1 = Math.max(current + t2, t1);
                t2 = temp;
            }
            return t1;
        }
}



//MAXIAML SQUARE
//Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
//https://leetcode.com/problems/maximal-square/discuss/600149/Python-Thinking-Process-Diagrams-DP-Approach


//PERFECT SQUARES - DP
class Solution {
    public int numSquares(int n) {
        int dp[]=new int[n+1];
        Arrays.fill(dp,n);
        dp[0]=0;
        for(int i=0;i<n+1;i++){
            for(int j=1;j<i+1;j++){
                int square=j*j;
                if(i-square<0){
                    break;
                }else{
                    dp[i]=Math.min(dp[i],1+dp[i-square]);
                }
            }
        }
        return dp[n];
    }
}

//LONGEST INCREASING SUBSEQUENCE
public int lengthOfLIS(int[] nums) {
    int[] tails = new int[nums.length];
    int size = 0;
    for (int x : nums) {
        int i = 0, j = size;
        while (i != j) {
            int m = (i + j) / 2;
            if (tails[m] < x)
                i = m + 1;
            else
                j = m;
        }
        tails[i] = x;
        if (i == size) ++size;
    }
    return size;
}

//COMBINATION SUM IV
class Solution {
    public int combinationSum4(int[] nums, int target) {
        Integer[] memo = new Integer[target + 1];
        return recurse(nums, target, memo);
    }
    
    public int recurse(int[] nums, int remain, Integer[] memo){
        
        if(remain < 0) return 0;
        if(memo[remain] != null) return memo[remain];
        if(remain == 0) return 1;
        
        int ans = 0;
        for(int i = 0; i < nums.length; i++){
            ans += recurse(nums, remain - nums[i], memo);
        }
        
        memo[remain] = ans;
        return memo[remain];
    }
}

//Find All Duplicates in an Array
//Given an integer array nums of length n where all the integers of nums are in the range [1, n] and each integer appears once or twice, return an array of all the integers that appears twice.
//You must write an algorithm that runs in O(n) time and uses only constant extra space.
class Solution 
{
    public List<Integer> findDuplicates(int[] nums) 
    {
        List <Integer> result = new ArrayList <>();
        
        for (int index = 0; index < nums.length; index++)
		{
            int newIndex = Math.abs (nums [index]) - 1;
			
            if (nums [newIndex] < 0) {
                result.add (newIndex + 1);
            }
            nums [newIndex] = (-1) * nums [newIndex];
        }
        return result;
    }
}

//Longest Palindromic Subsequence
public int longestPalindromeSubseq(String s) {
        int n = s.length();
        String reverse = "";
        
        for (int i = n - 1;  i >= 0; i--) {
            reverse += s.charAt(i) + "";
        }
        
        int[][] dp = new int[n+1][n+1];
        
        for (int i = 1; i < n+1; i++) {
            for (int j = 1; j < n+1; j++) {
                if (s.charAt(i-1) == reverse.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                }
            }
        }
        
        return dp[n][n];
    }


//Permutation in String
//Java Sliding Window
public class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int len1 = s1.length(), len2 = s2.length();
        if (len1 > len2) return false;
        
        int[] count = new int[26];
        for (int i = 0; i < len1; i++) {
            count[s1.charAt(i) - 'a']++;
            count[s2.charAt(i) - 'a']--;
        }
        if (allZero(count)) return true;
        
        for (int i = len1; i < len2; i++) {
            count[s2.charAt(i) - 'a']--;
            count[s2.charAt(i - len1) - 'a']++;
            if (allZero(count)) return true;
        }
        
        return false;
    }
    
    private boolean allZero(int[] count) {
        for (int i = 0; i < 26; i++) {
            if (count[i] != 0) return false;
        }
        return true;
    }
}

// NUMBER OF CLOSED ISLANDS
class Solution 
{
    public int closedIsland(int[][] grid)
    {
        // O(n * m) time 
        // O(n * m) space
        int count = 0;
        
        for(int i = 0; i < grid.length; i++)
        {
            for(int j = 0; j < grid[i].length; j++)
            {
                // trigger DFS once we visit land
                if(grid[i][j] == 0 && isClosed(grid, i, j))
                    count++;
            }
        }
        return count;
    }
    
    public boolean isClosed(int[][] grid, int i, int j)
    {
        if(i < 0 || i >= grid.length || j < 0 || j >= grid[0].length)
            return false;
        
        if(grid[i][j] == 1) 
            return true;
        
        // once we visited the land, we change it from land -> water
        grid[i][j] = 1;
        
        boolean up = isClosed(grid, i+1, j);
        boolean down = isClosed(grid, i-1, j);
        boolean left = isClosed(grid, i, j-1);
        boolean right = isClosed(grid, i, j+1);
        
        return (up && down && left && right);
    }
}



//*** Longest Common Subsequence*** */
//https://leetcode.com/problems/longest-common-subsequence/discuss/351689/JavaPython-3-Two-DP-codes-of-O(mn)-and-O(min(m-n))-spaces-w-picture-and-analysis
//https://en.m.wikipedia.org/wiki/Longest_common_subsequence_problem
public int longestCommonSubsequence(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        for (int i = 0; i < s1.length(); ++i)
            for (int j = 0; j < s2.length(); ++j)
                if (s1.charAt(i) == s2.charAt(j)) dp[i + 1][j + 1] = 1 + dp[i][j];
                else dp[i + 1][j + 1] =  Math.max(dp[i][j + 1], dp[i + 1][j]);
        return dp[s1.length()][s2.length()];
    }

//NUMBER OF ENCLAVES
class Solution {
    public int numEnclaves(int[][] A) {
        int result = 0;
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < A[i].length; j++) {
                if(i == 0 || j == 0 || i == A.length - 1 || j == A[i].length - 1)
                    dfs(A, i, j);
            }
        }
        
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < A[i].length; j++) {
                if(A[i][j] == 1)
                    result++;
            }
        }
        
        return result;
    }
    
    public void dfs(int a[][], int i, int j) {
        if(i >= 0 && i <= a.length - 1 && j >= 0 && j <= a[i].length - 1 && a[i][j] == 1) {
            a[i][j] = 0;
            dfs(a, i + 1, j);
            dfs(a, i - 1, j);
            dfs(a, i, j + 1);
            dfs(a, i, j - 1);
        }
    }
}


//Minimum Falling Path Sum
public int minFallingPathSum(int[][] A) {
  for (int i = 1; i < A.length; ++i)
    for (int j = 0; j < A.length; ++j)
      A[i][j] += Math.min(A[i - 1][j], Math.min(A[i - 1][Math.max(0, j - 1)], A[i - 1][Math.min(A.length - 1, j + 1)]));
  return Arrays.stream(A[A.length - 1]).min().getAsInt();
} 

//Maximum Sum Circular Subarray
//KADANE
class Solution {
    public int maxSubarraySumCircular(int[] nums) {
        
        int S = 0;  // S = sum(A)
        for (int x : nums) {
            S += x;
        }
        
        if (nums.length == 1) {
            return S;
        }

        int ans1 = kadane(nums, 0, nums.length-1, 1);
        int ans2 = S + kadane(nums, 1, nums.length-1, -1);
        int ans3 = S + kadane(nums, 0, nums.length-2, -1);
        return Math.max(ans1, Math.max(ans2, ans3));
    }

    public int kadane(int[] nums, int i, int j, int sign) {
        // The maximum non-empty subarray for array
        // [sign * A[i], sign * A[i+1], ..., sign * A[j]]
        int ans = Integer.MIN_VALUE;
        int cur = Integer.MIN_VALUE;
        for (int k = i; k <= j; ++k) {
            cur = sign * nums[k] + Math.max(cur, 0);
            ans = Math.max(ans, cur);
        }
        return ans;
        
    }
}

//LETTER CASE PERMUITATION
//BFS
class Solution {
    public List<String> letterCasePermutation(String S) {
        if (S == null) {
            return new LinkedList<>();
        }
        Queue<String> queue = new LinkedList<>();
        queue.offer(S);
        
        for (int i = 0; i < S.length(); i++) {
            if (Character.isDigit(S.charAt(i))) continue;            
            int size = queue.size();
            for (int j = 0; j < size; j++) {
                String cur = queue.poll();
                char[] chs = cur.toCharArray();
                
                chs[i] = Character.toUpperCase(chs[i]);
                queue.offer(String.valueOf(chs));
                
                chs[i] = Character.toLowerCase(chs[i]);
                queue.offer(String.valueOf(chs));
            }
        }
        
        return new LinkedList<>(queue);
    }
}

//DFS
class Solution {
    public List<String> letterCasePermutation(String S) {
        if (S == null) {
            return new LinkedList<>();
        }
        
        List<String> res = new LinkedList<>();
        helper(S.toCharArray(), res, 0);
        return res;
    }
    
    public void helper(char[] chs, List<String> res, int pos) {
        if (pos == chs.length) {
            res.add(new String(chs));
            return;
        }
        if (chs[pos] >= '0' && chs[pos] <= '9') {
            helper(chs, res, pos + 1);
            return;
        }
        
        chs[pos] = Character.toLowerCase(chs[pos]);
        helper(chs, res, pos + 1);
        
        chs[pos] = Character.toUpperCase(chs[pos]);
        helper(chs, res, pos + 1);
    }
}



//MAX AREA OF AN ISLAND - JAVA DFS MEMOIZATION
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int maxIslandSize = 0;
        int[] memo = new int[1]; // 1 element array to save island size with ref
		
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    memo[0] = 0; // remove old value
                    search(grid, i, j, memo);
                    maxIslandSize = Math.max(maxIslandSize, memo[0]);
                }
            }
        }

        return maxIslandSize;
    }

    private void search(int[][] grid, int i, int j, int[] memo) {
        grid[i][j] = 2; // visit
        memo[0]++; // update memo value
		
        if (i < grid.length - 1 && 1 == grid[i + 1][j]) search(grid, i + 1, j, memo);
        if (i > 0 && 1 == grid[i - 1][j]) search(grid, i - 1, j, memo);
        if (j < grid[i].length - 1 && 1 == grid[i][j + 1]) search(grid, i, j + 1, memo);
        if (j > 0 && 1 == grid[i][j - 1]) search(grid, i, j - 1, memo);
    }
}






// SUM OF SQUARE NUMBERS - Given a non-negative integer c, decide whether there're two integers a and b such that a2 + b2 = c.
function judgeSquareSum(c: number): boolean {
    for (let a = 0; a**2 <= c; a++) {
        let b: number = c - a**2;
        if (binarySearch(0, b, b)) return true;
    }
    return false;
};

function binarySearch(start: number, end: number, n: number): boolean {
  if (start > end) return false;
  const middle: number = Math.floor(start + (end - start) / 2);
  if (middle ** 2 === n) return true;
  if (middle ** 2 > n) return binarySearch(start, middle - 1, n);
  return binarySearch(middle + 1, end, n);
}