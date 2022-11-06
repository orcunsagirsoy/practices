//Minimize Malware Spread II


// COUNT THE SET BITS
import java.util.ArrayList;
import java.io.*; 
import java.util.*;
public class Solution 
{
    public static int countSetBits(int n) 
    {
        int count = 0;
        ArrayList<Integer> resultList = new ArrayList<>();
        for(int i=0;i<=n;i++){
            count = count + bitcount(i);
            resultList.add(count%1000000007);
        }
        return resultList.get(resultList.size() - 1);
    }
    
    public static int bitcount(int i) 
    {
        int count = 0;
        while (i > 0)
        {
            if ((i & 1) == 1) count++;  //(i & 1) == 1  last bit is 1
            i = i >> 1;  // shifting it to right
        }
        return count;
    }
}