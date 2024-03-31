## Dynamic Programming vs. Recursive algorithm

Thanks **hello-algo** project. Strongly recommended 
https://www.hello-algo.com/chapter_preface/


### **Recursive algorithm** 
Recursive algorithm is a "Top-to-Bottom" method，decomposing the problem of scale N into smaller problems.

<img width="453" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/24d26165-e4c5-4942-b25c-1cf0514a03cc">


```java
//climbing_stairs_dfs.java
int dfs(int i) {
    if (i == 1) return 1;
    if (i == 2) return 2;
    // dp[i] = dp[i-1] + dp[i-2]
    int count = dfs(i - 1) + dfs(i - 2);
    return count;
}

int climbingStairsDFS(int n) {
    return dfs(n);
}
```

### **Dynamic Programming** 
Dynamic Programming is a "Bottom-to-Top" method, starting from the smallest sub-problem, iteratively construct the solution of the larger problem.

<img width="670" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/c9b55d9f-4174-446a-b224-9c558653ab74">

```java
//climbing_stairs_dp.java
int climbingStairsDP(int n) {
    if (n == 1) return 1;
    if (n == 2) return 2;

    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

