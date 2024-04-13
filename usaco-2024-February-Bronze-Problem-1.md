https://usaco.org/index.php?page=viewproblem2&cpid=1395


* a. if S in [0-9], Bessie take all stones, Bessie win.
* b. If S=10, Bessie cannot take 10, no matter K Bessie takes, Elsie will take the rest 10-K, Elsie win.
* c. If S in [11-19], Bessie take K=S%10, and Elsie will face 10 stones, then, Bessie win.
* d. If S=20, Bessie cannot take 10/20, no matter K Bessie takes, Elsie will face a/c, then Elsie win
* e. If S in [21-29], Bessie take K=S%10, and Elsie will face 20 stones, then Bessie win.
* f. If S=30, Bessie cannot take 10/20/30, no matter K Bessie takes, Elsie will face a/c/e, then Elsie win.

… so solution is quite easy, if ((S%10) != 0) Bessie win, else Elsie win.

One line code, I cannot believe that :_)

Is that the simplest problem in USACO Bronze Contest?

```java
public class PalindromeGame {
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] stones = new int[n];
        for (int i=0; i<n; i++){
            stones[i] = sc.nextInt();
        }
        for (int i=0; i<n; i++){
            if ((stones[i]%10) != 0) System.out.println("B");
            else System.out.println("E");
        }
    }
}
```
