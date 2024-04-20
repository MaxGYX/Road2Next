一些练习的code片段

### 1. 判断字符串是不是回文

```java
    static boolean IsPalindrome(String s){
        boolean retValue= true;
        int i = 0;
        int j = s.length()-1;
        for (i = 0; i <= j; i++,j--){
            if (s.charAt(i) != s.charAt(j)){
                retValue = false;
                break;
            }
        }
        return retValue;
    }
    public static void main(String[] args) throws IOException {
        System.out.println(IsPalindrome("abcdcba"));
        System.out.println(IsPalindrome("abcddcba"));
        System.out.println(IsPalindrome("kakaakak"));
        System.out.println(IsPalindrome("kakakak"));
        System.out.println(IsPalindrome("abcdsfescba"));
    }
```
