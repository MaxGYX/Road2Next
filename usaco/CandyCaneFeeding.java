import java.util.Scanner;

public class CandyCaneFeeding {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // 读取输入
        int N = scanner.nextInt(); // 奶牛的数量
        int M = scanner.nextInt(); // 棒棒糖的数量

        // 读取奶牛的初始高度
        int[] cowHeights = new int[N];
        for (int i = 0; i < N; i++) {
            cowHeights[i] = scanner.nextInt();
        }

        // 读取棒棒糖的高度
        int[] candyCanes = new int[M];
        for (int i = 0; i < M; i++) {
            candyCanes[i] = scanner.nextInt();
        }

        // 模拟喂食过程
        for (int candy : candyCanes) { //外层循环，循环每一根棒棒糖
            int candyBottom = 0; // 使用一个变量记录当前棒棒糖的底部高度，初识底部=0

            // 每头奶牛尝试吃糖果
            for (int i = 0; i < N; i++) { //内层循环，对当前的棒棒糖循环所有牛
                if (cowHeights[i] > candyBottom) {
                    // 奶牛可以吃到糖果
                    int eaten = Math.min(cowHeights[i] - candyBottom, candy - candyBottom);
                    cowHeights[i] += eaten; //更新牛高度
                    candyBottom += eaten; //更新棒棒糖底部高度
                }
                // 如果cowHeights[i] <= candyBottom，奶牛吃不到糖果，不需要任何操作
            }
        }

        // 输出结果
        for (int height : cowHeights) {
            System.out.println(height);
        }

        scanner.close();
    }
}
