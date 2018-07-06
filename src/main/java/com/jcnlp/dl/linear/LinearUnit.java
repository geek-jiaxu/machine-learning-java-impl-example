package com.jcnlp.dl.linear;

/**
 * 线性回归
 * 代码参考自https://www.zybuluo.com/hanbingtao/note/448086，python代码
 * @author jc
 */
public class LinearUnit {

    public int inputNum; // 输入特征数
    public double[] weights; // 特征权重
    public double bias; // 偏置项
    public double rate = 0.01;

    public LinearUnit(int inputNum) {
        this.inputNum = inputNum;
        this.weights = new double[inputNum];
        this.bias = 0.0;
    }

    /**
     * 预测输出结果
     */
    public double predict(double[] inputs) {
        double result = 0.0;
        for (int i = 0; i < inputNum; i ++) {
            result += inputs[i] * weights[i];
        }
        result += bias;
        return activator(result);
    }

    /**
     * 训练权重及偏置
     */
    public void train(double[][] inputss, double[] labels, int iter) {
        for (int i = 0; i < iter; i ++) {
            this.oneIteration(inputss, labels);
            System.out.println(this);
        }
    }

    /**
     * 一次迭代，训练所有的数据
     */
    private void oneIteration(double[][] inputss, double[] labels) {
        for (int i = 0; i < inputss.length; i ++) {
            double output = predict(inputss[i]);
            updateWeigths(inputss[i], output, labels[i]);
        }
    }


    /**
     * 权重更新
     * @param output 预测输出值
     * @param label 样本输出值
     * @return
     */
    private void updateWeigths(double[] inputs, double output, double label) {
        double delta = label - output;
        for (int i = 0; i < inputNum; i ++) {
            this.weights[i] += this.rate * delta * inputs[i];
        }
        this.bias += rate * delta;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("weights:[");
        for (int i = 0; i < weights.length; i ++) {
            sb.append(weights[i] + ", ");
        }
        sb.setLength(sb.length() - 2);
        sb.append("] bias:");
        sb.append(bias);
        return sb.toString();
    }

    /**
     * 激活函数
     * @param x
     * @return
     */
    private double activator(double x) {
        return x;
    }

    public static void main(String[] args) {
        double[][] inputss = {{5}, {3}, {8}, {1.4}, {10.1}};
        double[] labels = {5500, 2300, 7600, 1800, 11400};

        LinearUnit p = new LinearUnit(inputss[0].length);
        p.train(inputss, labels, 100);
        // output : weights:[0.1, 0.2] bias:-0.2
    }
}
