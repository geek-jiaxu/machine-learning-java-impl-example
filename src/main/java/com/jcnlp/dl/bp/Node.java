package com.jcnlp.dl.bp;

import java.util.List;

/**
 * 每层独立节点
 * @author jc
 */
public class Node {
    // 节点类型
    public NodeType nodeType;

    // 前向节点输入/输出
    public double forwardInput;
    public double forwardOutput;

    // 反向节点输入/输出
    public double backwardInput;
    public double backwardOutput;

    public List<Double> weights;

    public Node(NodeType nodeType) {
        this.nodeType = nodeType;
    }

    /**
     * 设置前向输入/输出
     * @param input
     */
    public void setForwarValue(double input) {
        this.forwardInput = input;
        this.forwardOutput = forwardSigmoid(input);
    }

    /**
     * 反向传播
     */
    public void setBackwardValue(double input) {
        this.backwardInput = input;
        this.backwardOutput = backwardPropagate(input);
    }

    private double forwardSigmoid(double input) {
        if (this.nodeType == NodeType.INPUT) {
            return input;
        }
        return tanhS(input);
    }

    public double backwardPropagate(double input) {
        if (NodeType.INPUT == this.nodeType) {
            return input;
        }
        return tanhSDerivative(input);
    }

    /**
     * 激活函数
     * @param input
     * @return
     */
    private double tanhS(double input) {
        return (double) ((Math.exp(input) - Math.exp(- input)) / (Math.exp(input) + Math
                .exp(- input)));
    }

    /**
     * 激活函数导数
     * @param input
     * @return
     */
    private double tanhSDerivative(double input) {
        return (double) ((1 - Math.pow(forwardOutput, 2)) * input);
    }
}
