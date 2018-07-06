package com.jcnlp.dl.bp;

import com.google.common.collect.Lists;

import java.util.List;

/**
 * BP神经网络
 * 参考https://github.com/jingchenUSTC/ANN
 * @author jc
 */
public class BPNetwork {

    public int inputNum;
    public int hiddenNum;
    public int outputNum;

    public List<Node> inputNodes;
    public List<Node> hiddenNodes;
    public List<Node> outputNodes;

    public double[][] inputHiddenWeights;
    public double[][] hiddenOutputWeights;

    public double eta = 0.01;

    private List<Data> trainDatas;

    public BPNetwork(List<Data> trainDatas
            , int inputNum, int hiddenNum
            , int outputNum, double eta) {
        this.trainDatas = trainDatas;
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.eta = eta;

        inputNodes = Lists.newArrayList();
        hiddenNodes = Lists.newArrayList();
        outputNodes = Lists.newArrayList();

        inputHiddenWeights = new double[inputNum][hiddenNum];
        hiddenOutputWeights = new double[hiddenNum][outputNum];
    }

    /**
     * 权重更新
     */
    private void updateWeights() {
        // 更新输入层到隐藏层的权重矩阵
        for (int i = 0; i < inputNum; i ++) {
            for (int j = 0; j < hiddenNum; j ++) {
                inputHiddenWeights[i][j] -= eta
                        * inputNodes.get(i).forwardInput
                        * hiddenNodes.get(j).backwardOutput;
            }
        }

        // 更新隐藏层到输出层的权重矩阵
        for (int i = 0; i < hiddenNum; i ++) {
            for (int j =0; j < outputNum; j ++) {
                hiddenOutputWeights[i][j] -= eta
                        * hiddenNodes.get(i).forwardOutput
                        * outputNodes.get(j).backwardOutput;
            }
        }
    }

    /**
     * 前向传播
     */
    private void forward(List<Double> datas) {
        // 输入层
        for (int k = 0; k < datas.size(); k ++)
            inputNodes.get(k).setForwarValue(datas.get(k));

        // 隐藏层
        for (int j = 0; j < hiddenNum; j ++) {
            double temp = 0.0;
            for (int k = 0; k < inputNum; k ++) {
                temp += inputHiddenWeights[k][j]
                        * inputNodes.get(k).forwardOutput;
            }
            hiddenNodes.get(j).setForwarValue(temp);
        }

        // 输出层
        for (int j = 0; j < outputNum; j ++) {
            double temp = 0;
            for (int k = 0; k < hiddenNum; k ++)
                temp += hiddenOutputWeights[k][j]
                        * hiddenNodes.get(k).forwardOutput;
            outputNodes.get(j).setForwarValue(temp);
        }
    }

    /**
     * 反向传播
     */
    public void backward()

}
