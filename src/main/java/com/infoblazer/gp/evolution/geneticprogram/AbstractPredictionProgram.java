/*
 * MIT License
 *
 * Copyright (c) 2014-2018 David Moskowitz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.infoblazer.gp.evolution.geneticprogram;

import com.infoblazer.gp.application.data.model.Metrics;
import com.infoblazer.gp.application.data.service.MetricsService;
import com.infoblazer.gp.application.fitness.FitnessEvaluator;
import com.infoblazer.gp.evolution.model.Population;
import com.infoblazer.gp.evolution.model.RegimeDetectionProgram;
import com.infoblazer.gp.evolution.model.ResultProducingProgram;
import com.infoblazer.gp.evolution.model.Winners;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.springframework.beans.factory.annotation.Value;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by David on 7/6/2015.
 */
public abstract class AbstractPredictionProgram extends AbstractGeneticProgram {
    private final static Logger logger = Logger.getLogger(PredictionProgram.class.getName());

    @Value("${trainingWindow:#{null}}")
    private Integer trainingWindow;

    @Value("${predictionWindow:#{null}}")
    private Integer predictionWindow;//Number of periods forward to look from current position for prediction

    @Value("${predictionGenerations:#{1}}")
    private Integer predictionGenerations;   //number of training runs at each prediction phase  , this could be 0 for no additional training

    @Value("${maxPrediction:#{null}}")
    private Double maxPrediction;
    @Value("${minPrediction:#{null}}")
    private Double minPrediction;
    @Value("${defaultPrediction:#{0}}")
    private Double defaultPrediction; //if an invalid prediction is made, default to this

    @Value("${useAdaptiveTraining:#{false}}")
    private boolean useAdaptiveTraining;

    @Value("${maxPredictionGenerations:#{null}}")
    private Integer maxPredictionGenerations;

    @Override
    protected Population predict(MetricsService metricsService, Integer metricId, int testingIteration, final int startPredictionGeneration, final int startTrainPos, final int startPos, final int endPos, final ResultProducingProgram fittestSoFar, final RegimeDetectionProgram fittestRegimeDectionSoFar) {
        //Need to reevaluate fitness every round during prediction. as opposed to linear regression

        Integer predictionId = metricsService.recordNewPrediction(metricId);
        Double lastDifference = null;
        Integer actualPredictionGenerations = predictionGenerations;
        Double difference = null;
        int currentPos = startPos; //This will be passed in, but may represent a future prediction
        ResultProducingProgram lattestResultProducingProgram = fittestSoFar;
        RegimeDetectionProgram lattestRegimeDetectionProgram = fittestRegimeDectionSoFar;


        PredictionState predictionState = initializePredictionState(currentPos);
        double totalError = 0.0d;
        int totalPredictions = 0;
        DescriptiveStatistics fitnessStats = new DescriptiveStatistics(); //Prediction status
        DescriptiveStatistics nodeStats = new DescriptiveStatistics();
        DescriptiveStatistics adfNodeStats = new DescriptiveStatistics();
        DescriptiveStatistics depthStats = new DescriptiveStatistics();

        DescriptiveStatistics regimeFitnessStats = new DescriptiveStatistics(); //Prediction status
        DescriptiveStatistics regimeNodeStats = new DescriptiveStatistics();
        DescriptiveStatistics regimeAdfNodeStats = new DescriptiveStatistics();
        DescriptiveStatistics regimeDepthStats = new DescriptiveStatistics();

        int generation = startPredictionGeneration;
        int invalidPredictionCount = 0;
        boolean invalidPrediction = false;
        while (currentPos <= endPos && generation <= maxGenerations) {
            int predictionPoint = currentPos;
            if (predictionWindow != null) {
                predictionPoint = currentPos + predictionWindow;
            }
            if (predictionPoint > endPos) {
                predictionPoint = endPos;
            }
            generation++;


            Metrics metrics = new Metrics();
            Metrics regimeMetrics = new Metrics();
            nodeStats.addValue(lattestResultProducingProgram.getNodeCount());


            depthStats.addValue(lattestResultProducingProgram.getDepth());

            Double prediction = predict(predictionState, lattestResultProducingProgram, lattestRegimeDetectionProgram, currentPos, currentPos, fitnessEvaluator); //presents a prediction at prediciton point
            if (prediction == null || (maxPrediction != null && prediction > maxPrediction)
                    || (minPrediction != null && prediction < minPrediction)) {
                prediction = defaultPrediction;
                invalidPredictionCount++;
                invalidPrediction = true;
            } else {
                invalidPrediction = false;
            }


            metrics.setRegime(predictionState.getRegime() == null ? null : predictionState.getRegime().intValue());
            metrics.setPopulationSize(population.getResultPopulation().size());

            Double actual = fitnessEvaluator.getTargetSeries().getY(predictionPoint);

            metrics.setFittestNodeCount(lattestResultProducingProgram.getNodeCount());
            if (lattestResultProducingProgram.getTotalAdfNodeCount() != null) {
                adfNodeStats.addValue(lattestResultProducingProgram.getTotalAdfNodeCount());
                metrics.setFittestAdfNodeCount(lattestResultProducingProgram.getTotalAdfNodeCount());
            }
            metrics.setFittestDepth(lattestResultProducingProgram.getDepth());

            if (lattestRegimeDetectionProgram != null) {
                regimeDepthStats.addValue(lattestRegimeDetectionProgram.getDepth());
                regimeMetrics.setPopulationSize(population.getRegimePopulation().size());
                regimeMetrics.setFittestNodeCount(lattestRegimeDetectionProgram.getNodeCount());
                regimeMetrics.setFittestAdfNodeCount(lattestRegimeDetectionProgram.getTotalAdfNodeCount());
                regimeMetrics.setFittestAdfNodeCount(lattestRegimeDetectionProgram.getTotalAdfNodeCount());
                regimeMetrics.setFittestDepth(lattestRegimeDetectionProgram.getDepth());
                regimeNodeStats.addValue(lattestRegimeDetectionProgram.getNodeCount());
                if (lattestRegimeDetectionProgram.getTotalAdfNodeCount() != null) {
                    regimeAdfNodeStats.addValue(lattestRegimeDetectionProgram.getTotalAdfNodeCount());
                }

            }


            metricsService.addPrediction(predictionId, predictionPoint, prediction, actual, invalidPredictionCount, actualPredictionGenerations, metrics, lattestResultProducingProgram, lattestRegimeDetectionProgram);
            invalidPredictionCount = 0;

            String pctDifference = "";


            if (useMeanSquaredError) {
                totalError = totalError + Math.pow(Math.abs(actual.doubleValue() - prediction.doubleValue()), 2);
            } else {
                totalError = totalError + Math.abs(actual.doubleValue() - prediction.doubleValue());
            }
            fitnessStats.addValue(totalError);
            regimeFitnessStats.addValue(totalError);

            totalPredictions++;
            difference = Math.abs(actual - prediction);


            if (actual != 0) {
                pctDifference = '(' + String.valueOf(Math.abs(100 * (actual - prediction) / actual)) + "%)";
            }


            System.out.println("***************************************************");
            System.out.println("Predicted [" + fitnessEvaluator.getTargetSeries().getX(predictionPoint) + "] " + prediction);
            System.out.println("Actual [" + fitnessEvaluator.getTargetSeries().getX(predictionPoint) + "] " + actual);
            System.out.println("Difference [" + fitnessEvaluator.getTargetSeries().getX(predictionPoint) + "] " + difference + pctDifference);
            predictionState.setxVals(predictionPoint, fitnessEvaluator.getTargetSeries().getX(predictionPoint));
            predictionState.setyVals(predictionPoint, prediction);
            fitnessEvaluator.drawPredictedSeries(1, predictionState.getXyArray());
            System.out.println("***************************************************");
            printNewFittest("Fittest Prediction ", lattestResultProducingProgram, false, false);
            printNewFittest("Fittest Regime Prediction ", lattestRegimeDetectionProgram, false, false);


            int startTrain = 0;
            if (trainingWindow != null) { //prediction window is optional. otherwise, training will take place from start of series.
                startTrain = currentPos - trainingWindow;
                if (startTrain < 0) {
                    startTrain = 0;
                }
            }
            if (startTrain < startTrainPos) {
                startTrain = startTrainPos;
            }

            for (int trainingCount = 0; trainingCount < actualPredictionGenerations; trainingCount++) {

                Winners winners = doTraining(true, metricId, generation, testingIteration, startTrain, currentPos,
                        predictionState.getRegime() == null ? null : predictionState.getRegime().intValue(),
                        (trainingCount == actualPredictionGenerations - 1));
                lattestResultProducingProgram = winners.getResultProducingProgram();
                lattestRegimeDetectionProgram = winners.getRegimeDetectionProgram();
            }


            if (useAdaptiveTraining) {
                actualPredictionGenerations = recalculateTrainingGenerations(lastDifference, actualPredictionGenerations, difference, invalidPrediction);
            }
            lastDifference = difference;
            currentPos++;

        }
        Metrics predictionMetrics = new Metrics();
        predictionMetrics.setBestFitness(lattestResultProducingProgram.getFitness());
        predictionMetrics.setMeanFitness(fitnessStats.getMean());
        predictionMetrics.setMedianFitness(fitnessStats.getPercentile(50));
        predictionMetrics.setStddevFitness(fitnessStats.getStandardDeviation());
        predictionMetrics.setVarianceFitness(fitnessStats.getVariance());


        predictionMetrics.setMeanNodeCount(nodeStats.getMean());
        predictionMetrics.setMedianNodeCount(nodeStats.getPercentile(50));
        predictionMetrics.setStddevNodeCount(nodeStats.getStandardDeviation());
        predictionMetrics.setVarianceNodeCount(nodeStats.getVariance());
        predictionMetrics.setMeanAdfNodeCount(adfNodeStats.getMean());
        predictionMetrics.setMedianAdfNodeCount(adfNodeStats.getPercentile(50));
        predictionMetrics.setStddevAdfNodeCount(adfNodeStats.getStandardDeviation());
        predictionMetrics.setVarianceAdfNodeCount(adfNodeStats.getVariance());


        predictionMetrics.setMeanDepth(depthStats.getMean());
        predictionMetrics.setMedianDepth(depthStats.getPercentile(50));
        predictionMetrics.setStddevDepth(depthStats.getStandardDeviation());
        predictionMetrics.setVarianceDepth(depthStats.getVariance());
        Metrics regimePredictionMetrics = new Metrics();
        if (lattestRegimeDetectionProgram != null)

        {

            regimePredictionMetrics.setBestFitness(lattestRegimeDetectionProgram.getFitness());
            regimePredictionMetrics.setMeanFitness(regimeFitnessStats.getMean());
            regimePredictionMetrics.setMedianFitness(regimeFitnessStats.getPercentile(50));
            regimePredictionMetrics.setStddevFitness(regimeFitnessStats.getStandardDeviation());
            regimePredictionMetrics.setVarianceFitness(regimeFitnessStats.getVariance());


            regimePredictionMetrics.setMeanNodeCount(regimeNodeStats.getMean());
            regimePredictionMetrics.setMedianNodeCount(regimeNodeStats.getPercentile(50));
            regimePredictionMetrics.setStddevNodeCount(regimeNodeStats.getStandardDeviation());
            regimePredictionMetrics.setVarianceNodeCount(regimeNodeStats.getVariance());
            regimePredictionMetrics.setMeanAdfNodeCount(regimeAdfNodeStats.getMean());
            regimePredictionMetrics.setMedianAdfNodeCount(regimeAdfNodeStats.getPercentile(50));
            regimePredictionMetrics.setStddevAdfNodeCount(regimeAdfNodeStats.getStandardDeviation());
            regimePredictionMetrics.setVarianceAdfNodeCount(regimeAdfNodeStats.getVariance());


            regimePredictionMetrics.setMeanDepth(regimeDepthStats.getMean());
            regimePredictionMetrics.setMedianDepth(regimeDepthStats.getPercentile(50));
            regimePredictionMetrics.setStddevDepth(regimeDepthStats.getStandardDeviation());
            regimePredictionMetrics.setVarianceDepth(regimeDepthStats.getVariance());


        }


        metricsService.endPrediction(predictionId, predictionMetrics, regimePredictionMetrics);

        Population result = new Population(lattestResultProducingProgram, lattestRegimeDetectionProgram);
        System.out.println("***************************************************");

        printNewFittest("Final Prediction Program ", lattestResultProducingProgram, false, true);

        printNewFittest("Final Regime Prediction Program", lattestRegimeDetectionProgram, false, true);

        if (totalPredictions != 0)

        {
            double meanSquaredError = totalError / totalPredictions;
            System.out.println("MSE = " + meanSquaredError);
        } else

        {
            System.out.println("No Predictions were made due to training and testing window sized");
        }

        return result;
    }

    private Integer recalculateTrainingGenerations(Double lastDifference, final int currentPredictionGenerations, Double difference, boolean invalidPrediction) {
        Integer newPredictionGenerations = currentPredictionGenerations;
        if (lastDifference != null) {
            if (invalidPrediction || difference >= lastDifference) {
                if (maxPredictionGenerations == null || newPredictionGenerations < maxPredictionGenerations) {
                    newPredictionGenerations++;
                    logger.info("Prediction worsened [" + lastDifference + "->" + difference + "] Increasing training to " + newPredictionGenerations);
                }
            } else {
                newPredictionGenerations--;
                logger.info("Prediction improved  [" + lastDifference + "->" + difference + "]");
                if (newPredictionGenerations < predictionGenerations) {
                    newPredictionGenerations = predictionGenerations;
                } else {
                    logger.info("Reducing training to " + newPredictionGenerations);
                }
            }
        }
        return newPredictionGenerations;
    }


    protected abstract PredictionState initializePredictionState(int predictionPos);


    protected abstract Double predict(PredictionState predictionState, ResultProducingProgram program, RegimeDetectionProgram regimeDetectionProgram, int startPos, int predictionPos, FitnessEvaluator evaluator);

    @Override
    protected boolean terminateTraining(ResultProducingProgram program) {
        return program != null && program.getFitness().equals(0.0d);
    }
}


