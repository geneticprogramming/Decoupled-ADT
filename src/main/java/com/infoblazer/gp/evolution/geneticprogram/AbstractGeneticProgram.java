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


import com.infoblazer.gp.application.data.model.FitnessEvaluation;
import com.infoblazer.gp.application.data.model.Metrics;
import com.infoblazer.gp.application.data.service.MetricsService;
import com.infoblazer.gp.application.fitness.AbstractFitnessEvaluator;
import com.infoblazer.gp.application.fitness.FitnessEvaluator;
import com.infoblazer.gp.evolution.library.Library;
import com.infoblazer.gp.evolution.library.RegimeLibrary;
import com.infoblazer.gp.evolution.library.ResultLibrary;
import com.infoblazer.gp.evolution.model.*;
import com.infoblazer.gp.evolution.primitives.FunctionSet;
import com.infoblazer.gp.evolution.primitives.GP_TYPES;
import com.infoblazer.gp.evolution.primitives.Primitive;
import com.infoblazer.gp.evolution.primitives.TerminalSet;
import com.infoblazer.gp.evolution.primitives.functions.*;
import com.infoblazer.gp.evolution.primitives.terminals.AbstractTerminal;
import com.infoblazer.gp.evolution.primitives.terminals.Terminal;
import com.infoblazer.gp.evolution.selectionstrategy.AbstractSelectionStrategy;
import com.infoblazer.gp.evolution.selectionstrategy.SelectionStrategy;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import java.util.*;


/**
 * Created by David on 5/22/2014.
 */


public abstract class AbstractGeneticProgram {

    private final static Logger logger = Logger.getLogger(AbstractGeneticProgram.class.getName());
    @Autowired
    private MetricsService metricsService;
    @Value("${adfArity:#{null}}")
    private String[] adfArities;

    @Value("${maxValidFitness:#{null}}") //ignore any fitness over this
    private Double maxValidFitness;
    @Value("${printTrainingProgram:#{false}}")
    private boolean printTrainingProgram; //output best training program each generation   to stdout

    protected Population population;

    protected SelectionStrategy selectionStrategy;
    @Value("${meanSquaredError:#{false}}")
    protected boolean useMeanSquaredError;

    @Value("${regimeSelection:#{true}}")
    protected Boolean regimeSelection;  // always use reproduction on regimes


    @Value("${gcFrequency:#{1}}")
    private int gcFrequency;


    @Value("${populationSize}")
    private int initialPopulationSize;
    @Value("${regimePopulationSize:#{null}}")
    private Integer initialRegimePopulationSize;

    @Value("${maxTotalNodes:#{null}}")
    private Integer maxTotalNodes;
    private FunctionSet functionSet;
    private FunctionSet regimeFunctionSet;

    private List<String> series;
    private TerminalSet terminalSet;
    @Value("${maxDepth:#{999999}}")
    protected int maxDepth;
    @Value("${maxSize:#{999999}}")
    private int maxSize;
    @Value("${maxInitDepth}")
    private int maxInitDepth;

    @Value("${regimes:#{1}}")
    private int numberOfRegimes;
    @Value("${maxGenerations:#{999999}}")
    protected int maxGenerations;
    @Value("${testingGenerations:#{1}}")
    private int testingGenerations;
    @Value("${trainingGenerations}")
    private int trainingGenerations;

    @Value("${startTrain:#{null}}")
    private String startTrain = null;
    @Value("${endTrain:#{null}}")
    private String endTrain = null;
    @Value("${startTest:#{null}}")
    private String startTest = null;
    @Value("${endTest:#{null}}")
    private String endTest = null;
    @Autowired
    private ResultLibrary resultLibrary;
    @Autowired
    private RegimeLibrary regimeLibrary;

    private GP_TYPES returnType;
    private GrowMethod growMethod = GrowMethod.HALF_HALF; //pass as param
    protected FitnessEvaluator fitnessEvaluator;


    public AbstractGeneticProgram() {
    }

    public void init(FunctionSet functionSet,
                     FunctionSet regimeFunctionSet,
                     TerminalSet terminalSet,
                     FitnessEvaluator fitnessEvaluator,
                     SelectionStrategy selectionStrategy,

                     GP_TYPES returnType,
                     List<String> series) {

        this.selectionStrategy = selectionStrategy;
        this.functionSet = functionSet;
        this.regimeFunctionSet = regimeFunctionSet;
        this.terminalSet = terminalSet;
        this.series = series;
        this.fitnessEvaluator = fitnessEvaluator;
        this.returnType = returnType;


    }


    public Population run() {
        fitnessEvaluator.drawTargetSeries();
        Integer metricId = metricsService.recordNewRun();


        //need a stopping criteria
        int generation = 0;
        int currentTestingGeneration = 0;

        int start = fitnessEvaluator.getSeriesStart();
        int end = fitnessEvaluator.getSeriesEnd();

        int startTrainPos = start + ((end - start) / 3);
        if (startTrain != null) {
            startTrainPos = getPositionParameter(startTrain);
        }
        int endTrainPos = start + (end - start) * 2 / 3;
        if (endTrain != null) {
            endTrainPos = getPositionParameter(endTrain);
        }
        int startTestPos = start;
        if (startTest != null) {
            startTestPos = getPositionParameter(startTest);
        }

        int endTestPos = (int) (start + ((end - start) * 0.9));
        if (endTest != null) {
            endTestPos = getPositionParameter(endTest);
        }


        Map<String, FunctionSet> aritySet = new HashMap<>();
        if (adfArities != null) {
            for (String arityString : adfArities) {
                if (aritySet.get(arityString) == null) {
                    int arity = Integer.parseInt(arityString);
                    FunctionSet arityFunctionSet = FunctionSet.reduceArity(functionSet, arity);
                    aritySet.put(arityString, arityFunctionSet);
                }
            }
        }


        ResultProducingProgram fittestTesting = null;
        RegimeDetectionProgram fittestRegimeDetectionTesting = null;
        ResultProducingProgram fittestTraining = null;
        RegimeDetectionProgram fittestRegimeDetectionTraining = null;
        while (currentTestingGeneration < testingGenerations && !terminateTraining(fittestTesting)) {
            population = new Population();
            ResultProducingProgram[] resultProducingPrograms = initializePopulation(functionSet, terminalSet, aritySet, adfArities, series, initialPopulationSize, maxInitDepth, maxSize, numberOfRegimes, returnType, growMethod);
            population.setResultPopulation(resultProducingPrograms);
            RegimeDetectionProgram[] regimeDetectionPrograms = null;
            if (numberOfRegimes > 1) {
                if (initialRegimePopulationSize == null) {
                    initialRegimePopulationSize = initialPopulationSize;
                }
                regimeDetectionPrograms = initializeRegimePopulation(regimeFunctionSet, terminalSet, aritySet, adfArities, series,
                        initialRegimePopulationSize, maxInitDepth, maxSize, numberOfRegimes, growMethod, regimeSelection);
            }
            population.setRegimePopulation(regimeDetectionPrograms); //Only allow one regime for regime branch

            //random fitness calcs, probably not that important first round
            Random random = new Random();
            for (int i = 0; i < resultProducingPrograms.length; i++) {
                ResultProducingProgram resultProducingProgram = resultProducingPrograms[i];
                RegimeDetectionProgram regimeDetectionProgram = regimeDetectionPrograms == null ? null : regimeDetectionPrograms[random.nextInt(initialRegimePopulationSize)];
                FitnessEvaluation fitnessEvaluation = fitnessEvaluator.calculateProgramFitness(startTestPos, endTestPos, maxDepth,
                        resultProducingProgram, regimeDetectionProgram, selectionStrategy.getDirection());
                resultProducingProgram.setFitness(fitnessEvaluation.getFitness());

            }

            if (numberOfRegimes > 1) {
                for (int i = 0; i < regimeDetectionPrograms.length; i++) {
                    ResultProducingProgram resultProducingProgram = resultProducingPrograms[random.nextInt(initialPopulationSize)];
                    RegimeDetectionProgram regimeDetectionProgram = regimeDetectionPrograms == null ? null : regimeDetectionPrograms[i];
                    FitnessEvaluation fitnessEvaluation = fitnessEvaluator.calculateProgramFitness(startTestPos, endTestPos, maxDepth,
                            resultProducingProgram, regimeDetectionProgram, selectionStrategy.getDirection());
                    regimeDetectionProgram.setFitness(fitnessEvaluation.getFitness());

                }
            }


            generation = 0;
            currentTestingGeneration++;

            while (generation < trainingGenerations && !terminateTraining(fittestTraining)) {
                generation++;

                Winners winners = doTraining(false, metricId, generation, currentTestingGeneration, startTrainPos, endTrainPos, null, true);
                fittestTraining = winners.getResultProducingProgram();
                fittestRegimeDetectionTraining = winners.getRegimeDetectionProgram();


            }
            System.out.println("********************************************");
            printNewFittest("Fittest RP after training ", fittestTraining, true, false);
            printNewFittest("Fittest Regime after training ", fittestRegimeDetectionTraining, true, false);
            FitnessEvaluation fitnessEvaluation = fitnessEvaluator.calculateProgramFitness(startTestPos, endTestPos, maxDepth, fittestTraining, fittestRegimeDetectionTraining, selectionStrategy.getDirection());
            double testingFitness = fitnessEvaluation.getFitness();
            fittestTesting = compareFittest(fittestTesting, fittestTraining);
            fittestRegimeDetectionTesting = compareRegimeDetection(fittestRegimeDetectionTesting, fittestRegimeDetectionTraining);
            printNewFittest("Fittest RP after testing ", fittestTesting, true, false);
            printNewFittest("Fittest Regime after testing ", fittestRegimeDetectionTesting, true, false);
            System.out.println("populationSize: " + population.getResultPopulation().size());
            System.out.println("Regime populationSize: " + population.getRegimePopulation().size());
            fitnessEvaluator.drawPredictedSeries(0, fitnessEvaluation.getXyArray());
            if (numberOfRegimes > 1) {
                fitnessEvaluator.drawPredictedRegimeSeries(fitnessEvaluation.getRegimeXyArray());

            }

            fittestTesting.setFitness(testingFitness);

        }
       /* Testing phase in a loop here, then move on to prediction */

        fittestTraining = fittestTesting;
        fittestRegimeDetectionTraining = fittestRegimeDetectionTesting;
        System.out.println("********************************************");
        printNewFittest("Fittest RP after testing ", fittestTraining, true, false);
        printNewFittest("Fittest Regime after testing ", fittestRegimeDetectionTraining, true, false);
        System.out.println("populationSize: " + population.getResultPopulation().size());
        System.out.println("Regime populationSize: " + population.getRegimePopulation().size());
        System.out.println("********************************************");

        Population result = predict(metricsService, metricId, testingGenerations + 1, generation, startTrainPos, endTestPos + 1, end, fittestTraining, fittestRegimeDetectionTraining);


        metricsService.endRun(metricId);

        return result;
    }

    private void collectGarbage(Library library, List<? extends AbstractProgram> programs) {    // do every X generations
        Set<Integer> libaryInUse = new HashSet<>();
        for (AbstractProgram program : programs) {
            List<Primitive> primitives = new ArrayList<Primitive>();
            AbstractSelectionStrategy.addPrimitivesTyped(primitives, program.getRoot(), AatImpl.class);
            for (Primitive primitive : primitives) {
                Aat aat = (Aat) primitive;
                libaryInUse.add(aat.getLibaryKey());
                logger.trace("retaining " + aat.getLibaryKey());
            }
        }

        library.retainAll(libaryInUse);


    }


    protected Winners doTraining(final boolean isPrediction, final Integer metricId, final int generation, final int currentTestingGeneration, int startTrainPos,
                                 int endTrainPos, Integer predictedRegime, boolean lastTrainingThisGeneration) {
        Date trainingStart = new Date();

        System.out.println("************* Training Generation " + currentTestingGeneration + "|" + generation + "************************");
        logger.info("Evolving generation " + currentTestingGeneration + "|" + generation);
        Winners winners = train(generation, startTrainPos, endTrainPos, predictedRegime, lastTrainingThisGeneration);


        ResultProducingProgram fittestTraining = winners.getResultProducingProgram();
        RegimeDetectionProgram fittestRegimeDetectionTraining = winners.getRegimeDetectionProgram();
        //get metrics from population if this is the last round per generation
        Metrics metrics = null;
        Metrics regimeMetrics = null;
        if (lastTrainingThisGeneration) {
            metrics = this.population.getMetrics();
            regimeMetrics = this.population.getRegimeMetrics();

            metrics.setPopulationSize(population.getResultPopulation().size());
            metrics.setLibraryPopulationSize(fitnessEvaluator.getResultLibrary().getSize());
            metrics.setBestFitness(fittestTraining.getFitness());
            metrics.setFittestNodeCount(fittestTraining.getNodeCount());
            metrics.setFittestAdfNodeCount(fittestTraining.getTotalAdfNodeCount());
            metrics.setFittestDepth(fittestTraining.getDepth());


            regimeMetrics.setPopulationSize(population.getRegimePopulation().size());
            regimeMetrics.setLibraryPopulationSize(fitnessEvaluator.getRegimeLibrary().getSize());
            if (fittestRegimeDetectionTraining != null) {
                regimeMetrics.setBestFitness(fittestRegimeDetectionTraining.getFitness());
                regimeMetrics.setFittestNodeCount(fittestRegimeDetectionTraining.getNodeCount());
                regimeMetrics.setFittestAdfNodeCount(fittestRegimeDetectionTraining.getTotalAdfNodeCount());
                regimeMetrics.setFittestDepth(fittestRegimeDetectionTraining.getDepth());
            }

        }
        System.out.println("populationSize: " + population.getResultPopulation().size());
        System.out.println("Regime populationSize: " + population.getRegimePopulation().size());

        printNewFittest("Fittest:", fittestTraining, printTrainingProgram, false);
        System.out.println("Fittest Node Size:" + fittestTraining.getNodeCount());
        System.out.println("Fittest Depth:" + fittestTraining.getDepth());
        printNewFittest("Fittest Regime:", fittestRegimeDetectionTraining, printTrainingProgram, false);
        FitnessEvaluation trainingResult = fitnessEvaluator.evaluate(fittestTraining, fittestRegimeDetectionTraining, startTrainPos, endTrainPos, maxDepth, selectionStrategy.getDirection());
        fitnessEvaluator.drawTrainingSeries(trainingResult.getXyArray());
        Date trainingEnd = new Date();

        if (lastTrainingThisGeneration) {
            metricsService.addTraining(isPrediction, metricId, currentTestingGeneration, generation, trainingStart, trainingEnd,
                    trainingResult.getXyArray(), trainingResult.getRegimeXyArray(),
                    fitnessEvaluator.getTargetSeries(), metrics, regimeMetrics, fittestTraining, fittestRegimeDetectionTraining);
        }
        if (numberOfRegimes > 1) {
            fitnessEvaluator.drawTrainingRegimeSeries(trainingResult.getRegimeXyArray());

        }
        return winners;
    }

    protected abstract boolean terminateTraining(ResultProducingProgram program);


    protected abstract Population predict(MetricsService metricsService, Integer metricId, int testingIteration, int startTrainPos, int generation, int startPos, int endPos, ResultProducingProgram fittestSoFar, RegimeDetectionProgram fittestRegimeDectionSoFar);


    private Winners train(int generation, int windowStart, int windowEnd, Integer predictedRegime, boolean lastTrainingThisGeneration) {


        Population nextGeneration = selectionStrategy.selectionNextGeneration(generation, trainingGenerations, growMethod, population, maxTotalNodes, windowStart, windowEnd, predictedRegime, lastTrainingThisGeneration);
        Population nextPoulation = buildNextGen(nextGeneration);
        this.population = nextPoulation;

        if (generation % gcFrequency == 0) {
            if (resultLibrary.getSize() > 0) {
                collectGarbage(resultLibrary, population.getResultPopulation());
            }
            if (regimeLibrary.getSize() > 0) {
                collectGarbage(regimeLibrary, population.getRegimePopulation());
            }
        }

        Winners winners = findFittest();
        return winners;

    }

    private Winners findFittest() {
        ResultProducingProgram resultProducingProgram = (ResultProducingProgram) AbstractProgram.findFittest(population.getResultPopulation(), selectionStrategy.getDirection());
        RegimeDetectionProgram regimeDetectionProgram = (RegimeDetectionProgram) AbstractProgram.findFittest(population.getRegimePopulation(), selectionStrategy.getDirection());
        if (logger.isDebugEnabled()) {
            logger.debug("Best this round :" + resultProducingProgram.asLanguageString(maxDepth));
            logger.debug("Fitness :" + resultProducingProgram.getFitness());
        }
        Winners winners = new Winners(resultProducingProgram, regimeDetectionProgram);
        return winners;
    }

    private Population buildNextGen(final Population seedPopulation) {

        Metrics regimeMetrics = new Metrics();
        Metrics metrics = new Metrics();

        metrics.setFitnessEvaluations(fitnessEvaluator.getAndResetFitnessEvaluations());
        metrics.setFitnessCalculations(fitnessEvaluator.getAndResetFitnessCalculations());

        Population population = new Population();
        AbstractProgram[] resultProducingPrograms = new ResultProducingProgram[seedPopulation.getRPLength()];
        buildNextGenPrograms(seedPopulation.getResultPopulation(), resultProducingPrograms, metrics);
        population.setResultPopulation((ResultProducingProgram[]) resultProducingPrograms);


        AbstractProgram[] regimeDetectionPrograms = new RegimeDetectionProgram[seedPopulation.getRGLength()];

        buildNextGenPrograms(seedPopulation.getRegimePopulation(), regimeDetectionPrograms, regimeMetrics);

        population.setRegimePopulation((RegimeDetectionProgram[]) regimeDetectionPrograms);
        population.setMetrics(metrics);
        population.setRegimeMetrics(regimeMetrics);
        return population;

    }

    private void buildNextGenPrograms(List<? extends AbstractProgram> seedPopulation,
                                      AbstractProgram[] newPrograms,
                                      Metrics metrics) {

        boolean hasAdf = false;
        int invalidPopulationSize = 0;
        Integer totalNodeCount = 0;
        Integer totalAdfNodeCount = null;
        DescriptiveStatistics fitnessStats = new DescriptiveStatistics();
        DescriptiveStatistics nodeStats = new DescriptiveStatistics();
        DescriptiveStatistics depthStats = new DescriptiveStatistics();
        DescriptiveStatistics adfNodeStats = new DescriptiveStatistics();
        for (int i = 0; i < seedPopulation.size(); i++) {
            AbstractProgram program = seedPopulation.get(i);

            if (program.hasValidFitness(maxValidFitness)) {
                Double fitness = program.getFitness();
                if (logger.isTraceEnabled()) {
                    logger.trace("Fitness: " + fitness + " " + program.asLanguageString(100));
                }
                fitnessStats.addValue(fitness);
                totalNodeCount += program.getNodeCount();
                nodeStats.addValue(program.getNodeCount());
                depthStats.addValue(program.getDepth());
                if (program.getTotalAdfNodeCount() != null) {
                    adfNodeStats.addValue(program.getTotalAdfNodeCount());
                    if (totalAdfNodeCount == null) {
                        totalAdfNodeCount = 0;
                    }
                    totalAdfNodeCount += program.getTotalAdfNodeCount();
                    hasAdf = true;
                }
            } else {
                invalidPopulationSize++;
            }
            //    program.setId(i + 1); move this up to before fitness evaluiation, for dyfor purposes
            newPrograms[i] = program;
        }

        metrics.setInvalidPopulationSize(invalidPopulationSize);
        metrics.setMeanFitness(fitnessStats.getMean());
        metrics.setStddevFitness(fitnessStats.getStandardDeviation());
        metrics.setMedianFitness(fitnessStats.getPercentile(50));
        metrics.setVarianceFitness(fitnessStats.getVariance());
        metrics.setMeanNodeCount(nodeStats.getMean());
        metrics.setStddevNodeCount(nodeStats.getStandardDeviation());
        metrics.setMedianNodeCount(nodeStats.getPercentile(50));
        metrics.setVarianceNodeCount(nodeStats.getVariance());
        metrics.setTotalNodeCount(totalNodeCount);

        metrics.setMeanDepth(depthStats.getMean());
        metrics.setStddevDepth(depthStats.getStandardDeviation());
        metrics.setMedianDepth(depthStats.getPercentile(50));
        metrics.setVarianceDepth(nodeStats.getVariance());

        if (hasAdf) {
            metrics.setMeanAdfNodeCount(adfNodeStats.getMean());
            metrics.setTotalAdfNodeCount(totalAdfNodeCount);
            metrics.setStddevAdfNodeCount(adfNodeStats.getStandardDeviation());
            metrics.setMedianAdfNodeCount(adfNodeStats.getPercentile(50));
            metrics.setVarianceAdfNodeCount(adfNodeStats.getVariance());
        }

    }


    public static List<Adf> initializeAdf(FunctionSet functionSet, String[] adfArities, Map<String, FunctionSet> aritySet, List<String> series, int maxDepth, int maxSize, int regimes, GrowMethod growMethod) {
        List<Adf> adfs = new ArrayList<Adf>();

        int adfCounter = 0;
        for (String arityString : adfArities) {
            int symbolicArity = Integer.parseInt(arityString);
            //reduce function set by arity - should save this off as it is resuable
            AdfImpl newAdf = new AdfImpl();
            newAdf.setArity(symbolicArity);
            adfs.add(newAdf);
            newAdf.initializeRoots(regimes);
            newAdf.setName("adf" + adfCounter++);
            Terminal[] symbolicParameters = new Terminal[symbolicArity];
            for (int i = 0; i < symbolicArity; i++) {
                symbolicParameters[i] = AbstractSymbolicParameter.buildParmeter("arg" + i, null);
            }
            newAdf.setSymbolicParameters(new TerminalSet(symbolicParameters));
            //todo this should be a generic combine function, also make these sets immutable
            //this just serves the purpose of setting the return type of the adf.
            //perhaps there is a better way

            TerminalSet combinedTerminals = AbstractTerminal.addAll(symbolicParameters);
            newAdf.setFunctionSet(functionSet);
            newAdf.setTerminalSet(combinedTerminals);
            int regime = 0;
            GP_TYPES returnType = null;
            while (regime < regimes) {
                FunctionSet arityFunctionSet = aritySet.get(arityString);
                if (arityFunctionSet.getLength() == 0) {
                    System.err.println("FATAL ERROR: There are no available functions specified in the program paramters of arity " + arityString);
                    System.exit(1);
                }
                Primitive root = ResultProducingProgram.generatePrimitive(returnType,functionSet, arityFunctionSet, combinedTerminals, series, maxDepth, maxDepth,growMethod, true, null, true);
                if (regime == 0) {
                    returnType = root.getReturnType();
                }
                newAdf.setRoot(root, regime);
                regime++;


            }
        }


        return adfs;
    }

    public static RegimeDetectionProgram[] initializeRegimePopulation(FunctionSet functionSet, TerminalSet terminalSet, Map<String, FunctionSet> aritySet, String[] adfArities,
                                                                      List<String> series, int populationSize,
                                                                      int maxDepth, int maxSize, int regimes, GrowMethod growMethod, boolean regimeSelection) {
        RegimeDetectionProgram[] population = new RegimeDetectionProgram[populationSize];
        logger.info("Generating initial population");
        //initiaize population


        GrowMethod currentGrowMethod = growMethod;
        int added = 0;
        while (added < populationSize) {
            if (growMethod == GrowMethod.HALF_HALF) {
                if (added % 2 == 0) {
                    currentGrowMethod = GrowMethod.GROW;
                } else {
                    currentGrowMethod = GrowMethod.FULL;
                }
            }
            RegimeDetectionProgram program = RegimeDetectionProgram.generateProgram(functionSet, terminalSet, aritySet, adfArities, series, maxDepth,
                    maxSize, currentGrowMethod, regimes, regimeSelection);
            if (logger.isTraceEnabled()) {
                System.out.println("*********************************************");
                System.out.println(program.asLanguageString(100));
            }
            List<Primitive> primitivesTmp = new ArrayList<Primitive>();


            int check1Size = AbstractSelectionStrategy.addPrimitives(primitivesTmp, program.getRoot(), null);
            if (AbstractSelectionStrategy.checkSize(maxDepth, maxSize, check1Size, primitivesTmp.size())) {
                program.setId(added + 1);
                population[added] = program;
                added++;
            } else {
                logger.debug("hit size  limit generating program");
            }


        }
        return population;
    }

    public static ResultProducingProgram[] initializePopulation(FunctionSet functionSet, TerminalSet terminalSet, Map<String, FunctionSet> aritySet, String[] adfArities,
                                                                List<String> series, int populationSize, int maxDepth, int maxSize, int regimes, GP_TYPES returnType, GrowMethod growMethod) {
        ResultProducingProgram[] population = new ResultProducingProgram[populationSize];
        logger.info("Generating initial population");
        //initiaize population


        GrowMethod currentGrowMethod = growMethod;
        int added = 0;
        while (added < populationSize) {
            if (growMethod == GrowMethod.HALF_HALF) {
                if (added % 2 == 0) {
                    currentGrowMethod = GrowMethod.GROW;
                } else {
                    currentGrowMethod = GrowMethod.FULL;
                }
            }

            ResultProducingProgram program = ResultProducingProgram.generateProgram(functionSet, terminalSet, aritySet, adfArities, series, maxDepth, maxSize, regimes, currentGrowMethod, returnType);
            List<Primitive> primitivesTmp = new ArrayList<Primitive>();


            int checkDepth = AbstractSelectionStrategy.addPrimitives(primitivesTmp, program.getRoot(), null);
            if (AbstractSelectionStrategy.checkSize(maxDepth, maxSize, checkDepth, primitivesTmp.size())) {
                program.setId(added + 1);
                population[added] = program;
                added++;
            } else {
                logger.debug("hit size  limit generating program");
            }


        }


        return population;
    }


    protected void printNewFittest(String message, AbstractProgram program, boolean printProgram, boolean simplify) {
        if (program != null) {

            System.out.println(message + ": " + program.getFitness());
            if (printProgram) {

                if (simplify) {
                    System.out.println("Simplified :" + program.simplify().asLanguageString(0, maxDepth));
                } else {
                    System.out.println("Program :" + program.asLanguageString(maxDepth));
                }
                if (program.getAdfs() != null && program.getAdfs().size() > 0) {
                    for (int adfCount = 0; adfCount < program.getAdfs().size(); adfCount++) {
                        System.out.println("ADF" + adfCount);
                        Adf adf = program.getAdfs().get(adfCount);
                        for (int regime = 0; regime < adf.getNumberOfRoots(); regime++) {
                            System.out.println("Regime[" + regime + "]: " + adf.getRoot(regime).asLanguageString(0, maxDepth));
                        }
                    }

                }
            }

        }

    }

    protected RegimeDetectionProgram compareRegimeDetection(RegimeDetectionProgram fittestRegimeDectionSoFar, RegimeDetectionProgram fittestRegimeDetectionThisround) {

        if (numberOfRegimes > 1) {
            if (fittestRegimeDectionSoFar == null) {
                fittestRegimeDectionSoFar = fittestRegimeDetectionThisround;

            } else {
                Program fittest = AbstractFitnessEvaluator.fittest(fittestRegimeDetectionThisround, fittestRegimeDectionSoFar, selectionStrategy.getDirection());
                if (fittest != null && fittest == fittestRegimeDetectionThisround) {
                    fittestRegimeDectionSoFar = fittestRegimeDetectionThisround;

                }
            }
        }
        return fittestRegimeDectionSoFar;
    }

    protected ResultProducingProgram compareFittest(ResultProducingProgram fittestSoFar, ResultProducingProgram fittestThisround) {
        if (fittestSoFar == null) {
            fittestSoFar = fittestThisround;
        } else {

            Program fittest = AbstractFitnessEvaluator.fittest(fittestThisround, fittestSoFar, selectionStrategy.getDirection());
            if (fittest != null && fittest == fittestThisround) {
                fittestSoFar = fittestThisround;
            }
        }
        return fittestSoFar;
    }

    protected int getPositionParameter(String positionParam) {
        int pos;
        if (positionParam.endsWith("%")) {
            pos = fitnessEvaluator.getTargetSeries().getLength() * Integer.parseInt(positionParam.substring(0, positionParam.length() - 1)) / 100;

        } else {
            pos = Integer.parseInt(positionParam);
        }
        return pos;
    }


}

